"""The data layer: candles, books, funding, open interest and basis, through ccxt.

Everything downstream is only as good as this, so the rules here are strict.

**The in-progress candle is dropped, always.** A `fetch_ohlcv` call returns the bar currently being
formed as its last row. Feeding that to the model means forecasting from a close that has not
happened; scoring against it means grading a forecast on a partial outcome. Both are silent, both
produce plausible numbers, and both are wrong. Every candle this module returns is closed.

**No exchange host is ever taken from user input.** A caller names an exchange by ccxt id and the id
is checked against `ccxt.exchanges`; there is no path by which a request can make this process
connect to an arbitrary URL. That is the same discipline Doxa applies to page fetching, for the same
reason.

**Pagination is explicit.** OKX caps `fetch_ohlcv` at 300 rows per request (verified in
`repos/ccxt_ccxt/python/ccxt/okx.py`, `fetch_ohlcv`: `maxLimit = 100 if isMarkOrIndex else 300`).
Kronos wants 512 of context, so the fetch paginates and then asserts it actually got a contiguous
run — a silently short or gapped series would degrade the forecast invisibly.

**Failures are named.** A source that is down produces `MarketDataError` with what was being fetched
and from where. It never produces an empty list that a caller might read as "no funding rate", which
is the difference between "we could not measure" and "there is nothing there".
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from .markets import TIMEFRAME_SECONDS

# OKX's public API is unreachable on the canonical hostname from some networks while `my.okx.com`
# resolves and serves identically. Verified: `www.okx.com` times out here, `my.okx.com` answers.
_HOSTNAMES = {"okx": "my.okx.com"}

# Per-request row caps, from each exchange's own documented maximum. Asking for more than the
# exchange allows does not error — it silently returns fewer rows, which is how a 512-candle context
# quietly becomes 100.
_OHLCV_MAX_PER_REQUEST = {"okx": 300}
_DEFAULT_OHLCV_MAX = 500

MAX_CANDLES = 5_000          # a paid call cannot be made to pull unbounded history
MAX_ORDERBOOK_DEPTH = 400    # OKX's own ceiling: `request['sz'] = limit  # max 400`


# ccxt exception names that mean "the caller asked for something that does not exist" rather than
# "the venue is having trouble". Matched by name so ccxt stays an optional import.
_CALLER_SYMBOL_ERRORS = frozenset({
    "BadSymbol", "BadRequest", "ArgumentsRequired", "NotSupported",
})


class MarketDataError(RuntimeError):
    """A named data failure. Carries what was being fetched so the caller can act on it."""

    def __init__(self, message: str, *, source: str = "", symbol: str = "", code: str = "source_error"):
        super().__init__(message)
        self.source = source
        self.symbol = symbol
        self.code = code

    def as_dict(self) -> dict:
        return {"error": str(self), "code": self.code, "source": self.source, "symbol": self.symbol}


@dataclass
class Candles:
    """A contiguous run of closed candles."""
    symbol: str
    timeframe: str
    exchange: str
    rows: list[list[float]]          # [ts_ms, open, high, low, close, volume]
    fetched_at: float = field(default_factory=time.time)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def closes(self) -> list[float]:
        return [r[4] for r in self.rows]

    @property
    def highs(self) -> list[float]:
        return [r[2] for r in self.rows]

    @property
    def lows(self) -> list[float]:
        return [r[3] for r in self.rows]

    @property
    def volumes(self) -> list[float]:
        return [r[5] for r in self.rows]

    @property
    def last_close(self) -> float:
        return self.rows[-1][4]

    @property
    def last_ts(self) -> int:
        return int(self.rows[-1][0])

    def log_returns(self, step: int = 1) -> list[float]:
        """Log returns over `step` candles — the input to the empirical baseline."""
        import math
        c = self.closes
        return [math.log(c[i + step] / c[i]) for i in range(len(c) - step) if c[i] > 0]

    def to_dict(self, limit: int | None = None) -> dict:
        rows = self.rows[-limit:] if limit else self.rows
        return {"symbol": self.symbol, "timeframe": self.timeframe, "exchange": self.exchange,
                "count": len(rows),
                "columns": ["timestamp_ms", "open", "high", "low", "close", "volume"],
                "candles": rows}


class _TTLCache:
    """A small time-boxed cache.

    Market data is requested repeatedly within one call — a forecast needs candles, the surprise
    check needs the same candles, the baseline needs them again — and hitting the exchange three
    times for identical data is both slow and rude to a public endpoint. The TTL is short enough
    (20s by default) that no caller is ever served a stale price for a decision.
    """

    def __init__(self, ttl_seconds: float):
        self.ttl = ttl_seconds
        self._d: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any | None:
        with self._lock:
            hit = self._d.get(key)
            if hit and (time.time() - hit[0]) < self.ttl:
                return hit[1]
            if hit:
                self._d.pop(key, None)
            return None

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._d[key] = (time.time(), value)
            if len(self._d) > 512:                     # bounded; oldest out first
                for k, _ in sorted(self._d.items(), key=lambda kv: kv[1][0])[:128]:
                    self._d.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self._d.clear()


class MarketData:
    """One process-wide handle on every exchange we can reach."""

    def __init__(self, cache_ttl: float = 20.0, timeout_ms: int = 20_000):
        self._clients: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._cache = _TTLCache(cache_ttl)
        self.timeout_ms = timeout_ms

    # ── client management ────────────────────────────────────────────────────────────────────

    def exchange(self, exchange_id: str = "okx"):
        """A configured ccxt client, or a named error. Never an arbitrary host."""
        eid = (exchange_id or "okx").strip().lower()
        with self._lock:
            if eid in self._clients:
                return self._clients[eid]
        try:
            import ccxt
        except ImportError as e:                                        # pragma: no cover
            raise MarketDataError(f"ccxt is not installed: {e}", source=eid,
                                  code="dependency_missing") from e
        if eid not in ccxt.exchanges:
            raise MarketDataError(
                f"unknown exchange '{exchange_id}'. Names are ccxt exchange ids; "
                f"{len(ccxt.exchanges)} are supported, for example okx, binance, bybit, coinbase.",
                source=eid, code="unknown_exchange")
        cfg: dict[str, Any] = {"enableRateLimit": True, "timeout": self.timeout_ms}
        if eid in _HOSTNAMES:
            cfg["hostname"] = _HOSTNAMES[eid]
        client = getattr(ccxt, eid)(cfg)
        with self._lock:
            self._clients[eid] = client
        return client

    @staticmethod
    def supported_exchanges() -> list[str]:
        import ccxt
        return list(ccxt.exchanges)

    # ── candles ──────────────────────────────────────────────────────────────────────────────

    def candles(self, symbol: str, timeframe: str = "1h", limit: int = 512,
                exchange: str = "okx", include_open_candle: bool = False,
                until_ms: int | None = None) -> Candles:
        """`limit` **closed** candles, contiguous, most recent last.

        `until_ms` scores a historical window: it returns the candles that had closed as of that
        instant, which is what makes a forecast reproducible after the fact.
        """
        if timeframe not in TIMEFRAME_SECONDS:
            raise MarketDataError(
                f"unsupported timeframe '{timeframe}'. Supported: "
                f"{', '.join(sorted(TIMEFRAME_SECONDS, key=lambda t: TIMEFRAME_SECONDS[t]))}.",
                symbol=symbol, code="bad_timeframe")
        if limit < 1 or limit > MAX_CANDLES:
            raise MarketDataError(f"limit must be between 1 and {MAX_CANDLES}, got {limit}",
                                  symbol=symbol, code="bad_limit")

        key = ("ohlcv", exchange, symbol, timeframe, limit, include_open_candle, until_ms)
        hit = self._cache.get(key)
        if hit is not None:
            return hit

        client = self.exchange(exchange)
        step_ms = TIMEFRAME_SECONDS[timeframe] * 1000
        # One extra row, because the newest is the candle still forming and gets discarded.
        want = limit + (0 if include_open_candle else 1)
        rows = self._fetch_ohlcv_paginated(client, exchange, symbol, timeframe, want, until_ms)

        if not rows:
            raise MarketDataError(
                f"{exchange} returned no candles for {symbol} on {timeframe}. The symbol may not "
                f"exist on that exchange — unified symbols look like 'BTC/USDT' for spot and "
                f"'BTC/USDT:USDT' for a USDT-margined perpetual.",
                source=exchange, symbol=symbol, code="no_data")

        if not include_open_candle:
            now_ms = int(time.time() * 1000) if until_ms is None else until_ms
            # A candle is closed once its *next* boundary has passed.
            rows = [r for r in rows if int(r[0]) + step_ms <= now_ms]
            if not rows:
                raise MarketDataError(
                    f"{exchange} has no closed {timeframe} candle for {symbol} yet.",
                    source=exchange, symbol=symbol, code="no_closed_candle")

        rows = rows[-limit:]
        self._assert_contiguous(rows, step_ms, symbol, timeframe, exchange)
        out = Candles(symbol=symbol, timeframe=timeframe, exchange=exchange, rows=rows)
        self._cache.put(key, out)
        return out

    def _fetch_ohlcv_paginated(self, client, exchange: str, symbol: str, timeframe: str,
                               want: int, until_ms: int | None) -> list[list[float]]:
        """Walk backwards in per-request-sized pages until `want` rows are in hand."""
        per_call = _OHLCV_MAX_PER_REQUEST.get(exchange, _DEFAULT_OHLCV_MAX)
        step_ms = TIMEFRAME_SECONDS[timeframe] * 1000
        end = until_ms if until_ms is not None else int(time.time() * 1000)

        collected: dict[int, list[float]] = {}
        # A generous page budget: enough to satisfy MAX_CANDLES, not enough to loop forever if an
        # exchange keeps returning the same page.
        for _ in range(max(2, (want // per_call) + 2)):
            since = end - step_ms * per_call
            try:
                page = client.fetch_ohlcv(symbol, timeframe, since=since, limit=per_call)
            except Exception as e:                                       # noqa: BLE001
                # ccxt reports "this symbol does not exist" and "the exchange is unreachable" through
                # the same call, and collapsing both into source_error made a typo look like an
                # outage: the caller was told the venue refused them and to try again, for a symbol
                # that will never exist. BadSymbol and friends are the caller's to fix; a network
                # error is ours to report and worth retrying.
                if type(e).__name__ in _CALLER_SYMBOL_ERRORS:
                    raise MarketDataError(
                        f"{exchange} does not list {symbol}. Check the symbol against GET /services, "
                        f"which names every market this service covers.",
                        source=exchange, symbol=symbol, code="unknown_symbol") from e
                raise MarketDataError(
                    f"{exchange} refused the candle request for {symbol} {timeframe}: "
                    f"{type(e).__name__}: {e}",
                    source=exchange, symbol=symbol, code="source_error") from e
            page = [r for r in page if until_ms is None or int(r[0]) < until_ms]
            if not page:
                break
            before = len(collected)
            for r in page:
                collected[int(r[0])] = [int(r[0])] + [float(v) for v in r[1:6]]
            if len(collected) == before:            # the exchange has no more history to give
                break
            if len(collected) >= want:
                break
            end = min(collected)                    # step the window back past the oldest row held

        return [collected[k] for k in sorted(collected)]

    @staticmethod
    def _assert_contiguous(rows: Sequence[Sequence[float]], step_ms: int, symbol: str,
                           timeframe: str, exchange: str) -> None:
        """A gap means the series is not what it claims to be, so say so rather than model it."""
        for a, b in zip(rows, rows[1:]):
            gap = int(b[0]) - int(a[0])
            if gap != step_ms:
                raise MarketDataError(
                    f"{exchange} returned a gapped {timeframe} series for {symbol}: a "
                    f"{gap / 1000:.0f}s step where {step_ms / 1000:.0f}s was expected, at "
                    f"timestamp {int(a[0])}. Refusing to model a discontinuous series.",
                    source=exchange, symbol=symbol, code="gapped_series")

    # ── the outcome, for scoring ─────────────────────────────────────────────────────────────

    def window_outcome(self, symbol: str, timeframe: str, open_iso: str, close_iso: str,
                       exchange: str = "okx") -> dict:
        """What actually happened over a forecast window.

        Returns the open, the close, and the extremes over the whole window — the extremes because
        touch probabilities are questions about the path, and a candle that spiked through a level
        and came back did touch it.
        """
        from datetime import datetime, timezone

        def parse(s: str) -> int:
            return int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
                       .replace(tzinfo=timezone.utc).timestamp() * 1000)

        start_ms, end_ms = parse(open_iso), parse(close_iso)
        step_ms = TIMEFRAME_SECONDS[timeframe] * 1000
        need = max(1, (end_ms - start_ms) // step_ms)
        if int(time.time() * 1000) < end_ms:
            raise MarketDataError(
                f"the window for {symbol} does not close until {close_iso}; there is nothing to "
                f"score yet.", symbol=symbol, code="window_open")

        series = self.candles(symbol, timeframe, limit=int(need), exchange=exchange,
                              until_ms=end_ms)
        rows = [r for r in series.rows if start_ms <= int(r[0]) < end_ms]
        if not rows:
            raise MarketDataError(
                f"{exchange} has no {timeframe} candles for {symbol} between {open_iso} and "
                f"{close_iso}.", source=exchange, symbol=symbol, code="no_outcome_data")
        return {
            "symbol": symbol, "timeframe": timeframe, "exchange": exchange,
            "window": {"open": open_iso, "close": close_iso},
            "candles_used": len(rows),
            "expected_candles": int(need),
            "open": rows[0][1],
            "high": max(r[2] for r in rows),
            "low": min(r[3] for r in rows),
            "close": rows[-1][4],
            "volume": sum(r[5] for r in rows),
            "first_ts": int(rows[0][0]),
            "last_ts": int(rows[-1][0]),
        }

    # ── order book ───────────────────────────────────────────────────────────────────────────

    def orderbook(self, symbol: str, depth: int = 50, exchange: str = "okx") -> dict:
        depth = max(1, min(int(depth), MAX_ORDERBOOK_DEPTH))
        key = ("book", exchange, symbol, depth)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        client = self.exchange(exchange)
        try:
            ob = client.fetch_order_book(symbol, limit=depth)
        except Exception as e:                                           # noqa: BLE001
            raise MarketDataError(f"{exchange} refused the order book for {symbol}: "
                                  f"{type(e).__name__}: {e}",
                                  source=exchange, symbol=symbol, code="source_error") from e
        # Rows are [price, amount] in the ccxt contract but exchanges append their own fields —
        # OKX returns [price, amount, order_count], three elements. Unpacking as a pair raised
        # ValueError on every real OKX book, which on a paid endpoint is a bare 500 for a valid
        # request. Index rather than unpack, so a fourth field tomorrow changes nothing.
        bids = [[float(r[0]), float(r[1])] for r in (ob.get("bids") or [])[:depth]]
        asks = [[float(r[0]), float(r[1])] for r in (ob.get("asks") or [])[:depth]]
        if not bids or not asks:
            raise MarketDataError(f"{exchange} returned a one-sided book for {symbol}",
                                  source=exchange, symbol=symbol, code="empty_book")
        out = {"symbol": symbol, "exchange": exchange, "depth": depth,
               "timestamp_ms": ob.get("timestamp"), "bids": bids, "asks": asks}
        self._cache.put(key, out)
        return out

    # ── perpetual-specific features Kronos cannot see ────────────────────────────────────────

    def funding(self, symbol: str, exchange: str = "okx") -> dict:
        key = ("funding", exchange, symbol)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        client = self.exchange(exchange)
        if not client.has.get("fetchFundingRate"):
            raise MarketDataError(f"{exchange} does not expose funding rates through ccxt",
                                  source=exchange, symbol=symbol, code="unsupported")
        try:
            fr = client.fetch_funding_rate(symbol)
        except Exception as e:                                           # noqa: BLE001
            raise MarketDataError(f"{exchange} refused the funding rate for {symbol}: "
                                  f"{type(e).__name__}: {e}",
                                  source=exchange, symbol=symbol, code="source_error") from e
        self._cache.put(key, fr)
        return fr

    def funding_history(self, symbol: str, limit: int = 200, exchange: str = "okx") -> list[dict]:
        limit = max(1, min(int(limit), 1000))
        key = ("funding_hist", exchange, symbol, limit)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        client = self.exchange(exchange)
        if not client.has.get("fetchFundingRateHistory"):
            raise MarketDataError(f"{exchange} does not expose funding history through ccxt",
                                  source=exchange, symbol=symbol, code="unsupported")
        try:
            rows = client.fetch_funding_rate_history(symbol, limit=limit)
        except Exception as e:                                           # noqa: BLE001
            raise MarketDataError(f"{exchange} refused funding history for {symbol}: "
                                  f"{type(e).__name__}: {e}",
                                  source=exchange, symbol=symbol, code="source_error") from e
        self._cache.put(key, rows)
        return rows

    def open_interest(self, symbol: str, exchange: str = "okx") -> dict:
        key = ("oi", exchange, symbol)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        client = self.exchange(exchange)
        if not client.has.get("fetchOpenInterest"):
            raise MarketDataError(f"{exchange} does not expose open interest through ccxt",
                                  source=exchange, symbol=symbol, code="unsupported")
        try:
            oi = client.fetch_open_interest(symbol)
        except Exception as e:                                           # noqa: BLE001
            raise MarketDataError(f"{exchange} refused open interest for {symbol}: "
                                  f"{type(e).__name__}: {e}",
                                  source=exchange, symbol=symbol, code="source_error") from e
        self._cache.put(key, oi)
        return oi

    def open_interest_history(self, symbol: str, timeframe: str = "1h", limit: int = 200,
                              exchange: str = "okx") -> list[dict]:
        limit = max(1, min(int(limit), 1000))
        key = ("oi_hist", exchange, symbol, timeframe, limit)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        client = self.exchange(exchange)
        if not client.has.get("fetchOpenInterestHistory"):
            raise MarketDataError(f"{exchange} does not expose open-interest history through ccxt",
                                  source=exchange, symbol=symbol, code="unsupported")
        try:
            rows = client.fetch_open_interest_history(symbol, timeframe, limit=limit)
        except Exception as e:                                           # noqa: BLE001
            raise MarketDataError(f"{exchange} refused open-interest history for {symbol}: "
                                  f"{type(e).__name__}: {e}",
                                  source=exchange, symbol=symbol, code="source_error") from e
        self._cache.put(key, rows)
        return rows

    def ticker(self, symbol: str, exchange: str = "okx") -> dict:
        key = ("ticker", exchange, symbol)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        client = self.exchange(exchange)
        try:
            t = client.fetch_ticker(symbol)
        except Exception as e:                                           # noqa: BLE001
            raise MarketDataError(f"{exchange} refused the ticker for {symbol}: "
                                  f"{type(e).__name__}: {e}",
                                  source=exchange, symbol=symbol, code="source_error") from e
        self._cache.put(key, t)
        return t


# One instance for the process. Exchanges hold connection pools and rate-limit state, both of which
# must be shared or the limiter is not actually limiting anything.
_SHARED: MarketData | None = None
_SHARED_LOCK = threading.Lock()


def shared() -> MarketData:
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            from .config import get_settings
            s = get_settings()
            _SHARED = MarketData(cache_ttl=s.cache_ttl_seconds,
                                 timeout_ms=int(s.http_timeout_seconds * 1000))
        return _SHARED
