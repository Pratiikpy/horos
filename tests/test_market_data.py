"""The data layer's failure modes, which are all silent by default.

A short candle series, a gapped one, or one that includes the bar still being formed all look
completely normal downstream — you get a number, it is plausible, and it is wrong. So each of those
is forced here against a fake exchange whose behaviour is known exactly, rather than hoped about
against a live one.

The live-marked tests at the bottom hit OKX for real. They are excluded from the offline suite with
`-m "not live"` and run in the full one, because a data layer that has only ever been tested against
a fake is a data layer that has never been tested.
"""
from __future__ import annotations

import time

import pytest

from core.market_data import MAX_CANDLES, Candles, MarketData, MarketDataError

HOUR_MS = 3_600_000


class FakeExchange:
    """A ccxt-shaped exchange with exactly the behaviour each test needs."""

    def __init__(self, rows=None, has=None, raises=None, per_call=300):
        self.rows = rows or []
        self.has = has if has is not None else {"fetchFundingRate": True, "fetchOpenInterest": True,
                                                "fetchFundingRateHistory": True,
                                                "fetchOpenInterestHistory": True}
        self.raises = raises
        self.per_call = per_call
        self.calls = 0

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls += 1
        if self.raises:
            raise self.raises
        window = [r for r in self.rows if since is None or r[0] >= since]
        return window[: (limit or self.per_call)]

    def fetch_order_book(self, symbol, limit=None):
        if self.raises:
            raise self.raises
        return {"bids": [[100.0, 1.0]] * (limit or 50), "asks": [[101.0, 1.0]] * (limit or 50),
                "timestamp": 1}

    def fetch_funding_rate(self, symbol):
        if self.raises:
            raise self.raises
        return {"symbol": symbol, "fundingRate": 0.0001}

    def fetch_ticker(self, symbol):
        if self.raises:
            raise self.raises
        return {"symbol": symbol, "last": 100.0}


def series(n: int, end_ms: int | None = None, step: int = HOUR_MS, start_price: float = 100.0):
    """`n` hourly candles ending at `end_ms` (exclusive of the bar in progress)."""
    end_ms = end_ms if end_ms is not None else int(time.time() * 1000)
    last_open = (end_ms // step) * step
    return [[last_open - step * (n - 1 - i), start_price + i, start_price + i + 1,
             start_price + i - 1, start_price + i + 0.5, 10.0 + i] for i in range(n)]


def md_with(exchange: FakeExchange, ttl: float = 0.0) -> MarketData:
    m = MarketData(cache_ttl=ttl)
    m._clients["okx"] = exchange       # bypass the real client factory, not the logic under test
    return m


# ── the in-progress candle ───────────────────────────────────────────────────────────────────────

def test_the_candle_still_being_formed_is_dropped():
    """The default failure this module exists to prevent.

    `fetch_ohlcv` returns the bar currently forming as its last row. Its close has not happened.
    Forecasting from it, or scoring against it, produces a plausible wrong number in silence.
    """
    now = int(time.time() * 1000)
    rows = series(10, end_ms=now)
    live_bar_open = rows[-1][0]
    out = md_with(FakeExchange(rows)).candles("BTC/USDT:USDT", "1h", limit=5)
    assert live_bar_open not in [r[0] for r in out.rows]
    assert out.rows[-1][0] + HOUR_MS <= now


def test_the_in_progress_candle_is_returned_when_explicitly_asked_for():
    now = int(time.time() * 1000)
    rows = series(10, end_ms=now)
    out = md_with(FakeExchange(rows)).candles("BTC/USDT:USDT", "1h", limit=10,
                                              include_open_candle=True)
    assert rows[-1][0] == out.rows[-1][0]


def test_a_market_with_no_closed_candle_yet_is_an_error_not_an_empty_list():
    now = int(time.time() * 1000)
    only_the_live_bar = [[(now // HOUR_MS) * HOUR_MS, 1, 2, 0.5, 1.5, 3.0]]
    with pytest.raises(MarketDataError) as e:
        md_with(FakeExchange(only_the_live_bar)).candles("BTC/USDT:USDT", "1h", limit=1)
    assert e.value.code == "no_closed_candle"


# ── contiguity ───────────────────────────────────────────────────────────────────────────────────

def test_a_gapped_series_is_refused_rather_than_modelled():
    """A hole in the history is not something to interpolate over; it is something to report."""
    rows = series(20)
    del rows[10]
    with pytest.raises(MarketDataError) as e:
        md_with(FakeExchange(rows)).candles("BTC/USDT:USDT", "1h", limit=19)
    assert e.value.code == "gapped_series"
    assert "gapped" in str(e.value)


def test_a_contiguous_series_passes():
    out = md_with(FakeExchange(series(600))).candles("BTC/USDT:USDT", "1h", limit=512)
    assert len(out) == 512
    steps = {out.rows[i + 1][0] - out.rows[i][0] for i in range(len(out) - 1)}
    assert steps == {HOUR_MS}


# ── pagination ───────────────────────────────────────────────────────────────────────────────────

def test_512_candles_are_assembled_from_pages_smaller_than_512():
    """OKX caps a single request at 300 rows. Kronos wants 512 of context.

    Verified in repos/ccxt_ccxt/python/ccxt/okx.py: `maxLimit = 100 if isMarkOrIndex else 300`.
    Without pagination a 512-candle request silently returns 300 and the model sees 40% less history
    than the response says it did.
    """
    ex = FakeExchange(series(900), per_call=300)
    out = md_with(ex).candles("BTC/USDT:USDT", "1h", limit=512)
    assert len(out) == 512
    assert ex.calls > 1


def test_pagination_stops_when_the_exchange_has_no_more_history():
    """Requesting more than exists must terminate, not loop."""
    ex = FakeExchange(series(50), per_call=300)
    out = md_with(ex).candles("BTC/USDT:USDT", "1h", limit=200)
    assert 0 < len(out) <= 50
    assert ex.calls < 10


# ── input validation ─────────────────────────────────────────────────────────────────────────────

def test_an_unsupported_timeframe_lists_the_supported_ones():
    with pytest.raises(MarketDataError) as e:
        md_with(FakeExchange(series(10))).candles("BTC/USDT:USDT", "7h")
    assert e.value.code == "bad_timeframe"
    assert "1h" in str(e.value)


def test_an_unbounded_history_request_is_capped():
    with pytest.raises(MarketDataError) as e:
        md_with(FakeExchange(series(10))).candles("BTC/USDT:USDT", "1h", limit=MAX_CANDLES + 1)
    assert e.value.code == "bad_limit"


def test_an_unknown_exchange_is_named_and_never_becomes_a_url():
    """The SSRF guard: exchange ids are checked against ccxt's own list, never used as a host."""
    with pytest.raises(MarketDataError) as e:
        MarketData().exchange("http://evil.example/")
    assert e.value.code == "unknown_exchange"
    assert "ccxt exchange ids" in str(e.value)


def test_a_source_failure_is_named_with_what_was_being_fetched():
    """Never a bare 500. OKX rejected an agent for exactly that — see OKX-REVIEW-RULES §1.6."""
    ex = FakeExchange(raises=ConnectionError("connection reset"))
    with pytest.raises(MarketDataError) as e:
        md_with(ex).candles("BTC/USDT:USDT", "1h", limit=10)
    assert e.value.code == "source_error"
    assert "BTC/USDT:USDT" in str(e.value)
    assert e.value.as_dict()["symbol"] == "BTC/USDT:USDT"


def test_an_empty_response_explains_the_symbol_format():
    """The most likely caller mistake is 'BTC/USDT' when they wanted the perpetual."""
    with pytest.raises(MarketDataError) as e:
        md_with(FakeExchange([])).candles("NOPE/USDT", "1h", limit=5)
    assert "BTC/USDT:USDT" in str(e.value)


def test_a_missing_capability_is_reported_as_unsupported_not_as_absence():
    """ccxt has no fetchLiquidations for OKX. That is 'we cannot see it', not 'there were none'."""
    ex = FakeExchange(series(10), has={"fetchFundingRate": False})
    with pytest.raises(MarketDataError) as e:
        md_with(ex).funding("BTC/USDT:USDT")
    assert e.value.code == "unsupported"


# ── caching ──────────────────────────────────────────────────────────────────────────────────────

def test_the_cache_saves_a_repeat_fetch_inside_its_ttl():
    ex = FakeExchange(series(600))
    md = md_with(ex, ttl=60.0)
    md.candles("BTC/USDT:USDT", "1h", limit=100)
    calls_after_first = ex.calls
    md.candles("BTC/USDT:USDT", "1h", limit=100)
    assert ex.calls == calls_after_first


def test_the_cache_does_not_serve_a_different_symbol_or_timeframe():
    ex = FakeExchange(series(600))
    md = md_with(ex, ttl=60.0)
    md.candles("BTC/USDT:USDT", "1h", limit=100)
    first = ex.calls
    md.candles("ETH/USDT:USDT", "1h", limit=100)
    assert ex.calls > first


# ── outcomes ─────────────────────────────────────────────────────────────────────────────────────

def test_the_outcome_uses_the_extremes_over_the_whole_window_not_just_the_close():
    """Touch questions are about the path. A wick through a level counts."""
    end = 1_785_150_000_000                       # 2026-07-27T11:00:00Z
    rows = series(6, end_ms=end)                  # opens 06:00 … 11:00
    by_hour = {r[0]: r for r in rows}
    at = lambda h: by_hour[end - HOUR_MS * (11 - h)]   # noqa: E731
    at(8)[2] = 999.0                              # a spike mid-window
    at(9)[3] = 1.0                                # and a flush
    md = md_with(FakeExchange(rows))
    # A window of [06:00, 10:00) is the four candles opening at 06, 07, 08 and 09.
    out = md.window_outcome("BTC/USDT:USDT", "1h", "2026-07-27T06:00:00Z", "2026-07-27T10:00:00Z")
    assert out["candles_used"] == 4
    assert out["high"] == 999.0
    assert out["low"] == 1.0
    assert out["close"] == at(9)[4]               # the close of the last candle *inside* the window
    assert out["open"] == at(6)[1]


def test_scoring_a_window_that_has_not_closed_is_refused():
    """The single most important guard in the whole system: no score before the outcome exists."""
    md = md_with(FakeExchange(series(10)))
    with pytest.raises(MarketDataError) as e:
        md.window_outcome("BTC/USDT:USDT", "1h", "2099-01-01T00:00:00Z", "2099-01-02T00:00:00Z")
    assert e.value.code == "window_open"


# ── derived helpers ──────────────────────────────────────────────────────────────────────────────

def test_log_returns_are_over_the_requested_step():
    import math
    c = Candles("X", "1h", "okx", [[0, 1, 1, 1, 100.0, 1], [1, 1, 1, 1, 110.0, 1],
                                   [2, 1, 1, 1, 121.0, 1]])
    assert c.log_returns(step=1)[0] == pytest.approx(math.log(1.1))
    assert c.log_returns(step=2)[0] == pytest.approx(math.log(1.21))


def test_orderbook_depth_is_capped_at_the_exchange_maximum():
    out = md_with(FakeExchange(series(2))).orderbook("BTC/USDT:USDT", depth=10_000)
    assert out["depth"] == 400


def test_a_one_sided_book_is_an_error():
    class OneSided(FakeExchange):
        def fetch_order_book(self, symbol, limit=None):
            return {"bids": [[100.0, 1.0]], "asks": [], "timestamp": 1}
    with pytest.raises(MarketDataError) as e:
        md_with(OneSided(series(2))).orderbook("BTC/USDT:USDT")
    assert e.value.code == "empty_book"


# ── against the real exchange ────────────────────────────────────────────────────────────────────

@pytest.mark.live
def test_live_okx_delivers_512_contiguous_closed_candles():
    out = MarketData().candles("BTC/USDT:USDT", "1h", limit=512)
    assert len(out) == 512
    assert {out.rows[i + 1][0] - out.rows[i][0] for i in range(511)} == {HOUR_MS}
    assert out.last_ts + HOUR_MS <= int(time.time() * 1000)
    assert out.last_close > 0


@pytest.mark.live
def test_live_okx_exposes_funding_and_open_interest():
    """The features Kronos cannot see. Verified present rather than assumed."""
    md = MarketData()
    assert "fundingRate" in md.funding("BTC/USDT:USDT")
    oi = md.open_interest("BTC/USDT:USDT")
    assert oi.get("openInterestAmount") or oi.get("openInterestValue")


@pytest.mark.live
def test_live_okx_has_no_liquidation_feed_which_is_why_we_do_not_sell_one():
    """Pinned so the honest scope note in the PRD cannot quietly become false."""
    import ccxt
    assert not ccxt.okx().has.get("fetchLiquidations")
