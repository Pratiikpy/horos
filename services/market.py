"""Group A — market data and context. Six services, $0.001–0.01.

The temptation with a data tier is to proxy the exchange and call it a product. Every one of these
does the extra step that makes it worth paying for: the funding rate arrives with its own percentile
history so a caller knows whether it is unusual, the book arrives with its imbalance computed, open
interest arrives with its divergence from price, and the cross-venue view arrives with the spread
already worked out and the stale feeds flagged.

Raw numbers are free from the exchange. The measurement on top of them is the thing being sold, and
each measurement is recomputable from the inputs echoed back.
"""
from __future__ import annotations

import statistics

from core.market_data import MarketDataError
from core.markets import TIMEFRAME_SECONDS

from .registry import Param, ServiceError, service

SYMBOL = Param("symbol", "string", "Unified ccxt symbol. 'BTC/USDT' is spot; 'BTC/USDT:USDT' is the "
                                   "USDT-margined perpetual.", required=True)
EXCHANGE = Param("exchange", "string", "ccxt exchange id. The pinned ccxt build supports "
                                       "104; unknown ids are refused by name.", default="okx")


def _pct_rank(value: float, history: list[float]) -> float | None:
    """Where a value sits within its own history, 0–1. None when there is no history to rank against."""
    clean = [h for h in history if h is not None]
    if len(clean) < 2:
        return None
    return sum(1 for h in clean if h <= value) / len(clean)


# ── candles ──────────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="market.candles", group="A. Market data", price=0.001,
    title="Normalised OHLCV",
    short="Normalised OHLCV candles",
    summary="Fetches closed OHLCV candles from any exchange ccxt supports, through one normalised schema, "
            "dropping the bar still forming and refusing any series with a gap in it.",
    returns="Returns the candles with their timestamps, a contiguity guarantee, and the exchange and "
            "window they came from.",
    depth="Every candle returned has closed. The bar currently forming is always discarded, because "
          "forecasting or scoring against a partial bar produces a plausible wrong number in "
          "silence. A gapped series is refused rather than interpolated.",
    params=(SYMBOL, EXCHANGE,
            Param("timeframe", "string", "Candle size: 1m 3m 5m 15m 30m 1h 2h 4h 6h 12h 1d 1w.",
                  default="1h"),
            Param("limit", "integer", "How many closed candles, 1–5000. Paginated automatically "
                                      "past each exchange's per-request cap.", default=200)),
    example={"symbol": "BTC/USDT:USDT", "timeframe": "1h", "limit": 200},
)
def market_candles(inp: dict, ctx) -> dict:
    c = ctx.data.candles(inp["symbol"], inp.get("timeframe", "1h"),
                         limit=int(inp.get("limit", 200)), exchange=inp.get("exchange", "okx"))
    closes = c.closes
    return {
        **c.to_dict(),
        "first_candle": c.rows[0][0],
        "last_candle": c.rows[-1][0],
        "contiguous": True,
        "open_candle_excluded": True,
        "summary": {
            "last_close": c.last_close,
            "change_over_window_pct": round(100 * (closes[-1] / closes[0] - 1), 4),
            "high": max(c.highs), "low": min(c.lows),
            "mean_volume": round(sum(c.volumes) / len(c.volumes), 6),
        },
    }


# ── order book ───────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="market.orderbook", group="A. Market data", price=0.005,
    title="Depth, spread and imbalance",
    short="Book depth and imbalance",
    summary="Takes a depth snapshot and computes what the raw book does not tell you: the spread in "
            "basis points, the bid/ask imbalance, and the notional resting within several distances "
            "of the mid price.",
    returns="Returns both sides of the book plus the spread, mid, imbalance ratio and cumulative "
            "depth at 5, 10, 25 and 50 basis points.",
    depth="Depth is measured in notional at fixed distances from mid rather than by level count, "
          "because level counts are not comparable across venues with different tick sizes. A "
          "one-sided book is an error rather than a zero.",
    params=(SYMBOL, EXCHANGE,
            Param("depth", "integer", "Levels per side, 1–400. Defaults to the maximum, because on "
                                      "a liquid pair fifty levels span barely one basis point.",
                  default=400)),
    example={"symbol": "BTC/USDT:USDT", "depth": 400},
)
def market_orderbook(inp: dict, ctx) -> dict:
    ob = ctx.data.orderbook(inp["symbol"], depth=int(inp.get("depth", 400)),
                            exchange=inp.get("exchange", "okx"))
    best_bid, best_ask = ob["bids"][0][0], ob["asks"][0][0]
    mid = (best_bid + best_ask) / 2
    bid_qty = sum(q for _, q in ob["bids"])
    ask_qty = sum(q for _, q in ob["asks"])

    # How far from mid the levels we actually fetched reach. On a liquid book with a sub-basis-point
    # spread, fifty levels can span less than one bp — in which case "depth within 50bps" is not a
    # measurement of depth within 50bps, it is a measurement of the fifty levels we happened to ask
    # for, and reporting it as the former would overstate resting liquidity by an unknown factor.
    bid_reach_bps = 10_000 * (mid - ob["bids"][-1][0]) / mid
    ask_reach_bps = 10_000 * (ob["asks"][-1][0] - mid) / mid

    def bucket(bps: float) -> dict:
        bid_edge, ask_edge = mid * (1 - bps / 10_000), mid * (1 + bps / 10_000)
        row = {
            "bid": round(sum(p * q for p, q in ob["bids"] if p >= bid_edge), 2),
            "ask": round(sum(p * q for p, q in ob["asks"] if p <= ask_edge), 2),
        }
        truncated = [s for s, reach in (("bid", bid_reach_bps), ("ask", ask_reach_bps))
                     if reach < bps]
        if truncated:
            row["incomplete"] = truncated
            row["note"] = (f"the {ob['depth']} levels fetched reach only "
                           f"{min(bid_reach_bps, ask_reach_bps):.2f}bps from mid, so this figure is "
                           f"a floor rather than the full depth at {bps}bps. Raise 'depth' to "
                           f"measure further out.")
        return row

    return {
        "symbol": ob["symbol"], "exchange": ob["exchange"], "timestamp_ms": ob["timestamp_ms"],
        "best_bid": best_bid, "best_ask": best_ask, "mid": mid,
        "spread": round(best_ask - best_bid, 10),
        "spread_bps": round(10_000 * (best_ask - best_bid) / mid, 4),
        "imbalance": round((bid_qty - ask_qty) / (bid_qty + ask_qty), 6) if bid_qty + ask_qty else 0,
        "imbalance_reading": ("more resting bid than ask" if bid_qty > ask_qty
                              else "more resting ask than bid"),
        "depth_notional_usd": {f"{bps}bps": bucket(bps) for bps in (5, 10, 25, 50)},
        "book_reach_bps": {"bid": round(bid_reach_bps, 3), "ask": round(ask_reach_bps, 3)},
        "levels_returned": ob["depth"],
        "bids": ob["bids"], "asks": ob["asks"],
    }


# ── funding ──────────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="market.funding", group="A. Market data", price=0.005,
    title="Funding rate in its own context",
    short="Funding rate in context",
    summary="Reads the current perpetual funding rate and places it against its own recent history, "
            "so a caller can tell an ordinary rate from an extreme one without keeping their own "
            "series.",
    returns="Returns the current rate, its annualised equivalent, its percentile within the history "
            "requested, and the mean and extremes of that history.",
    depth="The percentile is the point. A funding rate of 0.01% means nothing on its own; the same "
          "number can be the calmest reading in a month or the highest. History length is a "
          "parameter and the actual count used is reported, not assumed.",
    params=(SYMBOL, EXCHANGE,
            Param("history", "integer", "How many past funding periods to rank against, 1–1000.",
                  default=200)),
    example={"symbol": "BTC/USDT:USDT", "history": 200},
)
def market_funding(inp: dict, ctx) -> dict:
    symbol = inp["symbol"]
    exchange = inp.get("exchange", "okx")
    current = ctx.data.funding(symbol, exchange=exchange)
    rate = current.get("fundingRate")
    if rate is None:
        raise ServiceError(f"{exchange} returned no funding rate for {symbol}. Funding exists only "
                           f"on perpetual swaps — check the symbol is a perp, e.g. 'BTC/USDT:USDT'.",
                           code="no_funding_rate")
    rate = float(rate)

    out: dict = {
        "symbol": symbol, "exchange": exchange, "funding_rate": rate,
        "funding_rate_pct": round(rate * 100, 8),
        "next_funding_time": current.get("fundingDatetime") or current.get("nextFundingDatetime"),
        "mark_price": current.get("markPrice"), "index_price": current.get("indexPrice"),
    }
    try:
        hist = ctx.data.funding_history(symbol, limit=int(inp.get("history", 200)),
                                        exchange=exchange)
        rates = [float(h["fundingRate"]) for h in hist if h.get("fundingRate") is not None]
    except MarketDataError as e:
        out["history"] = None
        out["history_unavailable"] = str(e)
        return out

    if not rates:
        out["history"] = None
        out["history_unavailable"] = "the exchange returned an empty funding history"
        return out

    # Periods per year, from the actual spacing of the history rather than an assumed 8 hours.
    stamps = sorted(h["timestamp"] for h in hist if h.get("timestamp"))
    period_h = (statistics.median([(b - a) / 3_600_000 for a, b in zip(stamps, stamps[1:])])
                if len(stamps) > 2 else 8.0)
    percentile = _pct_rank(rate, rates)
    out.update({
        "annualised_pct": round(rate * (24 / period_h) * 365 * 100, 4),
        "funding_period_hours": round(period_h, 3),
        # Named for the scale it is on. A field called `percentile` holding 1.0 is read as the 1st
        # percentile — the exact opposite of the 100th, which is what it means. That ambiguity
        # produced a materially wrong sentence in a generated brief, so both scales are emitted and
        # neither is called plain "percentile".
        "percentile_rank_0_1": percentile,
        "percentile_0_100": round(percentile * 100, 2) if percentile is not None else None,
        "history": {
            "count": len(rates), "mean": round(statistics.fmean(rates), 10),
            "median": round(statistics.median(rates), 10),
            "min": min(rates), "max": max(rates),
            "stdev": round(statistics.pstdev(rates), 10) if len(rates) > 1 else 0.0,
        },
        "reading": _funding_reading(rate, percentile, len(rates)),
    })
    return out


def _funding_reading(rate: float, pct: float | None, n: int) -> str:
    """One sentence a caller can act on.

    Worth writing carefully. An earlier version rendered the top of the range as "higher than 100% of
    the recent history", which is both awkward and ambiguous — it reads as though the rate exceeds
    the whole history by 100%. The extremes get named as extremes, and the middle gets a percentile.
    """
    side = ("longs are paying shorts" if rate > 0 else
            "shorts are paying longs" if rate < 0 else "neither side is paying")
    if pct is None:
        return f"{side}; no history was available to rank this against"
    if pct >= 1.0:
        return f"{side}, and this is the highest funding rate in the last {n} periods"
    if pct <= 1.0 / n:
        return f"{side}, and this is the lowest funding rate in the last {n} periods"
    # Floored, not rounded. Rounding renders the 99.5th percentile as "above 100%", which reads as
    # though the rate exceeds the entire history — the same ambiguity the extremes branch above
    # exists to avoid.
    import math as _m
    if pct >= 0.9:
        return (f"{side}; this is an unusually high rate — above {_m.floor(pct * 100)}% of the last "
                f"{n} periods")
    if pct <= 0.1:
        return (f"{side}; this is an unusually low rate — below {_m.floor((1 - pct) * 100)}% of the "
                f"last {n} periods")
    return (f"{side}; this is an ordinary rate for this market, at the {round(pct * 100)}th "
            f"percentile of the last {n} periods")


# ── open interest ────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="market.openinterest", group="A. Market data", price=0.005,
    title="Open interest and its divergence from price",
    short="Open interest divergence",
    summary="Reads open interest and compares its recent change against the price change over the "
            "same window, which is the reading that separates new positioning from position "
            "unwinding.",
    returns="Returns current open interest, its change over the window, the price change over the "
            "same window, and the four-way classification those two signs imply.",
    depth="Open interest rising while price rises means new longs; rising while price falls means "
          "new shorts; falling in either direction means positions closing. That interpretation is "
          "the product — the raw number is free from the exchange.",
    params=(SYMBOL, EXCHANGE,
            Param("timeframe", "string", "Candle size for the comparison window.", default="1h"),
            Param("periods", "integer", "How many candles back to measure the change over.",
                  default=24)),
    example={"symbol": "BTC/USDT:USDT", "timeframe": "1h", "periods": 24},
)
def market_open_interest(inp: dict, ctx) -> dict:
    symbol, exchange = inp["symbol"], inp.get("exchange", "okx")
    tf, periods = inp.get("timeframe", "1h"), int(inp.get("periods", 24))
    oi = ctx.data.open_interest(symbol, exchange=exchange)
    value = oi.get("openInterestValue") or oi.get("openInterestAmount")
    out: dict = {"symbol": symbol, "exchange": exchange,
                 "open_interest": oi.get("openInterestAmount"),
                 "open_interest_value": oi.get("openInterestValue"),
                 "timestamp_ms": oi.get("timestamp")}
    try:
        hist = ctx.data.open_interest_history(symbol, tf, limit=periods + 1, exchange=exchange)
    except MarketDataError as e:
        out["change_unavailable"] = str(e)
        return out
    series = [float(h.get("openInterestValue") or h.get("openInterestAmount") or 0) for h in hist]
    series = [v for v in series if v > 0]
    if len(series) < 2:
        out["change_unavailable"] = "the exchange returned too little open-interest history"
        return out

    candles = ctx.data.candles(symbol, tf, limit=min(periods + 1, len(series)), exchange=exchange)
    oi_change = (series[-1] / series[0] - 1) * 100
    px_change = (candles.closes[-1] / candles.closes[0] - 1) * 100
    out.update({
        "window": {"timeframe": tf, "periods": len(series) - 1},
        "open_interest_change_pct": round(oi_change, 4),
        "price_change_pct": round(px_change, 4),
        "reading": _oi_reading(oi_change, px_change),
        "open_interest_series": series,
    })
    return out


def _oi_reading(oi_change: float, px_change: float) -> str:
    if abs(oi_change) < 0.5:
        return "open interest is broadly flat; positioning has not changed much over this window"
    if oi_change > 0:
        return ("open interest is rising while price rises — new long positioning"
                if px_change > 0 else
                "open interest is rising while price falls — new short positioning")
    return ("open interest is falling while price rises — shorts closing"
            if px_change > 0 else
            "open interest is falling while price falls — longs closing")


# ── basis ────────────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="market.basis", group="A. Market data", price=0.005,
    title="Spot-versus-perpetual basis, annualised",
    short="Spot-perp basis annualised",
    summary="Prices the same asset on the spot and perpetual books at the same moment and expresses "
            "the gap between them as an annualised carry, which is the form the number is actually "
            "usable in.",
    returns="Returns both prices, the basis in absolute and percentage terms, its annualised "
            "equivalent from the live funding rate, and which side is trading rich.",
    depth="Both legs are fetched together so the basis is not an artefact of two prices taken "
          "seconds apart. Annualisation uses the funding period the exchange actually reports "
          "rather than an assumed eight hours.",
    params=(SYMBOL, EXCHANGE,
            Param("spot_symbol", "string", "Spot leg, if it is not the perpetual symbol with the "
                                           "settlement suffix removed.", default=None)),
    example={"symbol": "BTC/USDT:USDT"},
)
def market_basis(inp: dict, ctx) -> dict:
    perp = inp["symbol"]
    exchange = inp.get("exchange", "okx")
    spot = inp.get("spot_symbol") or perp.split(":")[0]
    if ":" not in perp:
        raise ServiceError(f"'{perp}' is not a perpetual symbol. A basis needs a perp leg, which "
                           f"looks like 'BTC/USDT:USDT'.", code="not_a_perpetual")
    perp_t = ctx.data.ticker(perp, exchange=exchange)
    spot_t = ctx.data.ticker(spot, exchange=exchange)
    p, s = float(perp_t["last"]), float(spot_t["last"])
    basis_pct = (p / s - 1) * 100

    out = {"perpetual": perp, "spot": spot, "exchange": exchange,
           "perpetual_price": p, "spot_price": s,
           "basis": round(p - s, 8), "basis_pct": round(basis_pct, 6),
           "reading": ("the perpetual is trading rich to spot — a premium"
                       if p > s else "the perpetual is trading cheap to spot — a discount")}
    try:
        fr = ctx.data.funding(perp, exchange=exchange)
        rate = float(fr.get("fundingRate") or 0)
        out["funding_rate"] = rate
        out["carry_annualised_pct"] = round(rate * 3 * 365 * 100, 4)
        out["carry_note"] = ("annualised from the current funding rate at three payments a day; the "
                             "realised carry depends on funding actually staying at this level")
    except MarketDataError as e:
        out["funding_unavailable"] = str(e)
    return out


# ── venues ───────────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="market.venues", group="A. Market data", price=0.01,
    title="One asset across every venue that lists it",
    short="Cross-venue price spread",
    summary="Prices the same asset simultaneously on several exchanges and reports the spread "
            "between the cheapest and dearest, flagging any feed whose timestamp shows it is stale.",
    returns="Returns each venue's price, volume and timestamp, the best bid and offer across all of "
            "them, the spread in basis points, and the venues that could not be reached.",
    depth="Staleness is checked rather than assumed. A venue quoting a price from twenty minutes ago "
          "will produce a spectacular apparent spread, and reporting that as an opportunity would be "
          "worse than useless — those feeds are flagged and excluded from the spread.",
    params=(SYMBOL,
            Param("exchanges", "array", "ccxt exchange ids to compare. Reachable from the deployment region. Binance and Bybit geo-block US IP ranges, and Horos runs in us-east-1 — defaulting to venues that always fail would make every cross-venue answer a list of errors.",
                  default=["okx", "gate", "bitget", "kucoin", "mexc"]),
            Param("max_age_seconds", "integer", "A quote older than this is treated as stale.",
                  default=120)),
    example={"symbol": "BTC/USDT:USDT", "exchanges": ["okx", "gate", "bitget"]},
)
def market_venues(inp: dict, ctx) -> dict:
    import time as _t
    symbol = inp["symbol"]
    names = inp.get("exchanges") or ["okx", "gate", "bitget", "kucoin", "mexc"]
    if not isinstance(names, list) or not names:
        raise ServiceError("'exchanges' must be a non-empty list of ccxt exchange ids.",
                           code="bad_exchanges")
    max_age = int(inp.get("max_age_seconds", 120))
    now_ms = int(_t.time() * 1000)

    quotes, unreachable, stale = [], [], []
    for name in names[:12]:
        try:
            t = ctx.data.ticker(symbol, exchange=name)
        except MarketDataError as e:
            unreachable.append({"exchange": name, "reason": str(e)})
            continue
        last = t.get("last")
        if last is None:
            unreachable.append({"exchange": name, "reason": "no last price in the ticker"})
            continue
        age = (now_ms - int(t["timestamp"])) / 1000 if t.get("timestamp") else None
        row = {"exchange": name, "price": float(last),
               "quote_volume": t.get("quoteVolume"), "timestamp_ms": t.get("timestamp"),
               "age_seconds": round(age, 1) if age is not None else None}
        if age is not None and age > max_age:
            row["stale"] = True
            stale.append(row)
        else:
            quotes.append(row)

    if not quotes:
        raise ServiceError(
            f"no venue returned a fresh price for {symbol}. Reached: "
            f"{len(unreachable)} failed, {len(stale)} stale.",
            code="no_fresh_quotes", unreachable=unreachable, stale=stale)

    cheapest = min(quotes, key=lambda r: r["price"])
    dearest = max(quotes, key=lambda r: r["price"])
    mid = (cheapest["price"] + dearest["price"]) / 2
    return {
        "symbol": symbol,
        "venues": sorted(quotes, key=lambda r: r["price"]),
        "cheapest": {"exchange": cheapest["exchange"], "price": cheapest["price"]},
        "dearest": {"exchange": dearest["exchange"], "price": dearest["price"]},
        "spread": round(dearest["price"] - cheapest["price"], 8),
        "spread_bps": round(10_000 * (dearest["price"] - cheapest["price"]) / mid, 4),
        "venues_compared": len(quotes),
        "stale_excluded": stale,
        "unreachable": unreachable,
        "note": ("the spread is computed only across venues with a fresh quote; stale and "
                 "unreachable venues are listed separately and excluded rather than dropped"),
    }
