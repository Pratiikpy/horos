"""What Horos forecasts, on what clock, and why exactly those.

A scorecard is only meaningful per symbol and per horizon. Ten pairs scored properly beat every pair
scored loosely, because the number a buyer needs is "how good is this at *my* market over *my*
horizon" and an average across four hundred instruments answers nobody's question.

So the scored set is small, fixed, and justified from measurements rather than taste. Adding a symbol
means starting its record from zero, in public, which is the correct amount of friction.

**Why these three.** Measured against OKX's own USDT-margined perpetual book on 27 July 2026, all 415
instruments, notional = 24h base volume × last price:

    ETH-USDT-SWAP   $6.996B   43.60%
    BTC-USDT-SWAP   $3.826B   23.85%
    SOL-USDT-SWAP   $0.422B    2.63%
    ────────────────────────────────
    total book     $16.05B

ETH and BTC alone are 67.5% of everything traded. The two instruments ranked between SOL and the rest
— SKHYNIX and SNDK — are tokenised equities, a different asset class with a different clock, so SOL is
the third crypto perpetual by a factor of 1.8 over the next one. There is no fourth candidate that is
close, and inventing one would dilute the record for no gain in coverage.

**Why 1h.** Two constraints meet there.

Kronos-small and Kronos-base take a 512-candle context. At 1h that is 21.3 days — long enough to span
a regime change and see it. At 5m it would be 1.8 days, which is one session and no context at all;
at 1d it would be 1.4 years, most of which is irrelevant to a 24-hour question.

And the record has to accumulate fast enough to mean something. At 1h with four horizons on three
symbols the ledger takes 2,016 scored forecasts a week. At 1d it would take 84, and the scorecard
would still be statistically empty a month after launch. Empirical coverage of a 90% band needs
hundreds of observations before the figure stops being noise.

**Why four horizons.** 1h is the next candle — pure short-horizon noise, and the honest place to show
that the model has almost no edge. 24h is the horizon an agent holding a position actually cares
about. 4h and 12h sit between them so the scorecard can show *where* usefulness starts, which is the
question `forecast.reliability` sells an answer to.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# ── the scored set ───────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Market:
    symbol: str          # ccxt unified symbol
    okx_inst: str        # OKX instrument id, for the raw v5 API
    label: str
    share_of_book: float  # fraction of OKX USDT-perp notional, measured 2026-07-27
    notional_24h_usd: float


SCORED_MARKETS: tuple[Market, ...] = (
    Market("ETH/USDT:USDT", "ETH-USDT-SWAP", "Ether perpetual",  0.4360, 6.996e9),
    Market("BTC/USDT:USDT", "BTC-USDT-SWAP", "Bitcoin perpetual", 0.2385, 3.826e9),
    Market("SOL/USDT:USDT", "SOL-USDT-SWAP", "Solana perpetual",  0.0263, 0.422e9),
)

MEASURED_ON = "2026-07-27"
BOOK_NOTIONAL_24H_USD = 16.05e9
INSTRUMENTS_MEASURED = 415

SCORED_TIMEFRAME = "1h"

# Horizons, in candles of the scored timeframe.
HORIZONS: dict[str, int] = {"1h": 1, "4h": 4, "12h": 12, "24h": 24}

# Kronos context, in candles. 512 for small and base; mini takes 2048. The served checkpoint is
# pinned in the inference module, and this is the number the data layer must be able to supply.
CONTEXT_CANDLES = 512

TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "12h": 43200,
    "1d": 86400, "1w": 604800,
}


def by_symbol(symbol: str) -> Market | None:
    for m in SCORED_MARKETS:
        if m.symbol == symbol or m.okx_inst == symbol:
            return m
    return None


def is_scored(symbol: str) -> bool:
    """Whether this symbol carries a public record.

    Services answer for any symbol ccxt can reach — refusing to price ARB because it is not on the
    scorecard would be worse for the caller, not better. But a forecast on an unscored symbol says so
    in the response, because "we have a public accuracy record" and "we have one for *this*" are
    different claims and conflating them is the dishonesty this whole product is against.
    """
    return by_symbol(symbol) is not None


def context_span(timeframe: str = SCORED_TIMEFRAME, candles: int = CONTEXT_CANDLES) -> timedelta:
    """How far back the model can actually see. 512 hourly candles is 21 days 8 hours."""
    return timedelta(seconds=TIMEFRAME_SECONDS[timeframe] * candles)


def forecasts_per_week(symbols: int = len(SCORED_MARKETS), horizons: int = len(HORIZONS),
                       timeframe: str = SCORED_TIMEFRAME) -> int:
    """How fast the record fills. The reason the timeframe is 1h and not 1d."""
    per_symbol = 7 * 24 * 3600 // TIMEFRAME_SECONDS[timeframe]
    return per_symbol * symbols * horizons


def window_for_anchor(anchor_ts_ms: int, horizon: str,
                      timeframe: str = SCORED_TIMEFRAME) -> dict[str, str]:
    """The candle window a forecast is about, derived from the candle the model actually saw.

    **This is a function of the data, not of the clock, and that distinction is load-bearing.**

    An earlier version computed the window from wall-clock issue time and took "the next boundary
    after now" as the start. That is off by one candle against what the model does. The model is
    given closed candles only, and its first predicted step is the candle *following* the last closed
    one — which is the bar currently in progress. Deriving the window from `now` skipped that bar and
    pointed the window at the one after it, so every forecast would have been scored against a candle
    the model never predicted. Silently, consistently, and in a way that would have made the entire
    record meaningless while looking completely healthy.

    `anchor_ts_ms` is the opening timestamp of the last closed candle, exactly as ccxt reports it.
    From it:

        open  = anchor + one candle        the first bar the model predicts
        close = anchor + (1 + n) candles   the end of the last bar in the horizon

    Both edges land on exchange boundaries, so a published OHLCV bar always exists to score against
    and the scorer never has to interpolate an outcome it could then be argued with about.

    Note that the opening bar is already in progress when the forecast is issued. The model did not
    see any of it — it receives closed candles only — so this is not hindsight. It does mean the
    commitment is made partway into the first bar, which is why the integrity claim is about the
    window's *close* being in the future, and why the scorecard reports anchor time against window
    close for every forecast rather than asserting it in general.
    """
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon {horizon!r}; expected one of {sorted(HORIZONS)}")
    step = TIMEFRAME_SECONDS[timeframe]
    open_s = anchor_ts_ms // 1000 + step
    close_s = open_s + step * HORIZONS[horizon]
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return {
        "open": datetime.fromtimestamp(open_s, timezone.utc).strftime(fmt),
        "close": datetime.fromtimestamp(close_s, timezone.utc).strftime(fmt),
        "timeframe": timeframe,
        "candles": HORIZONS[horizon],
        "anchor_candle_ts": anchor_ts_ms,
    }


def coverage_note() -> dict:
    """The provenance block every scorecard and every forecast response carries.

    Served rather than written into the page, so the justification for the served set cannot drift
    away from the served set itself.
    """
    return {
        "symbols": [{"symbol": m.symbol, "okx_instrument": m.okx_inst, "label": m.label,
                     "share_of_okx_usdt_perp_notional": round(m.share_of_book, 4),
                     "notional_24h_usd": m.notional_24h_usd} for m in SCORED_MARKETS],
        "timeframe": SCORED_TIMEFRAME,
        "horizons": sorted(HORIZONS, key=lambda h: HORIZONS[h]),
        "context_candles": CONTEXT_CANDLES,
        "context_span": str(context_span()),
        "scored_forecasts_per_week": forecasts_per_week(),
        "selection": {
            "basis": "24h notional on OKX USDT-margined perpetuals, all instruments, "
                     "24h base volume x last price",
            "measured_on": MEASURED_ON,
            "instruments_measured": INSTRUMENTS_MEASURED,
            "book_notional_24h_usd": BOOK_NOTIONAL_24H_USD,
            "note": "ETH and BTC are 67.5% of the book between them. The instruments ranked between "
                    "SOL and the rest are tokenised equities, a different asset class; SOL is the "
                    "third crypto perpetual by 1.8x over the next one.",
        },
    }
