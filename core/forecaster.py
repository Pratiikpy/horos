"""Turning a model fan into a committed forecast.

This is the seam between the part that guesses and the part that keeps score. Everything above it is
statistical; everything below it is a promise that has been written down and hashed.

The one structural decision here is that a scheduled round generates **once per symbol** at the
longest horizon and slices, rather than generating four times. That is a four-fold saving, but the
reason it is right is correctness: four independent generations produce horizons that contradict
each other, because nothing ties the draws together. A 24-candle path set sliced at 1, 4, 12 and 24
gives four marginals of one joint distribution, so the 24h band contains the 4h band by construction.

What gets recorded is the fan itself — every path's terminal price, high and low — not a summary.
A commitment to a summary would let the summary be re-derived differently later; a commitment to the
draws cannot be argued with.
"""
from __future__ import annotations

import logging
from typing import Any

from .kronos_runner import LIVE_PATHS, SCHEDULED_PATHS, Fan, ForecastError, KronosRunner
from .market_data import MarketData
from .markets import CONTEXT_CANDLES, HORIZONS, SCORED_TIMEFRAME, is_scored
from .scoring import BAND_LEVELS, QUANTILE_LEVELS

log = logging.getLogger("horos.forecaster")

# The touch levels quoted on every scheduled forecast, as multiples of the anchor price. Fixed in
# advance so they cannot be chosen after the fact to be easy ones — a forecaster who picked its own
# levels per call could quote only the questions it was going to get right.
TOUCH_OFFSETS: tuple[float, ...] = (0.005, 0.01, 0.02, 0.05)


class Forecaster:
    """Produces the forecast payload the ledger records."""

    def __init__(self, data: MarketData, runner: KronosRunner):
        self.data = data
        self.runner = runner

    # ── the scheduled path ───────────────────────────────────────────────────────────────────

    def round_for(self, symbol: str, timeframe: str = SCORED_TIMEFRAME,
                  paths: int = SCHEDULED_PATHS) -> dict[str, dict]:
        """One generation, sliced into a forecast per horizon. Keyed by horizon name."""
        longest = max(HORIZONS.values())
        candles = self.data.candles(symbol, timeframe, limit=CONTEXT_CANDLES)
        fan = self.runner.forecast(candles, horizon_candles=longest, n_paths=paths)
        return {name: self._payload(fan.slice_to(steps), candles, name, sliced_from=longest)
                for name, steps in HORIZONS.items()}

    def one(self, symbol: str, timeframe: str = SCORED_TIMEFRAME, horizon: str = "1h",
            paths: int | None = None) -> dict:
        """A single forecast, generated live. Used for off-list symbols and bespoke horizons."""
        steps = HORIZONS.get(horizon)
        if steps is None:
            raise ForecastError(
                f"unknown horizon {horizon!r}; the scored horizons are "
                f"{', '.join(sorted(HORIZONS, key=lambda h: HORIZONS[h]))}", "bad_horizon")
        n = paths if paths is not None else (SCHEDULED_PATHS if is_scored(symbol) else LIVE_PATHS)
        candles = self.data.candles(symbol, timeframe, limit=CONTEXT_CANDLES)
        fan = self.runner.forecast(candles, horizon_candles=steps, n_paths=n)
        return self._payload(fan, candles, horizon, sliced_from=None)

    # ── the payload ──────────────────────────────────────────────────────────────────────────

    def _payload(self, fan: Fan, candles, horizon: str, sliced_from: int | None) -> dict:
        distribution = fan.to_distribution(QUANTILE_LEVELS, BAND_LEVELS)
        distribution["touch"] = self._touch(fan)
        distribution["calibrated"] = False       # the conformal layer has not run yet
        return {
            "model": fan.model,
            "model_version": fan.model_version,
            "distribution": distribution,
            "inputs": self._inputs(candles, fan, sliced_from),
        }

    def _touch(self, fan: Fan) -> dict[str, dict]:
        """Touch probabilities at fixed offsets either side of the anchor.

        Read off each path's own high and low over the *sliced* horizon, so a one-hour probability is
        answered with a one-hour path rather than the extreme of a 24-hour one.
        """
        out: dict[str, dict] = {}
        for offset in TOUCH_OFFSETS:
            up = fan.anchor_price * (1.0 + offset)
            down = fan.anchor_price * (1.0 - offset)
            out[f"up_{offset:g}"] = {"level": up, "direction": "up",
                                     "probability": fan.touch_probability(up, "up")}
            out[f"down_{offset:g}"] = {"level": down, "direction": "down",
                                       "probability": fan.touch_probability(down, "down")}
        return out

    def _inputs(self, candles, fan: Fan, sliced_from: int | None) -> dict:
        """Exactly what the model was shown, hashed, so the forecast is reproducible.

        The candle digest is the load-bearing part: it lets anyone re-fetch the same window from the
        exchange, hash it the same way, and confirm we forecast from the history we said we did.
        """
        from .crypto import digest
        return {
            "exchange": candles.exchange,
            "symbol": candles.symbol,
            "timeframe": candles.timeframe,
            "context_candles": len(candles),
            "first_candle_ts": int(candles.rows[0][0]),
            "last_candle_ts": int(candles.rows[-1][0]),
            "candles_sha256": digest(candles.rows),
            "anchor_close": candles.last_close,
            "n_paths": fan.n_paths,
            "generation_seconds": round(fan.seconds, 2),
            "sliced_from_candles": sliced_from,
            "features_used": ["open", "high", "low", "close", "volume", "calendar"],
            "features_not_used": ["funding", "open_interest", "basis", "order_book",
                                  "liquidations", "news"],
        }


def scheduled_forecaster(data: MarketData, runner: KronosRunner):
    """Adapter matching the callable the Recorder expects, with per-round generation reuse.

    The Recorder asks for one (symbol, horizon) at a time. Regenerating for each would cost four
    full generations per symbol per hour instead of one, so the round is generated on first ask and
    the other three horizons are served from the same path set — which is also what makes them
    mutually consistent.
    """
    fc = Forecaster(data, runner)
    cache: dict[tuple[str, str], dict[str, dict]] = {}

    def forecast(symbol: str, timeframe: str, horizon: str) -> dict:
        key = (symbol, timeframe)
        if key not in cache:
            cache.clear()                 # one round in flight at a time; never hold stale anchors
            cache[key] = fc.round_for(symbol, timeframe)
        payload = cache[key].get(horizon)
        if payload is None:
            raise ForecastError(f"no forecast was generated for horizon {horizon!r}", "bad_horizon")
        return payload

    forecast.reset = cache.clear          # type: ignore[attr-defined]
    return forecast


__all__: list[Any] = ["Forecaster", "scheduled_forecaster", "TOUCH_OFFSETS"]
