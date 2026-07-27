"""Groups C and D — forecasts, and the surprise/regime reads built on the same distribution.

**The committed-first rule.** For a scored market at a standard horizon, these services return the
forecast that was already issued on the last candle boundary, hash-committed on X Layer, and written
into the public ledger — not one generated when you asked. That is deliberately stronger: the answer
you bought provably predates your request, so it cannot have been shaped for you, and its
`forecast_id` will appear on the scorecard with its score once the window closes. Whether you were
served a committed forecast or a freshly generated one is stated in every response.

For anything else — an off-list symbol, a bespoke horizon — the model runs live, at a lower path
count, and the response says so plainly along with the fact that no public record exists for it.

**Bands are conformally calibrated where there is enough history to calibrate them, and labelled raw
where there is not.** A band that has not been corrected is never passed off as one that has.
"""
from __future__ import annotations

import math

import numpy as np

from core.conformal import Calibrator
from core.kronos_runner import ForecastError
from core.markets import HORIZONS, SCORED_TIMEFRAME, is_scored
from core.scoring import BAND_LEVELS, QUANTILE_LEVELS

from .registry import Param, ServiceError, service

SYMBOL = Param("symbol", "string", "Unified ccxt symbol, e.g. 'BTC/USDT:USDT'. The three symbols "
                                   "with a public accuracy record are BTC, ETH and SOL perpetuals.",
               required=True)
HORIZON = Param("horizon", "string", "How far ahead: 1h, 4h, 12h or 24h.", default="1h")


def _resolve(inp: dict, ctx, horizon_key: str = "horizon"):
    """Serve the committed forecast where one exists; otherwise generate and say so."""
    symbol = inp["symbol"]
    horizon = str(inp.get(horizon_key, "1h"))
    if horizon not in HORIZONS:
        raise ServiceError(
            f"unknown horizon {horizon!r}. Supported: "
            f"{', '.join(sorted(HORIZONS, key=lambda h: HORIZONS[h]))}.", code="bad_horizon")
    committed = ctx.latest_committed(symbol, horizon)
    if committed is not None:
        return committed, True
    try:
        payload = ctx.forecaster.one(symbol, SCORED_TIMEFRAME, horizon)
    except ForecastError as e:
        raise ServiceError(str(e), code=e.code) from e
    return {"forecast_id": None, "body": payload, "window": None}, False


def _provenance(row: dict, from_ledger: bool, symbol: str) -> dict:
    """The block that says where this answer came from and what record stands behind it."""
    p: dict = {
        "served_from": "ledger" if from_ledger else "live",
        "has_public_record": is_scored(symbol),
    }
    if from_ledger:
        p.update({
            "forecast_id": row["forecast_id"],
            "committed_at": row.get("issued_at"),
            "window": row.get("window"),
            "anchor": row.get("anchor"),
            "note": ("this forecast was issued and hash-committed on X Layer before you asked for "
                     "it, so it cannot have been shaped for this request. It will appear on the "
                     "public scorecard with its score once its window closes."),
        })
    else:
        p["note"] = ("generated live for this request. It is not part of the scored record, so no "
                     "accuracy history stands behind this particular forecast.")
    if not is_scored(symbol):
        p["record_warning"] = (
            f"Horos publishes a forward accuracy record for BTC, ETH and SOL perpetuals on the 1h "
            f"timeframe only. {symbol} is not one of them, so this answer carries no measured track "
            f"record. Treat the uncertainty bands as the model's own claim, not as a verified one.")
    return p


def _calibrated_bands(ctx, symbol: str, horizon: str, raw: dict) -> tuple[dict, dict]:
    """Apply the conformal correction to each band, reporting per level whether it applied."""
    out, meta = {}, {}
    for level in BAND_LEVELS:
        key = f"{level:g}"
        band = raw.get(key)
        if not band:
            continue
        cal = ctx.calibrator.from_ledger(ctx.ledger, symbol, horizon, level)
        out[key] = cal.apply(float(band[0]), float(band[1]))
        meta[key] = cal.as_dict()
    return out, meta


# ── C. forecast ──────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="forecast.path", group="C. Forecast", price=0.08,
    title="The full predictive distribution",
    short="Full forecast distribution",
    summary="Returns the complete percentile fan across future candles from a foundation model run "
            "over 512 candles of context, rather than the single averaged line the library returns "
            "by default.",
    returns="Returns every quantile from the 1st to the 99th, the central bands, the per-path "
            "terminal prices the whole thing was computed from, and the model's own blind spots.",
    depth="The model generates hundreds of independent future paths and its library averages them "
          "into one line before returning. Horos keeps the paths. Everything downstream — bands, "
          "touch probabilities, tail percentiles, volatility — comes out of that fan, and the raw "
          "terminal prices are returned so any statistic can be recomputed independently.",
    params=(SYMBOL, HORIZON,
            Param("include_paths", "boolean", "Include every full candle-by-candle path, not just "
                                              "terminal prices. Large.", default=False)),
    example={"symbol": "BTC/USDT:USDT", "horizon": "4h"},
)
def forecast_path(inp: dict, ctx) -> dict:
    row, from_ledger = _resolve(inp, ctx)
    body = row["body"]
    d = body["distribution"]
    symbol, horizon = inp["symbol"], str(inp.get("horizon", "1h"))
    bands, cal_meta = _calibrated_bands(ctx, symbol, horizon, d.get("bands") or {})
    out = {
        "symbol": symbol, "horizon": horizon, "timeframe": SCORED_TIMEFRAME,
        "anchor_price": d["anchor_price"],
        "anchor_candle_ts": d.get("anchor_timestamp_ms"),
        "model": body["model"], "model_version": body["model_version"],
        "paths": d["n_paths"],
        "quantile_levels": d["quantile_levels"],
        "quantiles": d["quantiles"],
        "bands_raw": d.get("bands"),
        "bands_calibrated": bands,
        "calibration": cal_meta,
        "terminal_prices": d["samples"],
        "expected_move_pct": round(100 * (float(np.median(d["samples"])) / d["anchor_price"] - 1), 4),
        "provenance": _provenance(row, from_ledger, symbol),
        "model_blind_spots": ctx.runner.describe()["cannot_see"],
    }
    if inp.get("include_paths") and d.get("paths"):
        out["full_paths"] = d["paths"]
    return out


@service(
    endpoint="forecast.range", group="C. Forecast", price=0.05,
    title="A high/low envelope with a real coverage guarantee",
    short="Calibrated price envelope",
    summary="Gives the price envelope for a horizon with conformally calibrated bands, so a stated "
            "90% band is corrected against how often this model's 90% bands have actually contained "
            "the outcome.",
    returns="Returns the 50, 80 and 90 percent bands both raw and calibrated, the correction applied "
            "to each, and how many past forecasts each correction rests on.",
    depth="Raw generative-model quantiles are miscalibrated — a sampler's '90% band' contains "
          "whatever it contains. Split conformal prediction corrects this with a finite-sample, "
          "distribution-free guarantee. Where too few forecasts have been scored to calibrate a "
          "level, the band is returned raw and explicitly labelled as carrying no guarantee.",
    params=(SYMBOL, HORIZON),
    example={"symbol": "ETH/USDT:USDT", "horizon": "24h"},
)
def forecast_range(inp: dict, ctx) -> dict:
    row, from_ledger = _resolve(inp, ctx)
    d = row["body"]["distribution"]
    symbol, horizon = inp["symbol"], str(inp.get("horizon", "1h"))
    bands, cal_meta = _calibrated_bands(ctx, symbol, horizon, d.get("bands") or {})
    anchor = d["anchor_price"]
    return {
        "symbol": symbol, "horizon": horizon, "anchor_price": anchor,
        "bands": {
            key: {"lower": v[0], "upper": v[1],
                  "lower_pct": round(100 * (v[0] / anchor - 1), 4),
                  "upper_pct": round(100 * (v[1] / anchor - 1), 4),
                  "width_pct": round(100 * (v[1] - v[0]) / anchor, 4),
                  "raw": (d.get("bands") or {}).get(key),
                  "calibration": cal_meta.get(key)}
            for key, v in bands.items()},
        "how_to_read": ("A 0.9 band claims the close at the end of the horizon falls inside it 90% "
                        "of the time. Where 'calibrated' is true that claim has been corrected "
                        "against this model's own record; where it is false the band is the model's "
                        "uncorrected output."),
        "provenance": _provenance(row, from_ledger, symbol),
    }


@service(
    endpoint="forecast.touch", group="C. Forecast", price=0.05,
    title="Probability of reaching a level before the horizon expires",
    short="Probability of touching",
    summary="Answers the question an open position actually implies: what are the odds this price is "
            "reached at any point before the horizon ends, measured across every generated path "
            "rather than from the closing price alone.",
    returns="Returns the probability for each level you name, the direction it would be reached "
            "from, the fraction of paths that got there, and the same for the standard offsets.",
    depth="Resolved on path extremes, not terminal prices. A path that spikes through your level and "
          "comes back did touch it, and a probability computed from closing prices alone would miss "
          "exactly the case a stop-loss cares about. Scored later with the Brier score against what "
          "the candle highs and lows actually did.",
    params=(SYMBOL, HORIZON,
            Param("levels", "array", "Price levels to test. Direction is inferred from whether each "
                                     "sits above or below the current price.", default=None)),
    example={"symbol": "BTC/USDT:USDT", "horizon": "12h", "levels": [70000, 60000]},
)
def forecast_touch(inp: dict, ctx) -> dict:
    row, from_ledger = _resolve(inp, ctx)
    d = row["body"]["distribution"]
    symbol, horizon = inp["symbol"], str(inp.get("horizon", "1h"))
    anchor = d["anchor_price"]
    highs = np.asarray(d.get("path_high") or d["samples"], dtype=float)
    lows = np.asarray(d.get("path_low") or d["samples"], dtype=float)

    asked = []
    for raw_level in (inp.get("levels") or []):
        try:
            level = float(raw_level)
        except (TypeError, ValueError):
            raise ServiceError(f"{raw_level!r} is not a price.", code="bad_level") from None
        if level <= 0:
            raise ServiceError(f"a price level must be positive, got {level}", code="bad_level")
        up = level >= anchor
        hit = float(np.mean(highs >= level)) if up else float(np.mean(lows <= level))
        asked.append({
            "level": level, "direction": "up" if up else "down",
            "distance_pct": round(100 * (level / anchor - 1), 4),
            "probability": round(hit, 4),
            "paths_reaching": int(round(hit * len(highs))),
            "paths_total": len(highs),
        })

    return {
        "symbol": symbol, "horizon": horizon, "anchor_price": anchor,
        "levels": asked,
        "standard_offsets": {
            k: {"level": v["level"], "direction": v["direction"],
                "probability": round(float(v["probability"]), 4)}
            for k, v in (d.get("touch") or {}).items()},
        "resolution": ("measured on each path's own high and low across the horizon, so a level "
                       "touched intraday counts even if price closed back inside"),
        "provenance": _provenance(row, from_ledger, symbol),
    }


@service(
    endpoint="forecast.volatility", group="C. Forecast", price=0.03,
    title="Expected realised volatility",
    short="Expected volatility",
    summary="Forecasts how much the market will move rather than which way, which is the part of a "
            "price process that is genuinely predictable, and compares it against volatility "
            "actually realised over the recent past.",
    returns="Returns expected volatility per candle and annualised, the recent realised figure for "
            "comparison, the ratio between them, and whether the model expects conditions to calm "
            "or intensify.",
    depth="Direction at short horizons is close to random and Horos does not sell it. Volatility is "
          "different: it clusters, it mean-reverts, and it is measurable after the fact. The "
          "forecast is the dispersion of the generated path ensemble; the comparison is the "
          "close-to-close realised volatility of the context window.",
    params=(SYMBOL, HORIZON,
            Param("lookback", "integer", "Candles of realised volatility to compare against.",
                  default=168)),
    example={"symbol": "ETH/USDT:USDT", "horizon": "24h", "lookback": 168},
)
def forecast_volatility(inp: dict, ctx) -> dict:
    row, from_ledger = _resolve(inp, ctx)
    d = row["body"]["distribution"]
    symbol, horizon = inp["symbol"], str(inp.get("horizon", "1h"))
    lookback = max(24, min(int(inp.get("lookback", 168)), 2000))

    per_candle = d.get("per_candle_log_return_sd")
    if per_candle is None:
        terminal = np.asarray(d["samples"], dtype=float)
        steps = HORIZONS[horizon]
        per_candle = float(np.std(np.log(terminal / d["anchor_price"]))) / math.sqrt(steps)

    candles = ctx.data.candles(symbol, SCORED_TIMEFRAME, limit=lookback)
    realised = float(np.std(candles.log_returns()))
    hours_per_year = 24 * 365
    ann = lambda v: round(100 * v * math.sqrt(hours_per_year), 3)   # noqa: E731

    ratio = per_candle / realised if realised else None
    return {
        "symbol": symbol, "horizon": horizon,
        "expected": {"per_candle_log_return_sd": round(per_candle, 8),
                     "annualised_pct": ann(per_candle),
                     "expected_move_over_horizon_pct":
                         round(100 * per_candle * math.sqrt(HORIZONS[horizon]), 4)},
        "realised": {"lookback_candles": len(candles),
                     "per_candle_log_return_sd": round(realised, 8),
                     "annualised_pct": ann(realised)},
        "ratio_expected_to_realised": round(ratio, 4) if ratio else None,
        "reading": (("the model expects conditions calmer than the recent past"
                     if ratio < 0.9 else
                     "the model expects conditions more volatile than the recent past"
                     if ratio > 1.1 else
                     "the model expects volatility broadly in line with the recent past")
                    if ratio else "no realised volatility was available to compare against"),
        "provenance": _provenance(row, from_ledger, symbol),
    }


# ── D. surprise and regime ───────────────────────────────────────────────────────────────────────

@service(
    endpoint="market.surprise", group="D. Surprise and regime", price=0.02,
    title="Was the last candle actually unusual",
    short="Was that candle unusual",
    summary="Takes the candle that just closed and asks where it fell inside the distribution the "
            "model had predicted for it before it happened, which separates a real dislocation from "
            "ordinary noise.",
    returns="Returns the tail percentile the outcome landed in, how many standard deviations from "
            "the predicted centre it was, whether it fell outside the 90 percent band, and the "
            "forecast it is being measured against.",
    depth="This is only meaningful against a prediction that predates the candle, which is why it "
          "reads from the committed ledger where one exists rather than fitting a distribution after "
          "the fact. Fitting afterwards would make every outcome look ordinary — the distribution "
          "would have been drawn around it.",
    params=(SYMBOL,),
    example={"symbol": "BTC/USDT:USDT"},
)
def market_surprise(inp: dict, ctx) -> dict:
    symbol = inp["symbol"]
    prior = ctx.most_recent_scored(symbol, "1h")
    candles = ctx.data.candles(symbol, SCORED_TIMEFRAME, limit=2)
    last = candles.rows[-1]
    actual_close = float(last[4])

    if prior is None:
        # An honest degraded answer rather than a fabricated one: measure against the empirical
        # distribution of recent returns, and say that is what is happening.
        history = ctx.data.candles(symbol, SCORED_TIMEFRAME, limit=500)
        rets = np.asarray(history.log_returns(), dtype=float)
        move = math.log(actual_close / float(history.rows[-2][4]))
        pct = float(np.mean(rets <= move))
        return {
            "symbol": symbol, "candle_ts": int(last[0]), "close": actual_close,
            "measured_against": "empirical distribution of the last 500 hourly returns",
            "percentile_rank_0_1": round(pct, 4),
            "percentile_0_100": round(pct * 100, 2),
            "sigma": round(move / float(np.std(rets)), 3) if np.std(rets) else None,
            "unusual": bool(pct < 0.05 or pct > 0.95),
            "note": ("No committed forecast covering this candle exists yet, so the comparison is "
                     "against recent realised returns rather than against a prediction that "
                     "predated the candle. That is a weaker test: it says the move was large "
                     "relative to history, not that it was unexpected."),
            "provenance": {"served_from": "empirical", "has_public_record": is_scored(symbol)},
        }

    samples = np.asarray(prior["body"]["distribution"]["samples"], dtype=float)
    pct = float(np.mean(samples <= actual_close))
    sd = float(np.std(samples))
    band = (prior["body"]["distribution"].get("bands") or {}).get("0.9")
    outside = bool(band and not (float(band[0]) <= actual_close <= float(band[1])))
    return {
        "symbol": symbol,
        "candle_ts": int(last[0]),
        "close": actual_close,
        "measured_against": "the forecast committed before this candle opened",
        "forecast_id": prior["forecast_id"],
        "predicted_median": float(np.median(samples)),
        "percentile_rank_0_1": round(pct, 4),
        "percentile_0_100": round(pct * 100, 2),
        "tail": round(min(pct, 1 - pct), 4),
        "sigma": round((actual_close - float(np.mean(samples))) / sd, 3) if sd else None,
        "outside_90_band": outside,
        "band_90": band,
        "unusual": bool(pct < 0.05 or pct > 0.95),
        "reading": _surprise_reading(pct, outside),
        "provenance": {"served_from": "ledger", "forecast_id": prior["forecast_id"],
                       "committed_at": prior.get("issued_at"),
                       "has_public_record": is_scored(symbol),
                       "note": "the distribution this is measured against was hash-committed on "
                               "X Layer before the candle opened"},
    }


def _surprise_reading(pct: float, outside: bool) -> str:
    tail = min(pct, 1 - pct)
    side = "above" if pct > 0.5 else "below"
    if tail < 0.01:
        return (f"a genuine dislocation — the close landed in the extreme {tail * 100:.1f}% tail "
                f"{side} what was predicted")
    if tail < 0.05:
        return f"unusual — the close landed in the {tail * 100:.1f}% tail {side} the prediction"
    if outside:
        return "outside the 90% band, but not far into the tail"
    return f"ordinary — the close landed at the {pct * 100:.0f}th percentile of the prediction"


@service(
    endpoint="market.regime", group="D. Surprise and regime", price=0.02,
    title="Which volatility regime this market is in",
    short="Volatility regime and Hurst",
    summary="Classifies the current state of a market by measuring realised volatility against its "
            "own recent distribution and estimating the Hurst exponent, which distinguishes a "
            "trending market from a mean-reverting one.",
    returns="Returns the volatility regime and its percentile, the Hurst exponent with its "
            "interpretation, the current drawdown from the window high, and a one-line state.",
    depth="Two independent measurements rather than one. Volatility percentile says how violent "
          "conditions are; Hurst says whether moves persist or revert, which is the difference "
          "between a trend worth following and chop that punishes following it. Hurst is computed "
          "by rescaled-range analysis over several lag scales.",
    params=(SYMBOL,
            Param("lookback", "integer", "Candles to classify over.", default=500),
            Param("window", "integer", "Rolling window for the volatility percentile.", default=24)),
    example={"symbol": "BTC/USDT:USDT", "lookback": 500},
)
def market_regime(inp: dict, ctx) -> dict:
    symbol = inp["symbol"]
    lookback = max(100, min(int(inp.get("lookback", 500)), 5000))
    window = max(6, min(int(inp.get("window", 24)), lookback // 4))
    candles = ctx.data.candles(symbol, SCORED_TIMEFRAME, limit=lookback)
    rets = np.asarray(candles.log_returns(), dtype=float)
    if rets.size < window * 3:
        raise ServiceError(f"only {rets.size} returns are available; at least {window * 3} are "
                           f"needed to classify a regime.", code="insufficient_history")

    rolling = np.array([rets[i:i + window].std() for i in range(len(rets) - window + 1)])
    current = float(rolling[-1])
    pct = float(np.mean(rolling <= current))
    hurst = _hurst(rets)
    closes = np.asarray(candles.closes, dtype=float)
    peak = float(closes.max())

    return {
        "symbol": symbol,
        "lookback_candles": len(candles),
        "volatility": {
            "current_window_sd": round(current, 8),
            "annualised_pct": round(100 * current * math.sqrt(24 * 365), 3),
            "percentile_of_own_history": round(pct, 4),
            "regime": ("high" if pct > 0.8 else "low" if pct < 0.2 else "normal"),
        },
        "hurst": {
            "exponent": round(hurst, 4) if hurst is not None else None,
            "interpretation": (
                "trending — moves tend to persist" if hurst and hurst > 0.55 else
                "mean-reverting — moves tend to reverse" if hurst and hurst < 0.45 else
                "close to a random walk — no persistence either way"
                if hurst is not None else "could not be estimated"),
            "method": "rescaled range over dyadic lag scales",
        },
        "drawdown": {
            "from_window_high_pct": round(100 * (closes[-1] / peak - 1), 4),
            "window_high": peak, "window_low": float(closes.min()),
        },
        "state": _regime_state(pct, hurst),
    }


def _hurst(returns: np.ndarray) -> float | None:
    """Rescaled-range estimate of the Hurst exponent.

    R/S is computed on the cumulative deviation series over several lag scales and the exponent is
    the slope of log(R/S) against log(lag). Below ~0.5 the series mean-reverts, above ~0.5 it trends.
    Returns None rather than a number when the series is too short or degenerate to support one.
    """
    x = np.asarray(returns, dtype=float)
    n = x.size
    if n < 64:
        return None
    lags, rs = [], []
    lag = 8
    while lag <= n // 2:
        chunks = n // lag
        vals = []
        for i in range(chunks):
            seg = x[i * lag:(i + 1) * lag]
            dev = np.cumsum(seg - seg.mean())
            spread = dev.max() - dev.min()
            sd = seg.std()
            if sd > 0 and spread > 0:
                vals.append(spread / sd)
        if vals:
            lags.append(lag)
            rs.append(float(np.mean(vals)))
        lag *= 2
    if len(lags) < 3:
        return None
    slope = float(np.polyfit(np.log(lags), np.log(rs), 1)[0])
    return slope


def _regime_state(vol_pct: float, hurst: float | None) -> str:
    vol = "high volatility" if vol_pct > 0.8 else "low volatility" if vol_pct < 0.2 else "normal volatility"
    if hurst is None:
        return f"{vol}; persistence could not be estimated"
    if hurst > 0.55:
        return f"{vol}, trending — moves have been persisting"
    if hurst < 0.45:
        return f"{vol}, choppy — moves have been reversing"
    return f"{vol}, no clear persistence — close to a random walk"


@service(
    endpoint="market.anomaly", group="D. Surprise and regime", price=0.03,
    title="Where price, funding and open interest disagree",
    short="Price-funding-OI divergence",
    summary="Cross-checks three signals that normally move together — price, funding rate and open "
            "interest — and reports where they have come apart, which is the reading no single feed "
            "can give you.",
    returns="Returns each signal's move over the window, every divergence found with its severity "
            "and what it implies about positioning, and an explicit list of signals that could not "
            "be fetched.",
    depth="These are exactly the features the price model cannot see, which is why they are checked "
          "separately. Price up on falling open interest is a short squeeze, not accumulation; "
          "price up with funding at an extreme is crowded positioning. A signal that could not be "
          "fetched is named as missing, never treated as showing no divergence.",
    params=(SYMBOL,
            Param("periods", "integer", "Candles to measure each signal's change over.",
                  default=24)),
    example={"symbol": "BTC/USDT:USDT", "periods": 24},
)
def market_anomaly(inp: dict, ctx) -> dict:
    from core.market_data import MarketDataError

    symbol = inp["symbol"]
    periods = max(2, min(int(inp.get("periods", 24)), 500))
    candles = ctx.data.candles(symbol, SCORED_TIMEFRAME, limit=periods + 1)
    px_change = 100 * (candles.closes[-1] / candles.closes[0] - 1)

    signals: dict = {"price_change_pct": round(px_change, 4)}
    unavailable: list[dict] = []
    divergences: list[dict] = []

    try:
        fr = ctx.data.funding(symbol)
        rate = float(fr["fundingRate"])
        hist = ctx.data.funding_history(symbol, limit=200)
        rates = [float(h["fundingRate"]) for h in hist if h.get("fundingRate") is not None]
        pct = (sum(1 for r in rates if r <= rate) / len(rates)) if rates else None
        signals["funding_rate"] = rate
        signals["funding_percentile"] = round(pct, 4) if pct is not None else None
        if pct is not None and pct > 0.95 and px_change > 0:
            divergences.append({
                "signals": ["price", "funding"], "severity": "high",
                "finding": "price is rising into funding at the top of its own range",
                "implication": "long positioning is crowded and paying heavily to stay on; such "
                               "conditions have historically preceded long liquidations, though "
                               "Horos publishes no measured claim about that"})
        elif pct is not None and pct < 0.05 and px_change < 0:
            divergences.append({
                "signals": ["price", "funding"], "severity": "high",
                "finding": "price is falling into funding at the bottom of its own range",
                "implication": "short positioning is crowded and paying to stay on"})
    except (MarketDataError, KeyError, TypeError) as e:
        unavailable.append({"signal": "funding", "reason": str(e)})

    try:
        oi_hist = ctx.data.open_interest_history(symbol, SCORED_TIMEFRAME, limit=periods + 1)
        series = [float(h.get("openInterestValue") or h.get("openInterestAmount") or 0)
                  for h in oi_hist]
        series = [v for v in series if v > 0]
        if len(series) >= 2:
            oi_change = 100 * (series[-1] / series[0] - 1)
            signals["open_interest_change_pct"] = round(oi_change, 4)
            if px_change > 1 and oi_change < -1:
                divergences.append({
                    "signals": ["price", "open_interest"], "severity": "high",
                    "finding": f"price rose {px_change:.2f}% while open interest fell "
                               f"{abs(oi_change):.2f}%",
                    "implication": "this is shorts closing rather than new buyers arriving — a "
                                   "squeeze, not accumulation"})
            elif px_change < -1 and oi_change < -1:
                divergences.append({
                    "signals": ["price", "open_interest"], "severity": "medium",
                    "finding": f"price fell {abs(px_change):.2f}% while open interest fell "
                               f"{abs(oi_change):.2f}%",
                    "implication": "longs are closing out; a de-risking move rather than new "
                                   "short pressure"})
            elif abs(px_change) < 0.3 and abs(oi_change) > 3:
                divergences.append({
                    "signals": ["price", "open_interest"], "severity": "medium",
                    "finding": f"open interest moved {oi_change:+.2f}% on almost no price change",
                    "implication": "positioning is building without moving price — both sides are "
                                   "adding"})
        else:
            unavailable.append({"signal": "open_interest",
                                "reason": "the exchange returned too little history"})
    except MarketDataError as e:
        unavailable.append({"signal": "open_interest", "reason": str(e)})

    return {
        "symbol": symbol,
        "window": {"timeframe": SCORED_TIMEFRAME, "periods": periods},
        "signals": signals,
        "divergences": divergences,
        "divergence_count": len(divergences),
        "signals_unavailable": unavailable,
        "verdict": ("no divergence found between the signals that could be measured"
                    if not divergences else
                    f"{len(divergences)} divergence(s) found"),
        "caveat": ("liquidation data is not included: ccxt exposes no liquidation feed for this "
                   "exchange, so cascade detection would have to be inferred rather than measured, "
                   "and Horos does not sell inferred data as measured data."),
    }


@service(
    endpoint="market.dislocation", group="D. Surprise and regime", price=0.05,
    title="Is this cross-venue spread real or an artefact",
    short="Real or artefact spread",
    summary="Compares the same asset across venues and then does the work that turns a spread into a "
            "usable signal: checks each feed's freshness, checks whether the resting book could "
            "absorb the size, and reports what the spread is actually worth.",
    returns="Returns the raw spread, the spread after excluding stale feeds, the depth available on "
            "each side at that price, and a verdict on whether the gap is tradeable or a data "
            "artefact.",
    depth="Most cross-venue spreads that look large are stale quotes or paper-thin books. This "
          "checks both before calling anything a dislocation: quote age against a threshold you "
          "set, and resting notional within the spread on both legs. A spread that survives neither "
          "check is reported as an artefact, which is the useful answer.",
    params=(SYMBOL,
            Param("exchanges", "array", "Venues to compare. Reachable from the deployment region. Binance and Bybit geo-block US IP ranges, and Horos runs in us-east-1 — defaulting to venues that always fail would make every cross-venue answer a list of errors.",
                  default=["okx", "gate", "bitget", "kucoin"]),
            Param("size_usd", "number", "Notional you would want to trade, for the depth check.",
                  default=10000),
            Param("max_age_seconds", "integer", "Quotes older than this are treated as stale.",
                  default=60)),
    example={"symbol": "BTC/USDT:USDT", "size_usd": 50000},
)
def market_dislocation(inp: dict, ctx) -> dict:
    from core.market_data import MarketDataError

    symbol = inp["symbol"]
    names = inp.get("exchanges") or ["okx", "gate", "bitget", "kucoin"]
    size = float(inp.get("size_usd", 10_000))
    if size <= 0:
        raise ServiceError("size_usd must be positive.", code="bad_size")
    max_age = int(inp.get("max_age_seconds", 60))

    venues_out = ctx.registry_call("market.venues", {
        "symbol": symbol, "exchanges": names, "max_age_seconds": max_age})
    if venues_out["venues_compared"] < 2:
        raise ServiceError(
            f"only {venues_out['venues_compared']} venue returned a fresh price for {symbol}; a "
            f"dislocation needs at least two to compare.", code="insufficient_venues")

    cheap, dear = venues_out["cheapest"], venues_out["dearest"]
    checks: dict = {}
    tradeable = True
    reasons: list[str] = []

    for role, v in (("buy_leg", cheap), ("sell_leg", dear)):
        try:
            ob = ctx.data.orderbook(symbol, depth=400, exchange=v["exchange"])
        except MarketDataError as e:
            checks[role] = {"exchange": v["exchange"], "depth_unavailable": str(e)}
            tradeable = False
            reasons.append(f"the book on {v['exchange']} could not be read, so the size check for "
                           f"the {role.replace('_', ' ')} did not run")
            continue
        side = ob["asks"] if role == "buy_leg" else ob["bids"]
        notional = sum(p * q for p, q in side)
        checks[role] = {"exchange": v["exchange"], "price": v["price"],
                        "resting_notional_usd": round(notional, 2),
                        "covers_requested_size": notional >= size}
        if notional < size:
            tradeable = False
            reasons.append(f"the {role.replace('_', ' ')} on {v['exchange']} shows only "
                           f"${notional:,.0f} of resting notional against a requested ${size:,.0f}")

    if venues_out["stale_excluded"]:
        reasons.append(f"{len(venues_out['stale_excluded'])} venue(s) were excluded as stale; the "
                       f"spread would have looked larger had they been included")

    return {
        "symbol": symbol,
        "spread_bps": venues_out["spread_bps"],
        "cheapest": cheap, "dearest": dear,
        "venues_compared": venues_out["venues_compared"],
        "stale_excluded": venues_out["stale_excluded"],
        "unreachable": venues_out["unreachable"],
        "size_checked_usd": size,
        "depth_checks": checks,
        "verdict": ("this spread survives the freshness and depth checks at the size requested"
                    if tradeable and venues_out["spread_bps"] > 1 else
                    "this spread is within normal cross-venue noise" if tradeable else
                    "this spread does not survive the checks — treat it as an artefact"),
        "reasons": reasons,
        "not_included": ["exchange fees", "funding differentials between venues",
                         "withdrawal and transfer time between venues", "slippage beyond the top "
                         "400 levels"],
    }
