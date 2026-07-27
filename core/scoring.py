"""Proper scoring rules — the arithmetic the whole product is judged by.

Every function here is deterministic and depends on nothing but its arguments, so a buyer holding
the ledger and public candle data can recompute any published number and get ours to the last digit.
That reproducibility is the point; if scoring needed our infrastructure it would be one more thing
to take on trust.

Three deliberate choices, each of which removes a way of looking better than we are:

**The fair CRPS estimator, not the standard one.** With ``m`` samples the usual estimator subtracts
``1/(2m²)`` times the pairwise spread; the fair estimator (Ferro 2014, Zamo & Naveau 2018) subtracts
``1/(2m(m−1))``. Both overstate CRPS by a term proportional to the ensemble's own spread, so the
usual one inflates a wide ensemble more than a narrow one and therefore ranks *under-dispersed*
forecasts better than they deserve — a model claiming more certainty than it has gets rewarded for
it. The fair estimator removes that term. It reads lower in absolute value, which is not the point;
the point is that it does not pay us for pretending to be sure.

**Skill against a real baseline, not a bare score.** A CRPS of 340 means nothing on its own. It means
something against the empirical-return random walk, which is what a caller gets for free by
resampling recent history. If the model cannot beat that, the scorecard says so.

**Interval score, not just coverage.** A forecaster can hit 90% coverage trivially by quoting an
absurdly wide band. The interval score charges for width and for misses together, so it cannot be
gamed in that direction.

Lower is better for every score here except the skill scores, where higher is better and zero means
"no better than the baseline".
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

# numpy renamed `trapz` to `trapezoid` in 2.0 and kept the old name as a deprecated alias. The pin
# here is 1.26.4, forced by torch 2.2, which has only the old name. Binding once means the scorer
# runs unchanged on either.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# Bump this whenever a formula changes. Scores carry it, so a published number always states which
# arithmetic produced it and an old score stays interpretable after the scorer moves on.
SCORER_VERSION = "1.0.0"

# The fair estimator is unbiased but not bounded below at zero: on a very small ensemble that happens
# to straddle the outcome it can come out slightly negative. That is a property of the estimator, not
# an error, and it stops mattering once the ensemble is large. Forecasts are issued with at least
# this many paths so a published CRPS is never an artefact of ensemble size.
MIN_ENSEMBLE = 200

# The quantile grid every forecast is issued on. Fixed, because a forecaster that chose its own grid
# per call could quietly drop the levels it does badly on.
QUANTILE_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)

# Central bands quoted and scored. Keyed by nominal coverage.
BAND_LEVELS: tuple[float, ...] = (0.50, 0.80, 0.90)


# ── point and quantile scores ────────────────────────────────────────────────────────────────────

def pinball_loss(actual: float, predicted: float, level: float) -> float:
    """Quantile loss at one level.

    ``level`` is the quantile the prediction claims to be. Under-predicting a high quantile is cheap
    and over-predicting it is expensive, which is what makes the minimiser the true quantile.
    """
    delta = actual - predicted
    return max(level * delta, (level - 1.0) * delta)


def pinball_by_level(actual: float, quantiles: Sequence[float],
                     levels: Sequence[float] = QUANTILE_LEVELS) -> dict[str, float]:
    """Pinball loss at every level, plus the mean across them."""
    if len(quantiles) != len(levels):
        raise ValueError(f"{len(quantiles)} quantiles for {len(levels)} levels")
    per = {f"{lv:g}": pinball_loss(actual, q, lv) for lv, q in zip(levels, quantiles)}
    per["mean"] = sum(v for k, v in per.items() if k != "mean") / len(levels)
    return per


def crps_from_samples(actual: float, samples: Sequence[float], fair: bool = True) -> float:
    """CRPS of an ensemble forecast, by the energy-form estimator.

        CRPS = E|X − y| − ½·E|X − X'|

    with the second term estimated over ordered pairs. ``fair=True`` divides the pairwise sum by
    ``m(m−1)`` rather than ``m²``, removing the finite-ensemble bias that rewards a forecast for
    being too narrow.

    Computed via the sorted closed form rather than the O(m²) double loop, which matters because the
    scoring job runs over every forecast ever issued whenever the scorecard is regenerated.
    """
    x = np.sort(np.asarray(samples, dtype=float))
    m = x.size
    if m == 0:
        raise ValueError("CRPS needs at least one sample")
    if m == 1:
        return abs(float(x[0]) - actual)

    term1 = float(np.abs(x - actual).mean())
    # Σ_i Σ_j |x_i − x_j| for a sorted vector equals 2·Σ_i (2i − m + 1)·x_i, which is exact and
    # linear in m. Verified against the naive double loop in the tests.
    i = np.arange(m, dtype=float)
    pair_sum = 2.0 * float(np.sum((2.0 * i - m + 1.0) * x))
    denom = 2.0 * m * (m - 1) if fair else 2.0 * m * m
    return term1 - pair_sum / denom


def crps_from_quantiles(actual: float, quantiles: Sequence[float],
                        levels: Sequence[float] = QUANTILE_LEVELS) -> float:
    """CRPS approximated from a quantile grid.

    Uses the identity ``CRPS = 2∫₀¹ pinball_τ dτ``, evaluated by the trapezoid rule over the grid.
    Only used when a forecast was recorded as quantiles without the underlying samples; the sample
    estimator above is exact and is preferred wherever the samples exist.
    """
    lv = np.asarray(levels, dtype=float)
    losses = np.array([pinball_loss(actual, q, t) for q, t in zip(quantiles, lv)])
    order = np.argsort(lv)
    return 2.0 * float(_trapezoid(losses[order], lv[order]))


# ── interval scores ──────────────────────────────────────────────────────────────────────────────

def interval_score(actual: float, lower: float, upper: float, coverage: float) -> float:
    """Winkler interval score for a central band with nominal ``coverage``.

        IS = (u − l) + (2/α)·(l − y)·1{y<l} + (2/α)·(y − u)·1{y>u},   α = 1 − coverage

    Proper: it charges width unconditionally and misses in proportion to how far outside they fall,
    so neither a hedged-wide band nor a confidently narrow one can win by construction.
    """
    alpha = 1.0 - coverage
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"coverage must be in (0,1), got {coverage}")
    score = upper - lower
    if actual < lower:
        score += (2.0 / alpha) * (lower - actual)
    elif actual > upper:
        score += (2.0 / alpha) * (actual - upper)
    return score


def covered(actual: float, lower: float, upper: float) -> bool:
    """Did the band contain the outcome. Aggregated across forecasts this is empirical coverage."""
    return lower <= actual <= upper


# ── probability scores ───────────────────────────────────────────────────────────────────────────

def brier_score(probability: float, occurred: bool) -> float:
    """Squared error of a probability forecast. Proper, bounded in [0,1], lower is better."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability out of range: {probability}")
    return (probability - (1.0 if occurred else 0.0)) ** 2


def log_score(probability: float, occurred: bool, floor: float = 1e-6) -> float:
    """Negative log likelihood of a binary outcome.

    Punishes confident misses far harder than Brier does, which is the honest treatment of a "97%
    chance it never touches" that then touches. Floored so a single wrong certainty does not make the
    aggregate infinite and unreadable — the floor is stated rather than hidden.
    """
    p = min(max(probability, floor), 1.0 - floor)
    return -math.log(p if occurred else 1.0 - p)


def probability_integral_transform(actual: float, samples: Sequence[float]) -> float:
    """Where the outcome landed inside the predictive distribution, on [0,1].

    Over many forecasts these values are uniform if and only if the distribution is calibrated.
    A U-shaped histogram means the forecasts are too narrow; a hump in the middle means too wide.
    This is the diagnostic that says *how* a model is wrong rather than only how much.
    """
    x = np.asarray(samples, dtype=float)
    if x.size == 0:
        raise ValueError("PIT needs at least one sample")
    below = float(np.count_nonzero(x < actual))
    ties = float(np.count_nonzero(x == actual))
    # Ties are vanishingly rare on float prices; splitting them keeps the transform unbiased if the
    # forecast was quantised to tick size.
    return (below + 0.5 * ties) / x.size


# ── baselines and skill ──────────────────────────────────────────────────────────────────────────

def empirical_return_samples(anchor: float, past_log_returns: Sequence[float],
                             count: int | None = None) -> list[float]:
    """The baseline every forecast must beat: a random walk drawn from recent history.

    Take the realised log returns over the same horizon, apply them to the anchor price, and that is
    a free, assumption-light predictive distribution. It is not a straw man — for short-horizon crypto
    prices it is genuinely hard to beat, which is precisely why it is the right thing to be measured
    against.
    """
    r = np.asarray(past_log_returns, dtype=float)
    if r.size == 0:
        raise ValueError("the empirical baseline needs at least one past return")
    if count is not None and count < r.size:
        # Most recent `count` returns — recency matters more than sample size in a regime shift.
        r = r[-count:]
    return [float(anchor * math.exp(v)) for v in r]


def skill_score(model: float, baseline: float) -> float | None:
    """1 − model/baseline. Positive means better than the baseline; 0 means no better.

    Returns None when the baseline scored a perfect zero, because the ratio is undefined and
    reporting a fabricated 0 or 1 there would be a lie in the direction that flatters us.
    """
    if baseline == 0:
        return None
    return 1.0 - (model / baseline)


# ── the aggregate a scorecard is made of ─────────────────────────────────────────────────────────

def score_forecast(*, actual_close: float, actual_high: float, actual_low: float,
                   distribution: dict, baseline_samples: Sequence[float] | None = None,
                   persistence: float | None = None) -> dict:
    """Score one issued forecast against what happened. The scoring job's whole payload.

    ``distribution`` is the block recorded in the ledger at issue time. Whatever is present gets
    scored and whatever is absent is reported as absent — never as a zero, and never silently
    dropped, because a missing score that looks like a good score is the exact failure this product
    exists to prevent.
    """
    out: dict = {"scorer_version": SCORER_VERSION, "actual_close": actual_close}
    not_scored: list[str] = []

    samples = distribution.get("samples") or []
    levels = distribution.get("quantile_levels") or list(QUANTILE_LEVELS)
    quantiles = distribution.get("quantiles") or []

    # --- CRPS, from samples where we have them ---------------------------------------------
    if samples:
        out["crps"] = crps_from_samples(actual_close, samples)
        out["crps_estimator"] = "fair (Ferro)"
        out["pit"] = probability_integral_transform(actual_close, samples)
    elif quantiles:
        out["crps"] = crps_from_quantiles(actual_close, quantiles, levels)
        out["crps_estimator"] = "quantile trapezoid"
        not_scored.append("pit: the forecast recorded quantiles but not the underlying samples")
    else:
        not_scored.append("crps: the forecast recorded neither samples nor quantiles")

    # --- pinball -----------------------------------------------------------------------------
    if quantiles and len(quantiles) == len(levels):
        out["pinball"] = pinball_by_level(actual_close, quantiles, levels)
    else:
        not_scored.append("pinball: no quantile grid was recorded")

    # --- bands: interval score and the coverage indicator -------------------------------------
    bands = distribution.get("bands") or {}
    if bands:
        band_scores: dict[str, dict] = {}
        for key, pair in bands.items():
            try:
                lo, hi = float(pair[0]), float(pair[1])
                nominal = float(key)
            except (TypeError, ValueError, IndexError):
                not_scored.append(f"band {key!r}: malformed, expected [lower, upper]")
                continue
            band_scores[key] = {
                "nominal": nominal,
                "lower": lo,
                "upper": hi,
                "covered": covered(actual_close, lo, hi),
                "interval_score": interval_score(actual_close, lo, hi, nominal),
                "width": hi - lo,
            }
        if band_scores:
            out["bands"] = band_scores
    else:
        not_scored.append("bands: none were recorded")

    # --- touch probabilities ------------------------------------------------------------------
    # Resolved against the realised high and low, not the close: "will it touch 120k" is a question
    # about the path, and a candle that spiked through and came back did touch it.
    touch = distribution.get("touch") or {}
    if touch:
        resolved: dict[str, dict] = {}
        for name, spec in touch.items():
            try:
                level = float(spec["level"])
                prob = float(spec["probability"])
                direction = str(spec.get("direction", "up")).lower()
            except (KeyError, TypeError, ValueError):
                not_scored.append(f"touch {name!r}: malformed, expected level/probability/direction")
                continue
            occurred = actual_high >= level if direction == "up" else actual_low <= level
            resolved[name] = {
                "level": level, "direction": direction, "probability": prob,
                "occurred": occurred,
                "brier": brier_score(prob, occurred),
                "log_score": log_score(prob, occurred),
            }
        if resolved:
            out["touch"] = resolved

    # --- skill against the baselines ------------------------------------------------------------
    if "crps" in out and baseline_samples:
        base = crps_from_samples(actual_close, baseline_samples)
        out["baseline_crps_empirical"] = base
        out["crps_skill_vs_empirical"] = skill_score(out["crps"], base)
    elif "crps" in out:
        not_scored.append("crps_skill_vs_empirical: no baseline sample set was supplied")

    if "crps" in out and persistence is not None:
        # A degenerate forecast at the anchor price. Its CRPS is simply the absolute error.
        base = abs(persistence - actual_close)
        out["baseline_crps_persistence"] = base
        out["crps_skill_vs_persistence"] = skill_score(out["crps"], base)

    if not_scored:
        out["not_scored"] = not_scored
    return out


# ── aggregation across many forecasts ────────────────────────────────────────────────────────────

def aggregate(scores: Iterable[dict]) -> dict:
    """Roll a set of per-forecast scores into the numbers a scorecard shows.

    Empirical coverage is the figure that matters most and the one most often quietly omitted: it is
    the fraction of outcomes that actually fell inside each stated band, next to the coverage that
    band claimed. A calibrated 90% band lands near 0.90. Ours will not, at first.
    """
    scores = list(scores)
    if not scores:
        return {"count": 0, "note": "no forecast has been scored yet"}

    def mean_of(key: str) -> float | None:
        vals = [s[key] for s in scores if isinstance(s.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else None

    agg: dict = {"count": len(scores), "scorer_version": SCORER_VERSION}
    for key in ("crps", "baseline_crps_empirical", "baseline_crps_persistence"):
        v = mean_of(key)
        if v is not None:
            agg[f"mean_{key}"] = v

    # Skill is recomputed from the aggregate scores rather than averaged. The mean of per-forecast
    # ratios is not the ratio of the means, and the second is the one that answers "over this period,
    # was the model better than the baseline".
    if "mean_crps" in agg and "mean_baseline_crps_empirical" in agg:
        agg["crps_skill_vs_empirical"] = skill_score(agg["mean_crps"],
                                                     agg["mean_baseline_crps_empirical"])
    if "mean_crps" in agg and "mean_baseline_crps_persistence" in agg:
        agg["crps_skill_vs_persistence"] = skill_score(agg["mean_crps"],
                                                       agg["mean_baseline_crps_persistence"])

    pits = [s["pit"] for s in scores if isinstance(s.get("pit"), (int, float))]
    if pits:
        agg["pit"] = {"count": len(pits), "mean": sum(pits) / len(pits),
                      "histogram": _pit_histogram(pits),
                      # The raw values, so a page with too few observations to draw a histogram can
                      # show them instead of drawing a misleading one.
                      "values": [round(p, 4) for p in pits]}

    # Coverage per band.
    band_keys = sorted({k for s in scores for k in (s.get("bands") or {})}, key=float)
    if band_keys:
        cov: dict[str, dict] = {}
        for key in band_keys:
            rows = [s["bands"][key] for s in scores if key in (s.get("bands") or {})]
            hits = sum(1 for r in rows if r["covered"])
            cov[key] = {
                "nominal": float(key),
                "empirical": hits / len(rows),
                "n": len(rows),
                "mean_interval_score": sum(r["interval_score"] for r in rows) / len(rows),
                "mean_width": sum(r["width"] for r in rows) / len(rows),
            }
        agg["coverage"] = cov

    touch_rows = [t for s in scores for t in (s.get("touch") or {}).values()]
    if touch_rows:
        agg["touch"] = {
            "n": len(touch_rows),
            "mean_brier": sum(t["brier"] for t in touch_rows) / len(touch_rows),
            "mean_log_score": sum(t["log_score"] for t in touch_rows) / len(touch_rows),
            "base_rate": sum(1 for t in touch_rows if t["occurred"]) / len(touch_rows),
        }
        # Brier skill against always predicting the base rate. Negative means the probabilities added
        # nothing over knowing how often the event happens.
        rate = agg["touch"]["base_rate"]
        ref = sum(brier_score(rate, t["occurred"]) for t in touch_rows) / len(touch_rows)
        agg["touch"]["brier_skill_vs_base_rate"] = skill_score(agg["touch"]["mean_brier"], ref)

    return agg


def _pit_histogram(pits: Sequence[float], bins: int = 10) -> list[int]:
    """Counts per decile of the PIT. Flat is calibrated; U-shaped is over-confident."""
    counts = [0] * bins
    for p in pits:
        idx = min(int(p * bins), bins - 1)
        counts[idx] += 1
    return counts
