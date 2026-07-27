"""Conformalised quantile regression — making a stated 90% band actually contain 90%.

Raw generative-model quantiles are almost always miscalibrated. The model's "90% band" is whatever
its sampling produced, and there is no reason for it to contain 90% of outcomes; in practice
autoregressive samplers run narrow, so the honest-looking band is over-confident.

Conformal prediction fixes this **without assuming anything about the distribution**. Take the
forecasts already scored in the ledger, measure by how much each band missed, and widen every future
band by the appropriate quantile of those misses. The guarantee that falls out is finite-sample and
distribution-free: given exchangeable data, the calibrated band contains the outcome with probability
at least 1−α.

Ported from `repos/aangelopoulos_conformal/notebooks/meps-cqr.ipynb`, which is the reference
implementation for this exact case:

    cal_scores = np.maximum(cal_labels - cal_upper, cal_lower - cal_labels)
    qhat = np.quantile(cal_scores, np.ceil((n+1)*(1-alpha))/n, interpolation='higher')
    prediction_sets = [val_lower - qhat, val_upper + qhat]

**The `ceil((n+1)(1−α))/n` and `interpolation='higher'` are the whole guarantee.** Using a plain
`np.quantile(scores, 1−α)` looks identical, is what most implementations ship, and silently forfeits
the finite-sample coverage property — the resulting band under-covers by roughly 1/n. That is exactly
the kind of detail that is invisible until someone checks, which is why it is ported from the source
rather than written from memory.

**Two honest caveats, both surfaced in the output rather than buried.**

*Exchangeability.* The guarantee assumes calibration and test data are exchangeable. Market returns
are not: volatility clusters, regimes shift. So calibration is drawn from a rolling recent window
rather than all history, which trades some of the guarantee's strictness for responsiveness to the
current regime. The window length is stated in every calibrated response.

*Sample size.* Below a floor, `ceil((n+1)(1−α))/n` exceeds 1 and no finite quantile exists — the
method is telling you it cannot certify anything yet. For α=0.1 that floor is 9 observations, and it
rises as the band gets tighter. Under it, the band is returned **uncalibrated and labelled as such**,
never silently passed through as though it had been corrected.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

CALIBRATOR_VERSION = "1.0.0"

# How many recent scored forecasts feed calibration, per (symbol, horizon, band). Long enough for a
# stable quantile, short enough to track a regime change. 200 hourly observations is ~8 days.
CALIBRATION_WINDOW = 200

# Below this the correction is not applied and the band is served raw and flagged. It is the point
# at which ceil((n+1)(1-alpha))/n <= 1 has meaningful headroom rather than being exactly attainable.
MIN_CALIBRATION = 20


def minimum_n(alpha: float) -> int:
    """Fewest observations for which a conformal quantile at this level exists at all.

    From the requirement ``ceil((n+1)(1−α)) <= n``. At α=0.1 this is 9; at α=0.5 it is 1; at α=0.01
    it is 99. Reported rather than assumed, because the answer to "why is my 99% band uncalibrated"
    is usually "you have not made 99 forecasts yet".
    """
    n = 1
    while n < 100_000:
        if math.ceil((n + 1) * (1.0 - alpha)) <= n:
            return n
        n += 1
    return n


def conformal_quantile(scores: Sequence[float], alpha: float) -> float | None:
    """The finite-sample-corrected quantile of the nonconformity scores.

    Returns None when `n` is too small for the level to exist, rather than falling back to the plain
    empirical quantile — a silent fallback would hand back a number carrying no guarantee while
    looking exactly like one that does.
    """
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0:
        return None
    level = math.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return None
    return float(np.quantile(s, level, method="higher"))


def cqr_scores(lower: Sequence[float], upper: Sequence[float],
               actual: Sequence[float]) -> np.ndarray:
    """Nonconformity score for each past forecast: how far outside its own band the outcome fell.

    Negative when the outcome was comfortably inside — that is not discarded, because those are the
    observations that let the band *shrink* when the model has been too cautious. Calibration is
    two-directional; a method that only ever widened would be a ratchet.
    """
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    y = np.asarray(actual, dtype=float)
    if not (lo.shape == hi.shape == y.shape):
        raise ValueError("lower, upper and actual must be the same length")
    return np.maximum(y - hi, lo - y)


@dataclass(frozen=True)
class Calibration:
    """The correction for one (symbol, horizon, coverage), and what it rests on."""
    coverage: float
    n: int
    qhat: float | None
    window: int
    calibrated: bool
    note: str

    def apply(self, lower: float, upper: float) -> list[float]:
        """Widen — or tighten — a raw band by the measured correction."""
        if not self.calibrated or self.qhat is None:
            return [lower, upper]
        lo, hi = lower - self.qhat, upper + self.qhat
        # A correction large enough to invert the band means the calibration set disagrees with the
        # model completely. Collapsing to a point is the honest degenerate case, not a crash.
        return [min(lo, hi), max(lo, hi)]

    def as_dict(self) -> dict:
        return {"coverage": self.coverage, "calibrated": self.calibrated,
                "calibration_n": self.n, "calibration_window": self.window,
                "qhat": self.qhat, "note": self.note,
                "method": "conformalised quantile regression (split conformal)",
                "version": CALIBRATOR_VERSION}


class Calibrator:
    """Builds corrections from the ledger's own scored history.

    Deliberately fed from the ledger and nothing else. Calibrating against a held-out set we chose
    would mean the guarantee rested on a choice nobody can check; calibrating against the published
    record means anyone holding the ledger can recompute every correction we applied.
    """

    def __init__(self, window: int = CALIBRATION_WINDOW, min_n: int = MIN_CALIBRATION):
        self.window = window
        self.min_n = min_n

    def from_history(self, history: Sequence[tuple[float, float, float]],
                     coverage: float) -> Calibration:
        """`history` is (lower, upper, actual) for past forecasts at this coverage level."""
        alpha = 1.0 - coverage
        recent = list(history)[-self.window:]
        n = len(recent)
        floor = max(self.min_n, minimum_n(alpha))

        if n < floor:
            return Calibration(
                coverage=coverage, n=n, qhat=None, window=self.window, calibrated=False,
                note=(f"not calibrated: {n} scored forecasts are available and a {coverage:g} band "
                      f"needs at least {floor} for a distribution-free guarantee to exist. The band "
                      f"below is the model's raw output and carries no coverage guarantee."))

        scores = cqr_scores([h[0] for h in recent], [h[1] for h in recent],
                            [h[2] for h in recent])
        qhat = conformal_quantile(scores, alpha)
        if qhat is None:                                          # pragma: no cover — floor covers it
            return Calibration(coverage, n, None, self.window, False,
                               f"not calibrated: no conformal quantile exists at n={n}")
        direction = "widened" if qhat > 0 else "tightened"
        return Calibration(
            coverage=coverage, n=n, qhat=qhat, window=self.window, calibrated=True,
            note=(f"calibrated on the {n} most recent scored forecasts for this symbol and horizon; "
                  f"the raw band was {direction} by {abs(qhat):.4f} in price terms. Split conformal, "
                  f"so the guarantee is finite-sample and assumes only exchangeability."))

    def from_ledger(self, ledger, symbol: str, horizon: str, coverage: float) -> Calibration:
        """Pull the history for one (symbol, horizon, coverage) straight out of the ledger."""
        key = f"{coverage:g}"
        issued: dict[str, dict] = {}
        for e in ledger:
            if e.kind == "issued" and e.body.get("symbol") == symbol \
                    and e.body.get("horizon") == horizon:
                band = ((e.body.get("distribution") or {}).get("bands") or {}).get(key)
                if band:
                    issued[e.forecast_id] = {"band": band}
        voided = ledger.voided_ids()
        history: list[tuple[float, float, float]] = []
        for e in ledger:
            if e.kind != "scored" or e.forecast_id in voided:
                continue
            row = issued.get(e.forecast_id)
            if not row:
                continue
            close = (e.body.get("actual") or {}).get("close")
            if close is None:
                continue
            history.append((float(row["band"][0]), float(row["band"][1]), float(close)))
        return self.from_history(history, coverage)


def empirical_coverage(lower: Sequence[float], upper: Sequence[float],
                       actual: Sequence[float]) -> float:
    """Fraction of outcomes inside the bands. The thing calibration is supposed to fix."""
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    y = np.asarray(actual, dtype=float)
    if y.size == 0:
        raise ValueError("no observations")
    return float(np.mean((y >= lo) & (y <= hi)))
