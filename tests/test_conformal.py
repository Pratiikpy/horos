"""Conformal calibration is sold as a guarantee, so it is tested as one.

The central test starts from a model that is deliberately, badly over-confident — bands a fifth of
the width they should be — and asserts that after calibration the 90% band contains close to 90% of
outcomes. That is the claim; anything less is a demonstration that the code runs, not that it works.

The finite-sample correction gets its own test, because the wrong version looks identical, is what
most implementations ship, and under-covers by about 1/n. It is exactly the sort of thing that would
never be noticed in production.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.conformal import (CALIBRATION_WINDOW, MIN_CALIBRATION, Calibrator, conformal_quantile,
                            cqr_scores, empirical_coverage, minimum_n)


def overconfident(rng, n, tightness=0.2, sd=10.0):
    """A model whose bands are `tightness` times as wide as they should be."""
    truth = rng.normal(100.0, sd, size=n)
    half = 1.645 * sd * tightness            # 1.645 sd is the correct 90% half-width
    return (truth - half, truth + half, rng.normal(100.0, sd, size=n))


# ── the guarantee ────────────────────────────────────────────────────────────────────────────────

def test_calibration_recovers_nominal_coverage_from_an_overconfident_model():
    """The headline claim, on a model that starts badly wrong.

    Raw coverage here is around 30% for a band labelled 90%. After calibration it must land near 90%.
    """
    rng = np.random.default_rng(7)
    lo, hi, y = overconfident(rng, 4000)
    cal_lo, cal_hi, cal_y = lo[:2000], hi[:2000], y[:2000]
    test_lo, test_hi, test_y = lo[2000:], hi[2000:], y[2000:]

    raw = empirical_coverage(test_lo, test_hi, test_y)
    assert raw < 0.5, f"the fixture is not over-confident enough to be a real test ({raw:.2f})"

    cal = Calibrator(window=2000).from_history(list(zip(cal_lo, cal_hi, cal_y)), 0.90)
    assert cal.calibrated
    bands = [cal.apply(a, b) for a, b in zip(test_lo, test_hi)]
    after = empirical_coverage([b[0] for b in bands], [b[1] for b in bands], test_y)
    assert after == pytest.approx(0.90, abs=0.03), f"raw {raw:.3f} -> calibrated {after:.3f}"


@pytest.mark.parametrize("coverage", [0.50, 0.80, 0.90])
def test_every_published_band_level_calibrates_to_its_own_nominal(coverage):
    rng = np.random.default_rng(int(coverage * 100))
    lo, hi, y = overconfident(rng, 4000)
    cal = Calibrator(window=2000).from_history(list(zip(lo[:2000], hi[:2000], y[:2000])), coverage)
    bands = [cal.apply(a, b) for a, b in zip(lo[2000:], hi[2000:])]
    after = empirical_coverage([b[0] for b in bands], [b[1] for b in bands], y[2000:])
    assert after == pytest.approx(coverage, abs=0.035)


def test_calibration_tightens_a_model_that_is_too_cautious():
    """Two-directional. A method that only ever widened would be a ratchet, not a calibration."""
    rng = np.random.default_rng(13)
    truth_sd = 10.0
    n = 4000
    half = 1.645 * truth_sd * 4.0                 # four times wider than it needs to be
    centre = rng.normal(100.0, truth_sd, size=n)
    lo, hi = centre - half, centre + half
    y = rng.normal(100.0, truth_sd, size=n)

    cal = Calibrator(window=2000).from_history(list(zip(lo[:2000], hi[:2000], y[:2000])), 0.90)
    assert cal.qhat < 0, "an over-wide band should be tightened, not left alone"
    assert "tightened" in cal.note
    bands = [cal.apply(a, b) for a, b in zip(lo[2000:], hi[2000:])]
    assert empirical_coverage([b[0] for b in bands], [b[1] for b in bands],
                              y[2000:]) == pytest.approx(0.90, abs=0.03)
    assert (bands[0][1] - bands[0][0]) < (hi[0] - lo[0])


def test_a_calibrated_band_is_narrower_than_a_hedged_one_at_the_same_coverage():
    """Calibration is worth something only if it is tighter than simply quoting a huge band."""
    rng = np.random.default_rng(21)
    lo, hi, y = overconfident(rng, 3000, tightness=0.5)
    cal = Calibrator(window=1500).from_history(list(zip(lo[:1500], hi[:1500], y[:1500])), 0.90)
    bands = [cal.apply(a, b) for a, b in zip(lo[1500:], hi[1500:])]
    width = float(np.mean([b[1] - b[0] for b in bands]))
    assert width < 4 * 1.645 * 10.0


# ── the finite-sample correction ─────────────────────────────────────────────────────────────────

def test_the_conformal_quantile_is_above_the_plain_empirical_one():
    """The correction that carries the guarantee.

    `np.quantile(scores, 1-alpha)` is what most implementations use and looks identical. It
    under-covers by roughly 1/n. Ported from meps-cqr.ipynb precisely to avoid that.
    """
    rng = np.random.default_rng(3)
    scores = rng.normal(0, 1, size=50)
    conformal = conformal_quantile(scores, 0.1)
    plain = float(np.quantile(scores, 0.9))
    assert conformal >= plain


def test_the_correction_uses_higher_interpolation_not_linear():
    """`interpolation='higher'` in the source. Linear interpolation loses the guarantee."""
    scores = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    qhat = conformal_quantile(scores, 0.1)
    assert qhat in scores, "the conformal quantile must be an observed score, not interpolated"


def test_no_quantile_exists_below_the_minimum_sample_size():
    """Rather than falling back to a number that looks like a guarantee and is not."""
    assert conformal_quantile([1.0, 2.0], 0.1) is None
    assert conformal_quantile(list(range(9)), 0.1) is not None


@pytest.mark.parametrize("alpha,expected", [(0.5, 1), (0.1, 9), (0.05, 19), (0.01, 99)])
def test_the_minimum_sample_size_is_reported_per_level(alpha, expected):
    """A 99% band needs 99 observations before it can be certified. Callers deserve that number."""
    assert minimum_n(alpha) == expected


def test_an_empty_calibration_set_yields_no_quantile():
    assert conformal_quantile([], 0.1) is None


# ── honest degradation ───────────────────────────────────────────────────────────────────────────

def test_an_uncalibrated_band_is_returned_untouched_and_says_so():
    """The failure that matters: never pass a raw band off as calibrated."""
    cal = Calibrator().from_history([(90.0, 110.0, 100.0)] * 3, 0.90)
    assert not cal.calibrated
    assert cal.qhat is None
    assert cal.apply(90.0, 110.0) == [90.0, 110.0]
    assert "carries no coverage guarantee" in cal.note


def test_the_uncalibrated_note_says_how_many_more_are_needed():
    cal = Calibrator().from_history([(90.0, 110.0, 100.0)] * 5, 0.90)
    assert "5 scored forecasts" in cal.note
    assert str(MIN_CALIBRATION) in cal.note


def test_an_empty_history_is_uncalibrated_not_an_error():
    """Day one. The scorecard publishes empty and the bands say they are raw."""
    cal = Calibrator().from_history([], 0.90)
    assert not cal.calibrated and cal.n == 0


def test_the_dict_form_always_states_whether_it_was_calibrated():
    """This block goes into every response, so it must never be ambiguous."""
    for history in ([], [(90.0, 110.0, 100.0)] * 500):
        d = Calibrator().from_history(history, 0.90).as_dict()
        assert "calibrated" in d and isinstance(d["calibrated"], bool)
        assert d["method"].startswith("conformalised quantile regression")


# ── scores ───────────────────────────────────────────────────────────────────────────────────────

def test_the_score_is_the_signed_distance_outside_the_band():
    s = cqr_scores([90.0, 90.0, 90.0], [110.0, 110.0, 110.0], [100.0, 115.0, 85.0])
    assert s[0] == pytest.approx(-10.0)      # comfortably inside
    assert s[1] == pytest.approx(5.0)        # above the top
    assert s[2] == pytest.approx(5.0)        # below the bottom


def test_scores_reject_mismatched_lengths():
    with pytest.raises(ValueError):
        cqr_scores([1.0, 2.0], [3.0], [4.0])


def test_calibration_uses_only_the_recent_window():
    """Market returns are not exchangeable across regimes, so old observations are dropped."""
    old = [(99.0, 101.0, 100.0)] * 500          # a calm regime, tiny misses
    new = [(50.0, 150.0, 300.0)] * 100          # a violent one, huge misses
    cal = Calibrator(window=100).from_history(old + new, 0.90)
    assert cal.n == 100
    assert cal.qhat > 100, "the calibration is still being dragged by the old regime"


def test_the_window_default_is_about_eight_days_of_hourly_forecasts():
    assert CALIBRATION_WINDOW == 200
    assert CALIBRATION_WINDOW / 24 == pytest.approx(8.33, abs=0.01)


# ── from the ledger ──────────────────────────────────────────────────────────────────────────────

def test_calibration_reads_the_ledger_and_skips_voided_forecasts(tmp_path):
    """A voided forecast must not influence calibration; it was withdrawn for a reason."""
    from core.crypto import Signer
    from core.ledger import Ledger

    led = Ledger(tmp_path / "l.jsonl", Signer(tmp_path / "k"))
    rng = np.random.default_rng(5)
    voided_id = None
    for i in range(60):
        e = led.record_forecast(
            symbol="BTC/USDT:USDT", timeframe="1h", horizon="1h", model="m", model_version="v",
            issued_for=f"2026-07-{(i % 28) + 1:02d}T00:00:00Z",
            window={"open": "2026-07-01T00:00:00Z", "close": f"2026-07-{(i % 28) + 1:02d}T01:00:00Z"},
            distribution={"bands": {"0.9": [99.0, 101.0]}},
            inputs={"i": i})
        actual = 100.0 + float(rng.normal(0, 3))
        if i == 0:
            voided_id = e.forecast_id
            led.record_void(forecast_id=e.forecast_id, reason="test")
            actual = 100_000.0          # an outlier that would wreck calibration if counted
        led.record_score(forecast_id=e.forecast_id, actual={"close": actual},
                         scores={"crps": 1.0}, scorer_version="1.0.0")

    cal = Calibrator().from_ledger(led, "BTC/USDT:USDT", "1h", 0.90)
    assert cal.calibrated
    assert cal.n == 59, "the voided forecast should not be in the calibration set"
    assert cal.qhat < 100, "the voided outlier leaked into calibration"
    assert voided_id in led.voided_ids()
