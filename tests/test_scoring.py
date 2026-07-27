"""The scorer is the one thing that must be right.

Every published number comes out of these functions, so they are tested against closed forms and
against deliberately naive reimplementations rather than against remembered values. A test that only
asserts "the number did not change" would pass just as happily on a wrong number.
"""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from core.scoring import (BAND_LEVELS, QUANTILE_LEVELS, aggregate, brier_score, covered,
                          crps_from_quantiles, crps_from_samples, empirical_return_samples,
                          interval_score, log_score, pinball_by_level, pinball_loss,
                          probability_integral_transform, score_forecast, skill_score)


def naive_crps(actual: float, samples, fair: bool = True) -> float:
    """The definition, written out with two loops. Slow, obviously correct, the reference."""
    x = list(map(float, samples))
    m = len(x)
    term1 = sum(abs(v - actual) for v in x) / m
    pair = sum(abs(a - b) for a in x for b in x)
    denom = 2 * m * (m - 1) if fair else 2 * m * m
    return term1 - pair / denom


# ── CRPS ─────────────────────────────────────────────────────────────────────────────────────────

def test_fast_crps_matches_the_naive_double_loop():
    rng = random.Random(11)
    for _ in range(30):
        m = rng.randint(2, 60)
        samples = [rng.gauss(100, 7) for _ in range(m)]
        actual = rng.gauss(100, 7)
        assert crps_from_samples(actual, samples) == pytest.approx(naive_crps(actual, samples),
                                                                   rel=1e-12, abs=1e-9)


def test_the_fair_estimator_subtracts_more_so_it_reads_lower():
    """Arithmetic first: m(m-1) < m², so the fair estimator subtracts the larger spread term."""
    samples = [99.0, 100.0, 101.0, 100.5]
    fair = crps_from_samples(100.2, samples, fair=True)
    biased = crps_from_samples(100.2, samples, fair=False)
    assert fair < biased
    assert fair == pytest.approx(naive_crps(100.2, samples, fair=True))
    assert biased == pytest.approx(naive_crps(100.2, samples, fair=False))


def test_the_biased_estimator_inflates_a_wide_ensemble_more_than_a_narrow_one():
    """This is the actual bias, and the reason the fair estimator is the one we publish.

    Both estimators inflate, by ``E|X−X'|/2m`` — a term proportional to ensemble spread. So a wide
    ensemble is inflated more than a narrow one, and the biased estimator therefore ranks
    under-dispersed forecasts better than they deserve. For an accountability product that is the
    wrong thumb on the scale, so we carry the unbiased one.
    """
    rng = np.random.default_rng(101)
    narrow = list(rng.normal(100, 1.0, size=12))
    wide = list(rng.normal(100, 10.0, size=12))
    narrow_gap = crps_from_samples(100.0, narrow, fair=False) - crps_from_samples(100.0, narrow)
    wide_gap = crps_from_samples(100.0, wide, fair=False) - crps_from_samples(100.0, wide)
    assert wide_gap > narrow_gap * 5


def test_the_fair_estimator_can_go_negative_on_a_tiny_ensemble():
    """A documented property, pinned so nobody later "fixes" it by clamping at zero.

    Clamping would reintroduce exactly the upward bias the fair estimator exists to remove. The
    protection is a minimum ensemble size at issue time, not a floor on the score.
    """
    from core.scoring import MIN_ENSEMBLE
    assert crps_from_samples(100.0, [99.0, 100.0, 101.0]) < 1e-9
    assert MIN_ENSEMBLE >= 200


def test_crps_of_a_single_sample_is_the_absolute_error():
    assert crps_from_samples(105.0, [100.0]) == pytest.approx(5.0)


def test_crps_is_zero_for_a_perfect_point_forecast():
    assert crps_from_samples(100.0, [100.0]) == pytest.approx(0.0)


def test_crps_rewards_the_better_centred_ensemble():
    actual = 100.0
    close = [99.0, 100.0, 101.0]
    far = [109.0, 110.0, 111.0]
    assert crps_from_samples(actual, close) < crps_from_samples(actual, far)


def test_crps_penalises_an_ensemble_that_is_wider_than_it_needs_to_be():
    """Ensembles at the size we actually issue at, because the property is asymptotic.

    On three points the fair estimator collapses to ~0 for any symmetric ensemble centred on the
    outcome, which says nothing about dispersion. At 400 paths the penalty is unambiguous.
    """
    rng = np.random.default_rng(41)
    tight = list(rng.normal(100.0, 0.5, size=400))
    loose = list(rng.normal(100.0, 20.0, size=400))
    assert crps_from_samples(100.0, tight) < crps_from_samples(100.0, loose)


def test_crps_empty_sample_set_raises_rather_than_returning_zero():
    with pytest.raises(ValueError):
        crps_from_samples(100.0, [])


def test_crps_from_quantiles_approximates_the_sample_estimator():
    """The identity CRPS = 2 integral of pinball loss, checked against the exact estimator.

    A normal ensemble is scored both ways; the trapezoid over nine levels should land close to the
    exact figure. This is what justifies using the quantile route when samples were not recorded.
    """
    rng = np.random.default_rng(3)
    samples = rng.normal(100, 5, size=4000)
    quantiles = [float(np.quantile(samples, q)) for q in QUANTILE_LEVELS]
    exact = crps_from_samples(101.0, samples)
    approx = crps_from_quantiles(101.0, quantiles, QUANTILE_LEVELS)
    assert approx == pytest.approx(exact, rel=0.10)


# ── pinball ──────────────────────────────────────────────────────────────────────────────────────

def test_pinball_is_minimised_at_the_true_quantile():
    """The defining property. If this fails the quantiles we publish are not quantiles."""
    rng = np.random.default_rng(5)
    draws = rng.normal(0, 1, size=200_000)
    for level in (0.1, 0.5, 0.9):
        true_q = float(np.quantile(draws, level))
        best = float(np.mean([pinball_loss(y, true_q, level) for y in draws[:20_000]]))
        for offset in (-0.3, -0.1, 0.1, 0.3):
            worse = float(np.mean([pinball_loss(y, true_q + offset, level)
                                   for y in draws[:20_000]]))
            assert worse > best


def test_pinball_at_the_median_is_half_the_absolute_error():
    assert pinball_loss(110.0, 100.0, 0.5) == pytest.approx(5.0)
    assert pinball_loss(90.0, 100.0, 0.5) == pytest.approx(5.0)


def test_pinball_is_asymmetric_in_the_direction_the_level_implies():
    """A 90% quantile should be cheap to overshoot and expensive to undershoot."""
    over = pinball_loss(100.0, 110.0, 0.9)     # predicted above the outcome
    under = pinball_loss(110.0, 100.0, 0.9)    # predicted below the outcome
    assert under > over


def test_pinball_by_level_reports_every_level_and_the_mean():
    q = [90, 92, 94, 97, 100, 103, 106, 108, 110]
    out = pinball_by_level(100.0, q, QUANTILE_LEVELS)
    assert set(out) == {f"{lv:g}" for lv in QUANTILE_LEVELS} | {"mean"}
    manual = sum(pinball_loss(100.0, p, lv) for p, lv in zip(q, QUANTILE_LEVELS)) / len(q)
    assert out["mean"] == pytest.approx(manual)


def test_pinball_by_level_refuses_a_mismatched_grid():
    with pytest.raises(ValueError):
        pinball_by_level(100.0, [1, 2, 3], QUANTILE_LEVELS)


# ── interval score ───────────────────────────────────────────────────────────────────────────────

def test_interval_score_of_a_covering_band_is_just_its_width():
    assert interval_score(100.0, 95.0, 105.0, 0.9) == pytest.approx(10.0)


def test_interval_score_charges_a_miss_in_proportion_to_the_distance():
    near = interval_score(106.0, 95.0, 105.0, 0.9)
    far = interval_score(115.0, 95.0, 105.0, 0.9)
    assert far > near > 10.0


def test_a_wider_band_cannot_win_by_hedging():
    """The property that makes coverage ungameable.

    A band inflated to guarantee coverage pays for every unit of that width, so it loses to a
    narrower band that also covered.
    """
    tight = interval_score(100.0, 98.0, 102.0, 0.9)
    absurd = interval_score(100.0, 0.0, 1_000.0, 0.9)
    assert tight < absurd


def test_a_narrow_band_that_misses_loses_to_a_wide_one_that_covers():
    """And the converse, so neither direction is free."""
    narrow_miss = interval_score(120.0, 99.0, 101.0, 0.9)
    wide_hit = interval_score(120.0, 80.0, 140.0, 0.9)
    assert wide_hit < narrow_miss


def test_interval_score_rejects_an_impossible_coverage():
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            interval_score(100.0, 95.0, 105.0, bad)


def test_covered_includes_the_endpoints():
    assert covered(95.0, 95.0, 105.0)
    assert covered(105.0, 95.0, 105.0)
    assert not covered(94.99, 95.0, 105.0)


# ── probability scores ───────────────────────────────────────────────────────────────────────────

def test_brier_is_zero_when_certain_and_right_and_one_when_certain_and_wrong():
    assert brier_score(1.0, True) == pytest.approx(0.0)
    assert brier_score(0.0, True) == pytest.approx(1.0)
    assert brier_score(0.5, True) == pytest.approx(0.25)


def test_brier_rejects_a_probability_outside_the_unit_interval():
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError):
            brier_score(bad, True)


def test_log_score_punishes_a_confident_miss_far_harder_than_brier_does():
    brier_gap = brier_score(0.99, False) - brier_score(0.6, False)
    log_gap = log_score(0.99, False) - log_score(0.6, False)
    assert log_gap > brier_gap * 3


def test_log_score_is_floored_so_one_certainty_cannot_make_the_aggregate_infinite():
    assert math.isfinite(log_score(1.0, False))
    assert math.isfinite(log_score(0.0, True))


# ── PIT ──────────────────────────────────────────────────────────────────────────────────────────

def test_pit_is_uniform_when_the_forecast_distribution_is_correct():
    """The calibration diagnostic, verified on a distribution that is calibrated by construction."""
    rng = np.random.default_rng(17)
    pits = []
    for _ in range(4000):
        samples = rng.normal(0, 1, size=200)
        outcome = float(rng.normal(0, 1))
        pits.append(probability_integral_transform(outcome, samples))
    # A uniform sample has mean 1/2 and variance 1/12.
    assert float(np.mean(pits)) == pytest.approx(0.5, abs=0.02)
    assert float(np.var(pits)) == pytest.approx(1 / 12, abs=0.01)


def test_pit_is_u_shaped_when_the_forecast_is_too_narrow():
    """Over-confidence is the failure mode this diagnostic exists to expose."""
    rng = np.random.default_rng(19)
    pits = []
    for _ in range(3000):
        samples = rng.normal(0, 0.2, size=200)      # claims far more certainty than the truth
        outcome = float(rng.normal(0, 1))
        pits.append(probability_integral_transform(outcome, samples))
    extremes = sum(1 for p in pits if p < 0.1 or p > 0.9) / len(pits)
    assert extremes > 0.5                            # a calibrated forecast would give ~0.2


# ── baselines and skill ──────────────────────────────────────────────────────────────────────────

def test_empirical_baseline_applies_past_returns_to_the_anchor():
    out = empirical_return_samples(100.0, [0.0, math.log(1.1), math.log(0.9)])
    assert out[0] == pytest.approx(100.0)
    assert out[1] == pytest.approx(110.0)
    assert out[2] == pytest.approx(90.0)


def test_empirical_baseline_keeps_the_most_recent_returns_when_truncating():
    returns = [math.log(1 + i / 100) for i in range(10)]
    out = empirical_return_samples(100.0, returns, count=3)
    assert len(out) == 3
    assert out[-1] == pytest.approx(100.0 * math.exp(returns[-1]))


def test_empirical_baseline_refuses_an_empty_history():
    with pytest.raises(ValueError):
        empirical_return_samples(100.0, [])


def test_skill_is_positive_only_when_the_model_beats_the_baseline():
    assert skill_score(5.0, 10.0) == pytest.approx(0.5)
    assert skill_score(10.0, 10.0) == pytest.approx(0.0)
    assert skill_score(20.0, 10.0) == pytest.approx(-1.0)


def test_skill_against_a_perfect_baseline_is_none_not_a_flattering_number():
    """A zero baseline makes the ratio undefined; inventing a value there would favour us."""
    assert skill_score(3.0, 0.0) is None


# ── the whole-forecast scorer ────────────────────────────────────────────────────────────────────

def _distribution(anchor: float = 100.0, seed: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    samples = list(rng.normal(anchor, 3.0, size=200))
    quantiles = [float(np.quantile(samples, q)) for q in QUANTILE_LEVELS]
    bands = {}
    for level in BAND_LEVELS:
        lo = float(np.quantile(samples, (1 - level) / 2))
        hi = float(np.quantile(samples, 1 - (1 - level) / 2))
        bands[f"{level:g}"] = [lo, hi]
    return {"anchor_price": anchor, "samples": samples,
            "quantile_levels": list(QUANTILE_LEVELS), "quantiles": quantiles, "bands": bands,
            "touch": {"up_105": {"level": 105.0, "direction": "up", "probability": 0.3},
                      "down_95": {"level": 95.0, "direction": "down", "probability": 0.3}}}


def test_score_forecast_produces_every_family_of_score():
    out = score_forecast(actual_close=101.0, actual_high=106.0, actual_low=99.0,
                         distribution=_distribution())
    assert out["crps"] > 0
    assert out["crps_estimator"] == "fair (Ferro)"
    assert 0.0 <= out["pit"] <= 1.0
    assert "mean" in out["pinball"]
    assert set(out["bands"]) == {"0.5", "0.8", "0.9"}
    assert out["touch"]["up_105"]["occurred"] is True       # the high reached 106
    assert out["touch"]["down_95"]["occurred"] is False     # the low stopped at 99


def test_touch_resolves_against_the_path_not_the_close():
    """A candle that spiked through a level and came back did touch it.

    Resolving on the close would let a forecast that said "3% chance of touching 105" escape being
    charged for a wick to 106, which is exactly the outcome the buyer cared about.
    """
    out = score_forecast(actual_close=100.0, actual_high=106.0, actual_low=94.0,
                         distribution=_distribution())
    assert out["touch"]["up_105"]["occurred"] is True
    assert out["touch"]["down_95"]["occurred"] is True


def test_a_missing_component_is_reported_as_not_scored_never_as_a_zero():
    """The failure this whole product exists to prevent, checked at the level of one forecast."""
    out = score_forecast(actual_close=101.0, actual_high=102.0, actual_low=100.0,
                         distribution={"anchor_price": 100.0})
    assert "crps" not in out
    assert "bands" not in out
    reasons = " ".join(out["not_scored"])
    assert "crps" in reasons and "bands" in reasons


def test_a_malformed_band_is_named_rather_than_silently_dropped():
    dist = _distribution()
    dist["bands"]["0.7"] = "not a pair"
    out = score_forecast(actual_close=101.0, actual_high=102.0, actual_low=100.0, distribution=dist)
    assert "0.7" not in out["bands"]
    assert any("0.7" in r for r in out["not_scored"])


def test_skill_is_absent_and_explained_when_no_baseline_was_supplied():
    out = score_forecast(actual_close=101.0, actual_high=102.0, actual_low=100.0,
                         distribution=_distribution())
    assert "crps_skill_vs_empirical" not in out
    assert any("baseline" in r for r in out["not_scored"])


def test_skill_is_computed_against_both_baselines_when_they_are_supplied():
    dist = _distribution()
    baseline = list(np.random.default_rng(9).normal(100.0, 12.0, size=300))
    out = score_forecast(actual_close=101.0, actual_high=102.0, actual_low=100.0,
                         distribution=dist, baseline_samples=baseline, persistence=100.0)
    assert out["crps_skill_vs_empirical"] > 0      # a 3-wide forecast beats a 12-wide one here
    assert "crps_skill_vs_persistence" in out


def test_quantile_only_forecast_is_scored_but_says_the_pit_is_unavailable():
    dist = _distribution()
    dist.pop("samples")
    out = score_forecast(actual_close=101.0, actual_high=102.0, actual_low=100.0, distribution=dist)
    assert out["crps_estimator"] == "quantile trapezoid"
    assert "pit" not in out
    assert any("pit" in r for r in out["not_scored"])


# ── aggregation ──────────────────────────────────────────────────────────────────────────────────

def test_aggregate_of_nothing_says_so_rather_than_reporting_zeros():
    out = aggregate([])
    assert out["count"] == 0
    assert "no forecast" in out["note"]


def test_empirical_coverage_recovers_the_nominal_level_on_calibrated_forecasts():
    """The headline number of the scorecard, checked on forecasts calibrated by construction.

    Draw the outcome from the same distribution the bands were taken from; the 90% band must then
    contain about 90% of outcomes. If this test drifts, every coverage figure we publish is wrong.
    """
    rng = np.random.default_rng(23)
    scores = []
    for _ in range(1500):
        samples = rng.normal(100, 4, size=400)
        bands = {f"{lv:g}": [float(np.quantile(samples, (1 - lv) / 2)),
                             float(np.quantile(samples, 1 - (1 - lv) / 2))] for lv in BAND_LEVELS}
        outcome = float(rng.normal(100, 4))
        scores.append(score_forecast(actual_close=outcome, actual_high=outcome, actual_low=outcome,
                                     distribution={"samples": list(samples), "bands": bands}))
    agg = aggregate(scores)
    for level in BAND_LEVELS:
        assert agg["coverage"][f"{level:g}"]["empirical"] == pytest.approx(level, abs=0.035)


def test_aggregate_skill_is_the_ratio_of_means_not_the_mean_of_ratios():
    """These differ, and only the first answers 'over this period, did the model beat the baseline'."""
    scores = [{"crps": 1.0, "baseline_crps_empirical": 2.0},
              {"crps": 10.0, "baseline_crps_empirical": 10.0}]
    agg = aggregate(scores)
    assert agg["mean_crps"] == pytest.approx(5.5)
    assert agg["mean_baseline_crps_empirical"] == pytest.approx(6.0)
    assert agg["crps_skill_vs_empirical"] == pytest.approx(1 - 5.5 / 6.0)
    mean_of_ratios = ((1 - 1 / 2) + (1 - 10 / 10)) / 2
    assert agg["crps_skill_vs_empirical"] != pytest.approx(mean_of_ratios)


def test_aggregate_reports_the_pit_histogram():
    rng = np.random.default_rng(29)
    scores = [{"pit": float(p)} for p in rng.uniform(0, 1, size=1000)]
    agg = aggregate(scores)
    assert sum(agg["pit"]["histogram"]) == 1000
    assert len(agg["pit"]["histogram"]) == 10


def test_touch_brier_skill_is_negative_when_the_probabilities_add_nothing():
    """A forecaster who ignores the question and guesses should score no better than the base rate."""
    rng = random.Random(31)
    rows = []
    for _ in range(500):
        occurred = rng.random() < 0.3
        rows.append({"touch": {"x": {"probability": rng.random(), "occurred": occurred,
                                     "brier": brier_score(rng.random(), occurred),
                                     "log_score": 0.0}}})
    agg = aggregate(rows)
    assert agg["touch"]["brier_skill_vs_base_rate"] < 0
