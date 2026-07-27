"""The served set, and the window arithmetic every forecast depends on.

Window alignment is the kind of thing that looks obviously right and is off by one candle. An
off-by-one here means every forecast is scored against the wrong bar — quietly, consistently, and in
a way that would make the whole record meaningless. So the boundaries are checked explicitly.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from core.markets import (BOOK_NOTIONAL_24H_USD, CONTEXT_CANDLES, HORIZONS, SCORED_MARKETS,
                          SCORED_TIMEFRAME, by_symbol, context_span, coverage_note,
                          forecasts_per_week, is_scored, window_for_anchor)


def at(iso: str) -> datetime:
    return datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def ms(iso: str) -> int:
    return int(at(iso).timestamp() * 1000)


# ── the served set ───────────────────────────────────────────────────────────────────────────────

def test_the_scored_set_is_three_markets_on_one_timeframe():
    assert len(SCORED_MARKETS) == 3
    assert SCORED_TIMEFRAME == "1h"
    assert {m.symbol for m in SCORED_MARKETS} == {"ETH/USDT:USDT", "BTC/USDT:USDT", "SOL/USDT:USDT"}


def test_the_shares_add_up_against_the_measured_book():
    """Each market's share must be consistent with its own notional and the measured total."""
    for m in SCORED_MARKETS:
        assert m.notional_24h_usd / BOOK_NOTIONAL_24H_USD == pytest.approx(m.share_of_book,
                                                                           abs=0.002)


def test_eth_and_btc_are_two_thirds_of_the_book_which_is_why_they_are_first():
    eth = by_symbol("ETH/USDT:USDT")
    btc = by_symbol("BTC/USDT:USDT")
    assert eth.share_of_book + btc.share_of_book > 0.65


def test_markets_resolve_by_unified_symbol_or_okx_instrument_id():
    assert by_symbol("BTC/USDT:USDT").okx_inst == "BTC-USDT-SWAP"
    assert by_symbol("BTC-USDT-SWAP").symbol == "BTC/USDT:USDT"
    assert by_symbol("DOGE/USDT:USDT") is None


def test_an_unscored_symbol_is_answerable_but_not_claimed_to_have_a_record():
    """Refusing to price an off-list symbol would be worse for the caller.

    But saying "we have an accuracy record" and "we have one for *this*" are different claims, and
    the response has to be able to tell them apart.
    """
    assert is_scored("ETH/USDT:USDT")
    assert not is_scored("ARB/USDT:USDT")


# ── context ──────────────────────────────────────────────────────────────────────────────────────

def test_the_context_window_is_three_weeks_of_hourly_candles():
    """512 hourly candles is 21 days 8 hours — the justification for choosing 1h over 5m or 1d."""
    span = context_span()
    assert CONTEXT_CANDLES == 512
    assert span.days == 21
    assert span.seconds // 3600 == 8


def test_the_record_fills_fast_enough_to_mean_something():
    """At 1h the ledger takes 2,016 scored forecasts a week; at 1d it would take 84."""
    assert forecasts_per_week() == 168 * 3 * 4 == 2016
    assert forecasts_per_week(timeframe="1d") == 84


# ── window arithmetic ────────────────────────────────────────────────────────────────────────────

ANCHOR = ms("2026-07-27T11:00:00Z")     # the last CLOSED candle, covering 11:00-12:00


def test_the_window_opens_at_the_candle_the_model_predicts_first():
    """The regression that mattered most.

    The model is fed closed candles and its first predicted step is the candle *following* the last
    closed one — the bar in progress. An earlier version derived the window from wall-clock "now" and
    took the next boundary after it, which pointed one candle too far ahead. Every forecast would
    then have been scored against a bar the model never predicted: silently, consistently, and while
    looking entirely healthy.
    """
    w = window_for_anchor(ANCHOR, "1h")
    assert w["open"] == "2026-07-27T12:00:00Z"      # the bar right after the 11:00 candle closed
    assert w["close"] == "2026-07-27T13:00:00Z"
    assert w["candles"] == 1


def test_the_window_does_not_depend_on_when_the_job_ran():
    """A function of the data, not of the clock — so a slow or retried round cannot shift it."""
    assert window_for_anchor(ANCHOR, "4h") == window_for_anchor(ANCHOR, "4h")
    assert window_for_anchor(ANCHOR, "4h")["open"] != window_for_anchor(ANCHOR + 3_600_000,
                                                                        "4h")["open"]


@pytest.mark.parametrize("horizon,expected_close", [
    ("1h", "2026-07-27T13:00:00Z"),
    ("4h", "2026-07-27T16:00:00Z"),
    ("12h", "2026-07-28T00:00:00Z"),
    ("24h", "2026-07-28T12:00:00Z"),
])
def test_every_horizon_closes_where_the_candle_count_says_it_should(horizon, expected_close):
    w = window_for_anchor(ANCHOR, horizon)
    assert w["open"] == "2026-07-27T12:00:00Z"
    assert w["close"] == expected_close


def test_the_window_spans_exactly_the_horizon_in_candles():
    for horizon, steps in HORIZONS.items():
        w = window_for_anchor(ANCHOR, horizon)
        assert ms(w["close"]) - ms(w["open"]) == steps * 3_600_000
        assert w["candles"] == steps


def test_windows_are_aligned_so_a_published_candle_always_exists_to_score_against():
    """Both edges land on exchange boundaries, so the scorer never interpolates an outcome."""
    for horizon in HORIZONS:
        w = window_for_anchor(ANCHOR, horizon)
        for edge in ("open", "close"):
            assert w[edge].endswith(":00:00Z")


def test_the_anchor_candle_is_carried_so_the_window_can_be_rechecked():
    assert window_for_anchor(ANCHOR, "1h")["anchor_candle_ts"] == ANCHOR


def test_an_unknown_horizon_is_refused():
    with pytest.raises(ValueError, match="unknown horizon"):
        window_for_anchor(ANCHOR, "3h")


# ── the provenance block ─────────────────────────────────────────────────────────────────────────

def test_the_coverage_note_carries_the_measurement_behind_the_selection():
    """Served rather than written, so the justification cannot drift from the served set."""
    note = coverage_note()
    assert len(note["symbols"]) == len(SCORED_MARKETS)
    assert note["selection"]["measured_on"] == "2026-07-27"
    assert note["selection"]["instruments_measured"] == 415
    assert note["horizons"] == ["1h", "4h", "12h", "24h"]
    assert note["scored_forecasts_per_week"] == 2016


# ── the regression that started this ─────────────────────────────────────────────────────────────

@pytest.mark.live
def test_the_window_matches_the_candle_the_model_is_actually_asked_to_predict():
    """The end-to-end guard against the off-by-one, checked against the real forecaster.

    Nothing else in the suite can catch this: `markets.py` and `forecaster.py` are each internally
    consistent, and the bug lived in the disagreement between them. So this asserts the one thing
    that matters — the first timestamp the model is asked for is the first timestamp the ledger
    claims the forecast is about.
    """
    import pandas as pd

    from core.market_data import MarketData
    from core.markets import CONTEXT_CANDLES

    candles = MarketData().candles("BTC/USDT:USDT", "1h", limit=CONTEXT_CANDLES)

    # Rebuilt exactly as core/forecaster.py builds it for the model.
    step_ms = int(candles.rows[-1][0]) - int(candles.rows[-2][0])
    first_predicted_ts = int(candles.rows[-1][0]) + step_ms
    last_predicted_ts = int(candles.rows[-1][0]) + step_ms * HORIZONS["24h"]

    window = window_for_anchor(int(candles.rows[-1][0]), "24h")
    assert ms(window["open"]) == first_predicted_ts, (
        "the window opens on a different candle than the model's first prediction")
    # The window closes at the END of the last predicted candle, one step past its opening stamp.
    assert ms(window["close"]) == last_predicted_ts + step_ms

    for horizon, steps in HORIZONS.items():
        w = window_for_anchor(int(candles.rows[-1][0]), horizon)
        assert ms(w["open"]) == first_predicted_ts
        assert ms(w["close"]) == int(candles.rows[-1][0]) + step_ms * (1 + steps)
