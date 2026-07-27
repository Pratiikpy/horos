"""The empty-record path, which a real buyer paid for and complained about.

Agent MantaRay bought "check BTC track record" on 2026-07-27 and left two stars: *"Paid but got
endpoint docs, not actual BTC track record data"*. They were right. At that moment ETH had scored
forecasts and BTC did not, and every service that filtered to an empty set returned the same block
of prose — a coverage policy, a how-to-verify paragraph, and the sentence "No forecast has been
scored yet", which was false for a caller who had asked about BTC specifically.

Two separate defects, so two kinds of test here. The first is truthfulness: an empty *filtered*
result must never claim the record is globally empty. The second is worth: a paid call must return
the committed evidence that does exist for that filter — the hashed, signed, on-chain-anchored
forecasts and when each grades — rather than documentation about evidence it declined to show.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.crypto import Signer
from core.ledger import Ledger
from services.accountability import forecast_reliability, leaderboard, scorecard
from services.context import Context

ETH = "ETH/USDT:USDT"
BTC = "BTC/USDT:USDT"


def _window(open_h: int, close_h: int) -> dict:
    return {"open": f"2026-07-27T{open_h:02d}:00:00Z", "close": f"2026-07-27T{close_h:02d}:00:00Z"}


def _issue(ledger: Ledger, symbol: str, horizon: str, window: dict):
    return ledger.record_forecast(
        symbol=symbol, timeframe="1h", horizon=horizon, model="kronos-small",
        model_version="NeoQuasar/Kronos-small@test", issued_for=window["close"], window=window,
        distribution={"bands": {"0.9": [1.0, 2.0]}, "n_paths": 8, "symbol": symbol},
        inputs={"symbol": symbol, "anchor_close": 100.0})


@pytest.fixture()
def ctx(tmp_path: Path) -> Context:
    """A ledger where ETH is scored and BTC is committed but not yet graded.

    This is the exact asymmetry that produced the false sentence — a record that exists, but not for
    the symbol the caller asked about.
    """
    signer = Signer(tmp_path / "key.hex")
    ledger = Ledger(tmp_path / "ledger.jsonl", signer)

    eth = _issue(ledger, ETH, "1h", _window(10, 11))
    ledger.record_score(
        forecast_id=eth.forecast_id,
        actual={"symbol": ETH, "close": 1900.0, "window": _window(10, 11)},
        scores={"count": 1, "crps": 12.0, "mean_crps": 12.0, "crps_skill_vs_empirical": 0.1},
        scorer_version="1.0.0")

    # Two BTC forecasts, committed and hashed, neither graded.
    _issue(ledger, BTC, "1h", _window(20, 21))
    _issue(ledger, BTC, "4h", _window(20, 24))
    ledger.record_anchor(head=ledger.head(), tx="0xdeadbeef", chain="eip155:196")

    return Context(data=None, runner=None, forecaster=None, ledger=ledger, signer=signer)


def _text(payload: dict) -> str:
    return json.dumps(payload, default=str).lower()


# ── defect A: the response must not lie about the rest of the record ─────────────────────────────

@pytest.mark.parametrize("call", [
    lambda c: scorecard({"symbol": BTC}, c),
    lambda c: forecast_reliability({"symbol": BTC}, c),
    lambda c: leaderboard({"symbol": BTC}, c),
])
def test_filtered_empty_never_claims_the_whole_record_is_empty(ctx, call):
    """The precise sentence the buyer was shown. ETH is scored, so it was untrue."""
    body = _text(call(ctx))
    assert "no forecast has been scored yet" not in body
    assert "nobody has a scored forecast in this ledger yet" not in body


def test_filtered_empty_says_which_symbols_do_have_a_record(ctx):
    out = scorecard({"symbol": BTC}, ctx)
    assert out["scored_forecasts"] == 0
    assert out["record_available_for"]["symbols"] == [ETH]
    assert BTC in out["note"] or "symbol=" in out["note"]


def test_globally_empty_still_says_so(tmp_path: Path):
    """The original sentence is correct when nothing anywhere has been scored — keep it then."""
    signer = Signer(tmp_path / "key.hex")
    ledger = Ledger(tmp_path / "l.jsonl", signer)
    _issue(ledger, BTC, "1h", _window(20, 21))
    empty = Context(data=None, runner=None, forecaster=None, ledger=ledger, signer=signer)

    out = scorecard({"symbol": BTC}, empty)
    assert "no forecast has been scored yet, for any symbol" in out["note"].lower()
    assert "record_available_for" not in out


# ── defect B: a paid empty response must still carry evidence ────────────────────────────────────

def test_empty_scorecard_returns_the_committed_forecasts_for_that_symbol(ctx):
    """What the buyer should have received: the BTC commitments, not a coverage policy."""
    out = scorecard({"symbol": BTC}, ctx)

    assert out["pending_count"] == 2
    assert {p["symbol"] for p in out["pending_forecasts"]} == {BTC}
    assert out["next_grades_at"] == "2026-07-27T21:00:00Z"          # soonest first

    for row in out["pending_forecasts"]:
        assert row["distribution_sha256"].startswith("sha256:")     # checkable now
        assert row["grades_after"]                                  # gradeable later
        assert row["anchor"]["tx"] == "0xdeadbeef"                  # on chain already
        assert row["anchor"]["chain"] == "eip155:196"


def test_horizon_filter_narrows_the_pending_set(ctx):
    out = scorecard({"symbol": BTC, "horizon": "4h"}, ctx)
    assert out["pending_count"] == 1
    assert out["pending_forecasts"][0]["horizon"] == "4h"
    assert out["filter"] == {"symbol": BTC, "horizon": "4h"}


def test_reliability_reports_pending_instead_of_only_refusing(ctx):
    out = forecast_reliability({"symbol": BTC}, ctx)
    assert out["verdict"] == "no_record_yet"
    assert out["pending_count"] == 2
    assert ETH.lower() in _text(out)            # names where a record does exist
    assert out["record_available_for"]["symbols"] == [ETH]
    assert out["next_grades_at"] == "2026-07-27T21:00:00Z"


def test_symbol_outside_coverage_says_so_rather_than_showing_an_empty_list(ctx):
    out = scorecard({"symbol": "DOGE/USDT:USDT"}, ctx)
    assert out["pending_count"] == 0
    assert "pending_note" in out
    assert "never forecast" in out["pending_note"]


def test_voided_forecasts_are_not_offered_as_pending(ctx):
    """A withdrawn forecast is not evidence, and must not pad the count."""
    before = scorecard({"symbol": BTC}, ctx)["pending_count"]
    victim = next(p for p in ctx.pending(BTC))
    ctx.ledger.record_void(forecast_id=victim["forecast_id"], reason="test")
    ctx.invalidate()

    after = scorecard({"symbol": BTC}, ctx)
    assert after["pending_count"] == before - 1
    assert victim["forecast_id"] not in _text(after)
