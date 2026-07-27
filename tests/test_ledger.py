"""The ledger's claim is that history cannot be edited without it showing.

So these tests do the editing. Each one takes a valid ledger, performs the specific tampering a
dishonest operator would actually attempt — drop the bad forecast, soften it, reorder two, forge a
signature — and asserts that `verify()` names it. A test that only checked the happy path would tell
us nothing about the property the product is sold on.
"""
from __future__ import annotations

import json

import pytest

from core.crypto import Signer, canonical, digest
from core.ledger import GENESIS, Entry, Ledger


@pytest.fixture()
def ledger(tmp_path) -> Ledger:
    return Ledger(tmp_path / "ledger.jsonl", Signer(tmp_path / "key.hex"))


def _forecast(led: Ledger, symbol: str = "BTC/USDT:USDT", horizon: str = "1h",
              close: str = "2026-07-27T13:00:00Z", anchor: float = 100.0):
    return led.record_forecast(
        symbol=symbol, timeframe="1h", horizon=horizon, model="kronos-small",
        model_version="test", issued_for=close,
        window={"open": "2026-07-27T12:00:00Z", "close": close, "timeframe": "1h", "candles": 1},
        distribution={"anchor_price": anchor, "samples": [99.0, 100.0, 101.0]},
        inputs={"candles_sha256": "abc", "context": 512})


def _rewrite(led: Ledger, rows: list[dict]) -> None:
    led.path.write_text("".join(canonical(r) + "\n" for r in rows), encoding="utf-8")


def _rows(led: Ledger) -> list[dict]:
    return [json.loads(l) for l in led.path.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── the happy path ───────────────────────────────────────────────────────────────────────────────

def test_an_empty_ledger_has_the_genesis_head():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        led = Ledger(Path(d) / "l.jsonl", Signer(Path(d) / "k"))
        assert led.head() == GENESIS
        assert led.count() == 0
        assert led.verify() == (True, [])


def test_entries_chain_to_each_other(ledger):
    a = _forecast(ledger, close="2026-07-27T13:00:00Z")
    b = _forecast(ledger, close="2026-07-27T14:00:00Z")
    assert a.prev == GENESIS
    assert b.prev == a.hash
    assert ledger.head() == b.hash
    assert ledger.count() == 2


def test_a_clean_ledger_verifies(ledger):
    for i in range(5):
        _forecast(ledger, close=f"2026-07-27T{13 + i}:00:00Z")
    ledger.record_score(forecast_id="x", actual={"close": 101.0}, scores={"crps": 0.4},
                        scorer_version="1.0.0")
    ok, problems = ledger.verify()
    assert ok, problems


def test_forecast_ids_are_deterministic_and_distinct(ledger):
    a = _forecast(ledger, symbol="BTC/USDT:USDT", close="2026-07-27T13:00:00Z")
    b = _forecast(ledger, symbol="ETH/USDT:USDT", close="2026-07-27T13:00:00Z")
    assert a.forecast_id != b.forecast_id
    assert len(a.forecast_id) == 32


def test_the_signature_covers_the_manifest_and_verifies_standalone(ledger):
    e = _forecast(ledger)
    sig = e.body["signature"]
    assert Signer.verify(sig["manifest_sha256"], sig["signature"], sig["public_key"])
    # And the manifest can be rebuilt from the published body alone, which is what makes the
    # signature checkable by someone who only has the response.
    manifest = {k: e.body[k] for k in ("symbol", "timeframe", "horizon", "model", "model_version",
                                       "issued_for", "window")}
    manifest["distribution_sha256"] = e.body["distribution_sha256"]
    manifest["inputs_sha256"] = e.body["inputs_sha256"]
    assert digest(manifest) == sig["manifest_sha256"]


# ── tampering: each of these is an attack, and each must be caught ───────────────────────────────

def test_deleting_a_losing_forecast_breaks_the_chain(ledger):
    """The single most likely dishonesty: quietly drop the one that went badly."""
    for i in range(4):
        _forecast(ledger, close=f"2026-07-27T{13 + i}:00:00Z")
    rows = _rows(ledger)
    del rows[1]
    _rewrite(ledger, rows)
    ok, problems = ledger.verify()
    assert not ok
    assert any("does not chain" in p for p in problems)


def test_softening_a_forecast_after_the_fact_is_caught(ledger):
    """Edit the recorded distribution to something the outcome fell inside."""
    _forecast(ledger, anchor=100.0)
    rows = _rows(ledger)
    rows[0]["body"]["distribution"]["samples"] = [140.0, 150.0, 160.0]
    _rewrite(ledger, rows)
    ok, problems = ledger.verify()
    assert not ok
    assert any("hash does not match" in p for p in problems)


def test_editing_a_field_and_recomputing_the_hash_still_fails_on_the_signature(ledger):
    """A more careful forger fixes the entry hash. The Ed25519 signature still refuses.

    This is why the chain and the signature are separate mechanisms: the chain is ours to recompute,
    the signature needs a key we would have to leak to make the forgery work.
    """
    _forecast(ledger)
    rows = _rows(ledger)
    rows[0]["body"]["horizon"] = "24h"
    rows[0]["hash"] = digest({k: rows[0][k] for k in ("seq", "kind", "at", "body", "prev")})
    _rewrite(ledger, rows)
    ok, problems = ledger.verify()
    assert not ok
    assert any("signature does not verify" in p for p in problems)


def test_an_entry_signed_with_a_different_key_is_caught(tmp_path, ledger):
    """A forger who edits a field *and* re-signs it with a key they control.

    Every individual check passes: the manifest rebuilds correctly, the signature is valid over it,
    the entry hash matches. Only pinning the expected public key refuses it. Without that, anyone
    could mint entries that verify.
    """
    _forecast(ledger)
    impostor = Signer(tmp_path / "impostor.hex")
    rows = _rows(ledger)
    rows[0]["body"]["horizon"] = "24h"
    from core.ledger import manifest_for
    rows[0]["body"]["signature"] = impostor.sign(manifest_for("issued", rows[0]["body"]))
    rows[0]["hash"] = digest({k: rows[0][k] for k in ("seq", "kind", "at", "body", "prev")})
    _rewrite(ledger, rows)
    ok, problems = ledger.verify()
    assert not ok
    assert any("expected" in p for p in problems)


def test_stripping_the_signature_off_an_entry_is_caught(ledger):
    """Removing the signature must not be a way to make an unsigned entry acceptable."""
    _forecast(ledger)
    rows = _rows(ledger)
    rows[0]["body"].pop("signature")
    rows[0]["hash"] = digest({k: rows[0][k] for k in ("seq", "kind", "at", "body", "prev")})
    _rewrite(ledger, rows)
    ok, problems = ledger.verify()
    assert not ok
    assert any("must carry a signature" in p for p in problems)


def test_reordering_two_entries_is_caught(ledger):
    a = _forecast(ledger, close="2026-07-27T13:00:00Z")
    b = _forecast(ledger, close="2026-07-27T14:00:00Z")
    rows = _rows(ledger)
    rows[0], rows[1] = rows[1], rows[0]
    _rewrite(ledger, rows)
    ok, problems = ledger.verify()
    assert not ok
    assert problems


def test_appending_a_forged_entry_to_a_real_chain_is_caught(ledger):
    """The forger has the real chain and adds one flattering row at the end."""
    _forecast(ledger)
    rows = _rows(ledger)
    forged = dict(rows[-1])
    forged["seq"] = 1
    forged["body"] = dict(forged["body"], forecast_id="forged", horizon="24h")
    forged["prev"] = rows[-1]["hash"]
    forged["hash"] = digest({k: forged[k] for k in ("seq", "kind", "at", "body", "prev")})
    _rewrite(ledger, rows + [forged])
    ok, problems = ledger.verify()
    assert not ok
    assert any("signature" in p for p in problems)


def test_rebuilding_the_whole_chain_verifies_which_is_why_anchoring_exists(ledger):
    """The honest limit of the hash chain, pinned as a test so it is never overstated.

    A hash chain rebuilt from scratch is internally consistent — it verifies. Only the on-chain
    anchor rules this out, because a rebuilt chain cannot reproduce a head that is already in a
    block. The scorecard therefore never calls an unanchored record "proven".
    """
    _forecast(ledger, close="2026-07-27T13:00:00Z")
    fresh = Ledger(ledger.path.parent / "rebuilt.jsonl", ledger.signer)
    _forecast(fresh, close="2026-07-27T13:00:00Z", anchor=999.0)
    ok, problems = fresh.verify()
    assert ok and not problems


# ── pending / queue semantics ────────────────────────────────────────────────────────────────────

def test_pending_returns_only_closed_unscored_windows(ledger):
    open_still = _forecast(ledger, close="2099-01-01T00:00:00Z")
    closed = _forecast(ledger, close="2020-01-01T00:00:00Z")
    pending = ledger.pending(now="2026-07-27T12:00:00Z")
    ids = {e.forecast_id for e in pending}
    assert closed.forecast_id in ids
    assert open_still.forecast_id not in ids


def test_a_scored_forecast_leaves_the_pending_queue(ledger):
    e = _forecast(ledger, close="2020-01-01T00:00:00Z")
    assert ledger.pending(now="2026-07-27T12:00:00Z")
    ledger.record_score(forecast_id=e.forecast_id, actual={"close": 1.0}, scores={"crps": 0.1},
                        scorer_version="1.0.0")
    assert not ledger.pending(now="2026-07-27T12:00:00Z")


def test_the_ledger_is_the_queue_so_a_restart_loses_nothing(ledger):
    """Reopening the file recovers the same work list, with no side state anywhere."""
    _forecast(ledger, close="2020-01-01T00:00:00Z")
    reopened = Ledger(ledger.path, ledger.signer)
    assert len(reopened.pending(now="2026-07-27T12:00:00Z")) == 1


# ── canonicalisation ─────────────────────────────────────────────────────────────────────────────

def test_canonical_form_is_stable_under_key_order():
    assert canonical({"b": 1, "a": 2}) == canonical({"a": 2, "b": 1})
    assert digest({"b": 1, "a": 2}) == digest({"a": 2, "b": 1})


def test_canonical_form_emits_unicode_as_unicode_not_escapes():
    """The asset name is USD₮0. If canonicalisation escaped it, our bytes and a buyer's would differ."""
    assert "₮" in canonical({"asset": "USD₮0"})
    assert "\\u" not in canonical({"asset": "USD₮0"})


def test_canonical_form_refuses_nan_rather_than_emitting_invalid_json():
    """`NaN` is not JSON. Emitting it would produce a receipt no standard parser can read back."""
    with pytest.raises(ValueError):
        canonical({"x": float("nan")})


def test_every_line_of_the_ledger_is_valid_json(ledger):
    _forecast(ledger)
    ledger.record_anchor(head=ledger.head(), tx="0xabc", chain="eip155:196")
    for line in ledger.path.read_text(encoding="utf-8").splitlines():
        assert isinstance(json.loads(line), dict)


def test_entries_survive_a_round_trip_through_the_file(ledger):
    written = _forecast(ledger)
    read_back = next(iter(ledger))
    assert isinstance(read_back, Entry)
    assert read_back.hash == written.hash
    assert read_back.body == written.body


# ── the recorder's coverage check ────────────────────────────────────────────────────────────────

def test_a_voided_forecast_does_not_block_a_replacement_for_the_same_window(ledger, tmp_path):
    """Caught in production: the recorder stood still while reporting 'already_issued'.

    Twelve forecasts were voided after a window-alignment fix. `already_covers` counted them as
    coverage, so every replacement round found the window taken and issued nothing — the record
    stopped growing while the logs looked healthy. A withdrawn forecast is precisely one that cannot
    be scored, so it can never count as covering its window.
    """
    from core.market_data import MarketData
    from core.recorder import Recorder

    rec = Recorder(ledger, MarketData())
    e = _forecast(ledger, close="2026-07-27T13:00:00Z")
    assert rec.already_covers("BTC/USDT:USDT", "1h", "2026-07-27T13:00:00Z")

    ledger.record_void(forecast_id=e.forecast_id, reason="window was misaligned")
    assert not rec.already_covers("BTC/USDT:USDT", "1h", "2026-07-27T13:00:00Z"), (
        "a voided forecast must not block its window from being forecast again")
