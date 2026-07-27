"""A request field we did not understand must be reported, not dropped in silence.

`market.venues` declares `exchanges`. A caller who typed `exchagnes` had the key discarded, the
declared default substituted, and a full-priced answer returned with nothing indicating any of it had
happened: they asked for two venues and received five. Measured against the live service.

The same slip on `horizon`, `max_age_seconds` or `limit` returns a confident answer to a question the
caller did not ask, which is the kind they act on.

This warns rather than refuses, deliberately. Refusing is cheaper for the caller and would also
reject every client that sends an extra field today; breaking a working integration to prevent a typo
is the wrong trade on a listed service. The note rides inside the signed envelope so it cannot be
separated from the answer it qualifies.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import server


@pytest.fixture()
def client(monkeypatch):
    class _Paid:
        ok = True; malformed = False; detail = "stubbed"; code = ""; field = ""
        response_header = ""; tx = ""
    monkeypatch.setattr(server, "verify_payment", lambda *a, **k: _Paid())
    monkeypatch.setattr(server, "malformed_payment", lambda *a, **k: None)
    return TestClient(server.app)


HEADER = {"PAYMENT-SIGNATURE": "stub"}


def _notes(body) -> str:
    return " ".join(body.get("notes") or [])


def test_a_typo_is_named_with_a_suggestion(client):
    r = client.post("/a2mcp/market.venues",
                    json={"symbol": "BTC/USDT:USDT", "exchagnes": ["okx"]}, headers=HEADER)
    assert r.status_code == 200
    notes = _notes(r.json())
    assert "exchagnes" in notes
    assert "did you mean 'exchanges'" in notes
    assert "may not be the one you intended" in notes


def test_it_lists_what_the_endpoint_does_accept(client):
    r = client.post("/a2mcp/market.venues",
                    json={"symbol": "BTC/USDT:USDT", "totally_made_up": 1}, headers=HEADER)
    notes = _notes(r.json())
    assert "totally_made_up" in notes
    assert "Accepted fields:" in notes and "exchanges" in notes


def test_a_correct_request_is_not_warned_about(client):
    r = client.post("/a2mcp/market.venues",
                    json={"symbol": "BTC/USDT:USDT", "exchanges": ["okx"]}, headers=HEADER)
    assert "Ignored" not in _notes(r.json())


@pytest.mark.parametrize("wrapper", ["options", "idempotency_key"])
def test_the_reserved_wrapper_fields_are_not_flagged(client, wrapper):
    """These are stripped before the handler by design and are not a caller mistake."""
    r = client.post("/a2mcp/market.venues",
                    json={"symbol": "BTC/USDT:USDT", wrapper: {} if wrapper == "options" else "k1"},
                    headers=HEADER)
    assert wrapper not in _notes(r.json())


def test_the_warning_is_inside_the_signed_envelope(client):
    """A warning that can be stripped from the answer it qualifies is not a warning."""
    body = client.post("/a2mcp/market.venues",
                       json={"symbol": "BTC/USDT:USDT", "exchagnes": ["okx"]}, headers=HEADER).json()
    assert body.get("notes"), "notes must be present"
    assert body.get("receipt"), "and covered by the receipt the buyer verifies"
