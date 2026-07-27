"""Incomplete input must be answered before the money moves.

Measured against the live service before this was fixed: a call to `commit` carrying a `digest` but
no `window_close` or `label` settled the full $0.01 and returned 422. The caller ended up with a fee
and an error message — the same complaint a real buyer already left against this agent, in a
different disguise.

The status codes are asserted here on purpose. They are unchanged from the behaviour that passed OKX
review — 200 with the contract for an empty body, 422 for a partial one — because the fix is meant to
be invisible to every client except in the one respect that they stop being billed. A future change
that "tidies" these into a single 400 would be a listing risk, not a cleanup.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import server
import x402_bridge

PAID = "/a2mcp/commit"          # requires digest, window_close and label


@pytest.fixture()
def client():
    return TestClient(server.app)


@pytest.fixture()
def never_settles(monkeypatch):
    """Any settlement attempt is a failure, whatever it would have returned."""
    def boom(*a, **k):                                               # noqa: ANN002, ANN003
        raise AssertionError("payment was settled for a request that could not be served")
    monkeypatch.setattr(server, "verify_payment", boom)


def test_partial_input_is_refused_before_settlement(client, never_settles):
    r = client.post(PAID, json={"digest": "0x" + "ab" * 32},
                    headers={"PAYMENT-SIGNATURE": _plausible_header()})
    assert r.status_code == 422
    body = r.json()
    assert body["code"] == "missing_required_field"
    assert "window_close" in body["error"] and "label" in body["error"]
    assert "not_charged" in body
    assert body["example_request"]["input"], "a refusal must show what a valid call looks like"


def test_empty_input_is_refused_before_settlement(client, never_settles):
    r = client.post(PAID, json={}, headers={"PAYMENT-SIGNATURE": _plausible_header()})
    assert r.status_code == 200                    # unchanged from the reviewed behaviour
    body = r.json()
    assert body["status"] == "input_required"
    assert "not_charged" in body
    assert "result" not in body, "a refusal must not be shaped like a result"


def test_a_broken_header_still_wins_over_a_thin_body(client):
    """Two bugs at once: the header error is the one that explains why retrying will not help."""
    r = client.post(PAID, json={}, headers={"PAYMENT-SIGNATURE": "!!!not-base64!!!"})
    assert r.status_code == 400
    assert r.json()["code"] == "malformed_payment_signature"


def test_the_unpaid_probe_is_untouched(client):
    """OKX's availability check sends no payment header and no body, and reads the status."""
    r = client.post(PAID, json={})
    assert r.status_code == 402
    assert r.headers.get(x402_bridge.PAYMENT_REQUIRED_HEADER)


@pytest.mark.parametrize("header", ["!!!not-base64!!!", "aGVsbG8=", "WzEsMiwzXQ==", "eyJub3BlIjoxfQ=="])
def test_malformed_headers_are_typed_400s_not_dropped_sockets(client, header):
    r = client.post(PAID, json={"digest": "0x" + "cd" * 32}, headers={"PAYMENT-SIGNATURE": header})
    assert r.status_code == 400
    assert r.json()["code"] == "malformed_payment_signature"


def _plausible_header() -> str:
    """A header that parses as an x402 v2 payload, so the malformed check passes and the input
    check is what decides the response."""
    import base64
    import json
    return base64.b64encode(json.dumps({
        "x402Version": 2,
        "accepted": {"network": "eip155:196"},
        "payload": {"authorization": {"value": "1", "nonce": "0x" + "11" * 32}, "signature": "0x00"},
    }).encode()).decode()
