"""An input we do not serve is the caller's problem, not an outage.

`MarketDataError` covers two different situations and every one of them was answered with **502**:
the exchange is unreachable, and the caller asked for something that does not exist. Measured as a
customer against the live service — a symbol of `FAKE/USDT:USDT` and a timeframe of `9y` both came
back 502, which tells the caller our upstream broke and to try again. Neither will ever succeed, and
one of them blamed the venue for a typo.

502 means "retry, this may be temporary". 4xx means "change what you sent". Sending the first when
the second is true costs the caller their money on every retry.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import server
from core.market_data import MarketDataError

CALLER = ["bad_timeframe", "bad_limit", "unknown_exchange", "unsupported", "window_open",
          "unknown_symbol"]
UPSTREAM = ["source_error", "no_data", "empty_book", "gapped_series", "dependency_missing"]


@pytest.fixture()
def client(monkeypatch):
    """Payment is not what this file is testing, so it is stubbed as already settled. The handler is
    what decides the status code, and it is only reached once a payment has been accepted."""
    class _Paid:
        ok = True
        malformed = False
        detail = "stubbed"
        code = ""
        field = ""
        response_header = ""
        tx = ""
    monkeypatch.setattr(server, "verify_payment", lambda *a, **k: _Paid())
    monkeypatch.setattr(server, "malformed_payment", lambda *a, **k: None)
    return TestClient(server.app)


def test_the_two_kinds_of_failure_are_kept_apart():
    """If these sets ever overlap, one of the two answers is wrong for some code."""
    assert not (set(CALLER) & set(UPSTREAM))
    assert set(CALLER) == set(server._CALLER_INPUT_CODES)


@pytest.mark.parametrize("code", CALLER)
def test_a_caller_input_failure_is_a_4xx(client, monkeypatch, code):
    def boom(*a, **k):
        raise MarketDataError("nope", symbol="FAKE/USDT:USDT", code=code)
    monkeypatch.setattr(server.REGISTRY.get("market.candles"), "handler", boom, raising=False)

    r = client.post("/a2mcp/market.candles", json={"symbol": "FAKE/USDT:USDT", "timeframe": "1h"},
                    headers={"PAYMENT-SIGNATURE": _dev_header()})
    assert r.status_code == 422, f"{code} should tell the caller to change their input"
    body = r.json()
    assert body["code"] == code
    assert "retrying will not change it" in body["note"].lower()
    assert body.get("example_request"), "a caller error must show what a valid call looks like"


@pytest.mark.parametrize("code", UPSTREAM)
def test_a_genuine_upstream_failure_is_still_a_502(client, monkeypatch, code):
    def boom(*a, **k):
        raise MarketDataError("venue down", symbol="BTC/USDT:USDT", code=code)
    monkeypatch.setattr(server.REGISTRY.get("market.candles"), "handler", boom, raising=False)

    r = client.post("/a2mcp/market.candles", json={"symbol": "BTC/USDT:USDT", "timeframe": "1h"},
                    headers={"PAYMENT-SIGNATURE": _dev_header()})
    assert r.status_code == 502, f"{code} is ours, not the caller's"
    assert "worth retrying" in r.json()["note"].lower()


def test_ccxt_bad_symbol_is_classified_as_a_caller_error():
    """The class-name match is what routes a typo away from 502; it must cover the real names."""
    from core.market_data import _CALLER_SYMBOL_ERRORS
    assert "BadSymbol" in _CALLER_SYMBOL_ERRORS
    assert "NetworkError" not in _CALLER_SYMBOL_ERRORS
    assert "ExchangeNotAvailable" not in _CALLER_SYMBOL_ERRORS


def _dev_header() -> str:
    import base64
    import json as _json
    return base64.b64encode(_json.dumps({
        "x402Version": 2, "accepted": {"network": "eip155:196"},
        "payload": {"authorization": {"value": "1", "nonce": "0x" + "22" * 32}, "signature": "0x00"},
    }).encode()).decode()
