"""The listing gates, as tests.

Every assertion here corresponds to a documented reason a real agent was rejected in live OKX review
(`OKX-REVIEW-RULES.md`). They are cheap to run and they are the difference between finding out now
and finding out from a reviewer.
"""
from __future__ import annotations

import base64
import json

import pytest
from fastapi.testclient import TestClient

import server
from services.registry import MAX_DESCRIPTION, REGISTRY
from x402_bridge import PAYMENT_REQUIRED_HEADER, decode_challenge, make_dev_payment


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(server.app)


PAID = "/a2mcp/forecast.path"
FREE = "/a2mcp/scorecard"


# ── §1.2 the endpoint must return 402, never 200/400/405 ─────────────────────────────────────────

@pytest.mark.parametrize("method", ["GET", "POST", "PUT", "PATCH", "HEAD", "OPTIONS", "DELETE"])
def test_every_method_with_no_body_returns_402_with_a_challenge(client, method):
    """The validator probes with no body and reads the status code.

    Four agents were rejected for answering this with 405 or 400. A handler that validates input
    before resolving payment never gets to show a challenge at all.
    """
    r = client.request(method, PAID)
    assert r.status_code == 402
    assert PAYMENT_REQUIRED_HEADER in r.headers


def test_an_empty_json_body_still_gets_a_challenge_not_a_validation_error(client):
    r = client.post(PAID, json={})
    assert r.status_code == 402


def test_no_service_is_listed_at_a_zero_fee(client):
    """Two OKX sources disagree, and this is the resolution.

    `OKX-REVIEW-RULES.md` §1.2 says a free service should still gate with a zero-amount challenge,
    citing TRACE #7515 as precedent. The builder chat (25 Jul 2026) says the opposite and more
    specifically: a `$0` x402 "exact" gate confuses `task-402-pay` into an amount mismatch, so a
    zero-fee service should either return 200 directly or carry a small non-zero fee.

    A zero-amount challenge therefore advertises a purchase path that does not complete — worse than
    either alternative. Horos prices every listed service at a tenth of a cent or more, and keeps the
    record genuinely free where it actually matters: the unauthenticated web pages.
    """
    for svc in REGISTRY.all():
        assert svc.price_usdt > 0, f"{svc.endpoint} is listed at zero"
    r = client.post(FREE, json={})
    assert r.status_code == 402
    assert decode_challenge(r.headers[PAYMENT_REQUIRED_HEADER])["accepts"][0]["amount"] != "0"


def test_the_public_record_needs_no_payment_at_all(client):
    """The free promise, kept where it counts: no payment, no account, no x402."""
    for path in ("/scorecard", "/ledger", "/ledger/verify", "/services", "/verify"):
        assert client.get(path).status_code == 200, path


def test_every_listed_name_fits_the_marketplace_limit():
    """`serviceName` caps at 30 characters — undocumented until registration rejected it."""
    for svc in REGISTRY.all():
        name = svc.short or svc.title
        assert len(name) <= 30, f"{svc.endpoint}: {name!r} is {len(name)} chars"


def test_the_challenge_carries_every_constant_the_validator_checks(client):
    r = client.post(PAID, json={})
    ch = decode_challenge(r.headers[PAYMENT_REQUIRED_HEADER])
    assert ch["x402Version"] == 2
    a = ch["accepts"][0]
    assert a["scheme"] == "exact"
    assert a["network"] == "eip155:196"
    assert a["asset"].lower() == "0x779ded0c9e1022225f8e0630b35a9b54be713736"
    assert a["maxTimeoutSeconds"] == 300
    # decimals MUST live inside `extra`: USD₮0 is not in OKX's token list and a top-level `decimals`
    # is dropped by their canonical re-serialisation, misreading the amount by 10^6.
    assert a["extra"]["decimals"] == 6
    assert a["extra"]["name"] == "USD₮0"


def test_the_challenge_amount_matches_the_listed_price_for_every_service(client):
    """A challenge amount that disagrees with the listing is a rejection."""
    for svc in REGISTRY.all():
        r = client.post(f"/a2mcp/{svc.endpoint}", json={})
        assert r.status_code == 402, svc.endpoint
        a = decode_challenge(r.headers[PAYMENT_REQUIRED_HEADER])["accepts"][0]
        assert a["amount"] == str(int(round(svc.price_usdt * 10 ** 6))), svc.endpoint


def test_the_challenge_advertises_the_input_contract(client):
    """A buying agent reads the challenge to learn how to call the service.

    Without it, it sends an empty body, pays, and gets an error — which is exactly what happened on
    the first real A2A purchase of a sibling ASP.
    """
    ch = decode_challenge(client.post(PAID, json={}).headers[PAYMENT_REQUIRED_HEADER])
    schema = ch["resource"]["inputSchema"]
    assert "symbol" in schema["properties"]
    assert schema["example"]


# ── §1.10 a malformed header must return a typed 400, never a dropped socket ─────────────────────

@pytest.mark.parametrize("value,label", [
    ("!!!not-base64!!!", "garbage"),
    (base64.b64encode(b"hello").decode(), "valid base64, not JSON"),
    (base64.b64encode(b"[1,2,3]").decode(), "JSON array, not an object"),
    (base64.b64encode(b'{"nope":1}').decode(), "JSON object with no payment fields"),
    ("", "present but empty"),
])
def test_a_malformed_payment_header_returns_a_typed_error_not_a_dropped_socket(client, value, label):
    """Our own rejection, verbatim: 'a malformed PAYMENT-SIGNATURE header kills the connection
    instead of returning an error. I sent three variants and every one gave empty reply from
    server.' Every variant must produce an HTTP response with a code a caller can act on."""
    r = client.post(PAID, json={}, headers={"PAYMENT-SIGNATURE": value})
    assert r.status_code in (400, 402), label
    body = r.json()
    assert body.get("code"), label
    assert body.get("error"), label


def test_an_absent_header_and_a_blank_one_are_treated_differently(client):
    """No header means no payment was attempted — a challenge. A blank header is a client bug."""
    assert client.post(PAID, json={}).status_code == 402
    blank = client.post(PAID, json={}, headers={"PAYMENT-SIGNATURE": ""})
    assert blank.status_code == 400
    assert blank.json()["code"] == "malformed_payment_signature"


def test_both_payment_header_names_are_accepted(client):
    """The OKX agentic wallet sends PAYMENT-SIGNATURE; X-PAYMENT is the older name."""
    for header in ("PAYMENT-SIGNATURE", "X-PAYMENT"):
        ch = decode_challenge(client.post(PAID, json={}).headers[PAYMENT_REQUIRED_HEADER])
        r = client.post(PAID, json={"input": {"symbol": "BTC/USDT:USDT", "horizon": "1h"}},
                        headers={header: make_dev_payment(ch)})
        assert r.status_code == 200, header


def test_there_is_no_second_auth_layer_on_top_of_x402(client):
    """An agent was rejected for demanding a Bearer apiKey as well; the standard client cannot send
    one. A signed payment alone must unlock the service."""
    ch = decode_challenge(client.post(PAID, json={}).headers[PAYMENT_REQUIRED_HEADER])
    r = client.post(PAID, json={"input": {"symbol": "BTC/USDT:USDT", "horizon": "1h"}},
                    headers={"PAYMENT-SIGNATURE": make_dev_payment(ch)})
    assert r.status_code == 200


def test_a_replayed_payment_is_refused(client):
    ch = decode_challenge(client.post(PAID, json={}).headers[PAYMENT_REQUIRED_HEADER])
    pay = make_dev_payment(ch)
    body = {"input": {"symbol": "BTC/USDT:USDT", "horizon": "1h"}}
    assert client.post(PAID, json=body, headers={"PAYMENT-SIGNATURE": pay}).status_code == 200
    again = client.post(PAID, json=body, headers={"PAYMENT-SIGNATURE": pay})
    assert again.status_code == 402


# ── after payment ────────────────────────────────────────────────────────────────────────────────

def _paid(client, endpoint: str, payload: dict):
    url = f"/a2mcp/{endpoint}"
    ch = decode_challenge(client.post(url, json={}).headers[PAYMENT_REQUIRED_HEADER])
    return client.post(url, json={"input": payload},
                       headers={"PAYMENT-SIGNATURE": make_dev_payment(ch)})


def test_a_paid_call_with_an_empty_body_returns_the_contract_at_200_not_an_error(client):
    """Payment settles before the handler runs, so answering 'you forgot the symbol' and keeping
    the money is charging for nothing."""
    r = _paid(client, "forecast.path", {})
    assert r.status_code == 200
    assert r.json()["status"] == "input_required"
    assert r.json()["example_request"]["input"]


def test_a_paid_call_missing_one_field_names_it(client):
    r = _paid(client, "risk.size", {"symbol": "BTC/USDT:USDT"})
    assert r.status_code == 422
    assert "account_equity" in r.json()["error"]


def test_a_bad_argument_returns_a_typed_code_never_a_bare_500(client):
    """§1.6: an agent was rejected for a reproducible HTTP 500 on valid input."""
    r = _paid(client, "forecast.path", {"symbol": "BTC/USDT:USDT", "horizon": "3h"})
    assert r.status_code != 500
    assert r.json()["code"] == "bad_horizon"
    assert "1h" in r.json()["error"]


def test_an_unknown_service_is_a_404_that_lists_the_real_ones(client):
    r = client.post("/a2mcp/does.not.exist", json={})
    assert r.status_code == 404
    assert "forecast.path" in r.json()["services"]


def test_the_response_is_signed_and_verifies_from_its_own_fields(client):
    """The receipt must be checkable from the response alone, with nothing from us but the key."""
    import hashlib

    from nacl.signing import VerifyKey

    env = _paid(client, "forecast.path", {"symbol": "BTC/USDT:USDT", "horizon": "1h"}).json()
    manifest = {"endpoint": env["endpoint"], "input_sha256": env["input_sha256"],
                "output_sha256": env["output_sha256"], "tool": "horos",
                "price_usdt": env["price_usdt"], "job_id": env["job_id"]}
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = "sha256:" + hashlib.sha256(blob.encode()).hexdigest()
    assert digest == env["receipt"]["manifest_sha256"]
    VerifyKey(bytes.fromhex(env["receipt"]["public_key"])).verify(
        digest.encode(), bytes.fromhex(env["receipt"]["signature"]))


def test_the_published_verification_snippet_actually_runs(client):
    """Verification instructions that do not work are worse than none."""
    env = _paid(client, "scorecard", {}).json()
    snippet = client.get("/verify").json()["python"]
    exec(snippet, {"env": env})          # noqa: S102 — executing our own published instructions


# ── §1.4 listing quality ─────────────────────────────────────────────────────────────────────────

def test_every_service_has_a_complete_description_within_the_limit():
    """§1.4: 'missing a complete description, parameter details, and usage examples'."""
    for svc in REGISTRY.all():
        assert len(svc.description) <= MAX_DESCRIPTION, svc.endpoint
        assert "Example:" in svc.description, svc.endpoint
        assert len(svc.summary.split()) >= 8, svc.endpoint
        assert svc.returns.strip(), svc.endpoint
        assert svc.depth.strip(), svc.endpoint


def test_every_required_parameter_appears_in_its_own_example():
    for svc in REGISTRY.all():
        for name in svc.required:
            assert name in svc.example, f"{svc.endpoint} omits {name} from its example"


def test_the_catalogue_is_the_split_the_design_calls_for():
    a2a = [s for s in REGISTRY.all() if s.group.startswith("H.")]
    assert len(REGISTRY.all()) == 40
    assert len(a2a) == 1, "OKX's second gate is a live A2A probe; exactly one A2A service"


def test_the_route_table_is_keyed_without_a_method(client):
    """A method-qualified key makes the validator's probe miss the route entirely."""
    routes = client.get("/.well-known/x402").json()["routes"]
    assert len(routes) == len(REGISTRY.all())
    for key in routes:
        assert key.startswith("* /a2mcp/"), key


# ── the public record ────────────────────────────────────────────────────────────────────────────

def test_the_scorecard_renders_and_says_what_it_cannot_yet_show(client):
    r = client.get("/scorecard")
    assert r.status_code == 200
    assert "Horos" in r.text
    # Whatever the state, the page must never imply a backtest exists.
    assert "backtest" in r.text.lower()


def test_the_ledger_downloads_as_valid_jsonl(client):
    r = client.get("/ledger")
    assert r.status_code == 200
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    assert lines
    for ln in lines:
        assert isinstance(json.loads(ln), dict)


def test_the_chain_verifies_over_http(client):
    r = client.get("/ledger/verify")
    assert r.status_code == 200
    assert r.json()["verified"] is True, r.json()["problems"]


def test_the_published_verify_steps_actually_rebuild_the_manifest(client):
    """/verify tells a buyer the six manifest fields are "echoed in every response, so the manifest
    can be rebuilt from the response alone". That sentence was false: `tool` was signed but never
    echoed, so anyone following the steps hashed `tool: null`, got a different digest, and concluded
    the receipt was bad — on every service, every time. Caught by re-implementing the instructions
    from scratch rather than by reading them.

    This rebuilds the manifest exactly as the page says to, from the envelope alone.
    """
    import hashlib
    import json

    steps = client.get("/verify").json()
    fields = steps["manifest"]["fields"]

    ch = decode_challenge(client.post(PAID, json={}).headers[PAYMENT_REQUIRED_HEADER])
    env = client.post(PAID, json={"input": {"symbol": "BTC/USDT:USDT", "horizon": "1h"}},
                      headers={"PAYMENT-SIGNATURE": make_dev_payment(ch)}).json()

    for f in fields:
        assert f in env, f"/verify says {f!r} is echoed in every response, and it is not"

    rebuilt = {f: env[f] for f in fields}
    recomputed = "sha256:" + hashlib.sha256(
        json.dumps(rebuilt, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")).hexdigest()
    assert recomputed == env["receipt"]["manifest_sha256"], (
        "following the published instructions does not reproduce the receipt")

    from core.crypto import Signer
    assert Signer.verify(env["receipt"]["manifest_sha256"], env["receipt"]["signature"],
                         env["receipt"]["public_key"])
    assert env["receipt"]["public_key"] == steps["public_key_ed25519"]
