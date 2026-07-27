"""The HTTP surface — listing, x402 payment, the 40 paid endpoints, and the public record.

Four details here are not obvious and each has a rejected listing behind it. See OKX-REVIEW-RULES.md.

**The validator probes with no body.** It sends a bare request to the resource URL and expects a 402
with a challenge. A handler that validates input first answers with a 400 and the listing is rejected
for "not implementing x402" — so payment is resolved before the body is even parsed.

**The route table is keyed without a method.** `"* /a2mcp/<endpoint>"`. A method-qualified key makes
the validator's probe miss the route entirely and see no challenge.

**A malformed payment header returns a typed 400, never a dropped socket.** This was our own
rejection on a sibling agent: three malformed variants each produced "empty reply from server".

**Payment settles before the handler runs.** Every failure after settlement is one the customer has
already paid for, so a handler that cannot answer well must say so rather than return a confident
wrong answer — and a paid call arriving with an empty body gets the input contract at 200 rather than
an error, because charging for "you forgot the symbol" is charging for nothing.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response

from core.commit import AnchorError, Anchorer
from core.config import PROVENANCE, SERVICE_NAME, TAGLINE, get_settings
from core.conformal import Calibrator
from core.crypto import Signer
from core.forecaster import Forecaster
from core.kronos_runner import KronosRunner
from core.ledger import Ledger
from core.market_data import MarketData, MarketDataError
from core.markets import coverage_note
from services import a2a, accountability, forecast, market, metrics, risk  # noqa: F401
from services.context import Context
from services.registry import REGISTRY, ServiceError, envelope
from x402_bridge import (PAYMENT_REQUIRED_HEADER, PAYMENT_RESPONSE_HEADER, build_challenge,
                         malformed_payment, verify_payment)

log = logging.getLogger("horos")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

SETTINGS = get_settings()
REGISTRY.validate()          # a catalogue unfit to list must fail at import, not at review

_LEDGER = Ledger(SETTINGS.ledger_path, Signer(SETTINGS.signing_key_path))
_DATA = MarketData(cache_ttl=SETTINGS.cache_ttl_seconds,
                   timeout_ms=int(SETTINGS.http_timeout_seconds * 1000))
_RUNNER = KronosRunner(device=SETTINGS.__dict__.get("device", "cpu"))
CTX = Context(_DATA, _RUNNER, Forecaster(_DATA, _RUNNER), _LEDGER, _LEDGER.signer,
              Calibrator(), REGISTRY)

app = FastAPI(
    title=SERVICE_NAME,
    version="1.0",
    description="Market forecasts with a public, on-chain accuracy record. Every prediction is "
                "hash-committed before its window closes and scored against what happened.")


def _price(endpoint: str) -> str:
    svc = REGISTRY.get(endpoint)
    return svc.price_str() if svc else "0"


def _describe(endpoint: str) -> str:
    svc = REGISTRY.get(endpoint)
    return svc.description if svc else f"Horos service {endpoint}."


def _contract(endpoint: str) -> dict | None:
    svc = REGISTRY.get(endpoint)
    return svc.input_contract() if svc else None


# ── public pages ─────────────────────────────────────────────────────────────────────────────────

@app.get("/")
def root() -> dict:
    return {
        "name": SERVICE_NAME,
        "tagline": TAGLINE,
        "what": "Market forecasts judged by a public record, not by their own confidence.",
        "provenance": PROVENANCE,
        "services": len(REGISTRY.all()),
        "docs": "/services",
        "record": "/scorecard",
        "verify": "/verify",
        "health": "/health",
        "ledger": "/ledger",
    }


@app.get("/health")
def health() -> dict:
    counts = CTX.counts()
    return {"status": "ok", "services": len(REGISTRY.all()), "ledger": counts,
            "ts": int(time.time())}


@app.get("/services")
def services() -> dict:
    out = []
    for s in REGISTRY.all():
        out.append({**s.listing(),
                    "url": f"{SETTINGS.public_base_url.rstrip('/')}/a2mcp/{s.endpoint}"})
    return {"service": SERVICE_NAME, "count": len(out), "coverage": coverage_note(),
            "services": out}


@app.get("/.well-known/x402")
def well_known_x402() -> dict:
    """The route table, keyed **without** a method.

    A method-qualified key (`"POST /a2mcp/..."`) causes the OKX validator's probe — which does not
    use POST — to miss the route entirely and conclude there is no challenge.
    """
    routes: dict[str, Any] = {}
    for s in REGISTRY.all():
        routes[f"* /a2mcp/{s.endpoint}"] = {
            "scheme": SETTINGS.x402_scheme,
            "network": SETTINGS.x402_network,
            "asset": SETTINGS.x402_asset,
            "amount": str(int(round(s.price_usdt * 10 ** SETTINGS.x402_asset_decimals))),
            "payTo": SETTINGS.pay_to,
            "maxTimeoutSeconds": SETTINGS.x402_max_timeout_seconds,
            "extra": {"name": SETTINGS.x402_asset_name,
                      "version": SETTINGS.x402_asset_version,
                      "decimals": SETTINGS.x402_asset_decimals},
            "description": s.description,
        }
    return {"x402Version": SETTINGS.x402_version, "routes": routes}


@app.get("/scorecard", response_class=Response)
def scorecard_page() -> Response:
    """The public record, rendered. Free and unauthenticated: evidence nobody can read is not
    evidence."""
    import scorecard_page as page
    return Response(content=page.render(CTX), media_type="text/html; charset=utf-8")


@app.get("/proof", response_class=Response)
def proof_deck() -> Response:
    """Every service, bought for real, with the answer it returned.

    Rendered from a recorded run rather than written by hand, so the page cannot claim a number that
    did not happen. Free and unauthenticated: evidence nobody can read is not evidence.
    """
    import proof
    return Response(content=proof.page(), media_type="text/html; charset=utf-8")


@app.get("/ledger")
def ledger_download(limit: int = 0) -> Response:
    """The whole ledger, as JSONL, so anyone can replay the scoring themselves."""
    lines = []
    for e in _LEDGER:
        lines.append(json.dumps({"seq": e.seq, "kind": e.kind, "at": e.at, "body": e.body,
                                 "prev": e.prev, "hash": e.hash},
                                sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    if limit:
        lines = lines[-abs(limit):]
    return Response(content="\n".join(lines) + "\n", media_type="application/x-ndjson",
                    headers={"Content-Disposition": 'attachment; filename="horos-ledger.jsonl"'})


@app.get("/ledger/verify")
def ledger_verify() -> dict:
    """Recompute the whole chain and every signature. Run this against us."""
    ok, problems = _LEDGER.verify()
    return {
        "verified": ok,
        "problems": problems,
        "entries": _LEDGER.count(),
        "head": _LEDGER.head(),
        "public_key_ed25519": _LEDGER.signer.public_key,
        "checks": ["sequence numbers are contiguous",
                   "each entry's prev is the previous entry's hash",
                   "each stored hash matches its own contents",
                   "each signature verifies against a manifest rebuilt from the entry's own fields",
                   "every signature is by the published key"],
        "note": ("if this ever returns false, the ledger has been edited and Horos is broken. "
                 "It is served rather than asserted so you never have to take our word for it."),
    }


@app.get("/verify")
def verify_instructions() -> dict:
    """Everything needed to check a Horos receipt and a Horos anchor without trusting Horos."""
    from core.commit import FORMAT_VERSION, MAGIC, PAYLOAD_BYTES
    return {
        "public_key_ed25519": _LEDGER.signer.public_key,
        "algorithm": "Ed25519",
        "signed_bytes": "the UTF-8 bytes of receipt.manifest_sha256, verbatim",
        "manifest": {
            "fields": ["endpoint", "input_sha256", "output_sha256", "tool", "price_usdt", "job_id"],
            "how": "sha256 of canonical JSON — sorted keys, separators (',',':'), no whitespace, "
                   "ensure_ascii false. The digest carries a 'sha256:' prefix.",
            "note": "all six are echoed in every response, so the manifest can be rebuilt from the "
                    "response alone.",
        },
        "python": (
            "import hashlib, json\n"
            "from nacl.signing import VerifyKey\n"
            "m = {'endpoint': env['endpoint'], 'input_sha256': env['input_sha256'],\n"
            "     'output_sha256': env['output_sha256'], 'tool': 'horos',\n"
            "     'price_usdt': env['price_usdt'], 'job_id': env['job_id']}\n"
            "blob = json.dumps(m, sort_keys=True, separators=(',', ':'), ensure_ascii=False)\n"
            "d = 'sha256:' + hashlib.sha256(blob.encode('utf-8')).hexdigest()\n"
            "assert d == env['receipt']['manifest_sha256']\n"
            "VerifyKey(bytes.fromhex(env['receipt']['public_key'])).verify(\n"
            "    d.encode('utf-8'), bytes.fromhex(env['receipt']['signature']))"),
        "anchors": {
            "chain": SETTINGS.x402_network,
            "explorer": SETTINGS.xlayer_explorer,
            "calldata_layout": {
                "bytes_0_5": f"ASCII {MAGIC.decode()}",
                "byte_5": f"format version, currently {FORMAT_VERSION}",
                "bytes_6_38": "the 32-byte sha256 ledger head",
                "bytes_38_46": "uint64 big-endian entry count",
                "total": PAYLOAD_BYTES,
            },
            "python": ("raw = bytes.fromhex(tx_input.removeprefix('0x'))\n"
                       "assert raw[:5] == b'HOROS'\n"
                       "head = 'sha256:' + raw[6:38].hex()\n"
                       "entries = int.from_bytes(raw[38:46], 'big')"),
            "note": "pull the input field of any Horos anchor transaction from an X Layer explorer "
                    "and decode it. It must match a head in the ledger you downloaded.",
        },
        "ledger": "GET /ledger downloads the whole chain; GET /ledger/verify recomputes it.",
    }


# ── the paid surface ─────────────────────────────────────────────────────────────────────────────

def _headers_for(payment) -> dict:
    return {PAYMENT_RESPONSE_HEADER: payment.response_header} if payment.response_header else {}


def _challenge_response(endpoint: str, extra: dict | None = None, code: str = "") -> Response:
    header_val, challenge = build_challenge(endpoint, _price(endpoint), SETTINGS,
                                            _describe(endpoint), _contract(endpoint))
    body = {"error": "payment required", "endpoint": endpoint,
            "price_usdt": _price(endpoint), "challenge": challenge}
    if code:
        body["code"] = code
    if extra:
        body.update(extra)
    return JSONResponse(status_code=402, content=body,
                        headers={PAYMENT_REQUIRED_HEADER: header_val})


@app.api_route("/a2mcp/{endpoint:path}",
               methods=["GET", "POST", "PUT", "PATCH", "HEAD", "OPTIONS", "DELETE"])
async def paid_endpoint(endpoint: str, request: Request,
                        x_payment: str | None = Header(default=None),
                        payment_signature: str | None = Header(default=None)) -> Response:
    started = time.time()
    svc = REGISTRY.get(endpoint)
    if svc is None:
        return JSONResponse(status_code=404,
                            content={"error": f"unknown service '{endpoint}'",
                                     "code": "unknown_service",
                                     "services": REGISTRY.endpoints()})

    # The OKX agentic wallet sends PAYMENT-SIGNATURE; X-PAYMENT is the older name and is still
    # accepted. Reading only one hands a 402 to a customer who has already paid.
    #
    # Absent and blank are different failures and get different answers. No header at all means no
    # payment was attempted — that is the validator's probe, and it must receive a 402 with a
    # challenge. A header that is present but empty is a client that built the request wrong; paying
    # again will not fix it, so it gets a typed 400 naming the field, the same as any other
    # unparseable header.
    supplied = [h for h in (payment_signature, x_payment) if h is not None]
    if not supplied:
        # No body is read at all on this path.
        return _challenge_response(endpoint)
    authorization = next((h for h in supplied if h.strip()), "")
    if not authorization:
        return JSONResponse(status_code=400, content={
            "error": "the PAYMENT-SIGNATURE header was present but empty.",
            "code": "malformed_payment_signature", "field": "PAYMENT-SIGNATURE",
            "endpoint": endpoint,
            "expected": "base64 of the x402 v2 payment payload from your wallet. Call this "
                        "endpoint with no payment header at all to get a challenge to sign."})

    # An unreadable header is answered first. A caller with both a broken header and a thin body has
    # two problems, and "your header is malformed" is the one that explains why retrying the payment
    # will not help — so it must not be masked by the input check below.
    broken = malformed_payment(authorization)
    if broken is not None:
        return JSONResponse(status_code=400, content={
            "error": broken.detail, "code": broken.code, "field": broken.field,
            "endpoint": endpoint,
            "expected": "base64 of the x402 v2 payment payload from your wallet. Call this "
                        "endpoint with no payment header to get a challenge to sign."})

    # Read and check the input *before* settling. Payment used to clear first, so a caller who left
    # out a required field was charged and handed an error — a fee on their statement and no work
    # done. The status codes below are deliberately identical to what this endpoint returned before
    # (200 with the contract for an empty body, 422 for a partial one); the only change is that the
    # money no longer moves. Keeping the codes fixed means no client and no listing check can notice
    # anything except that they stopped being billed for it.
    try:
        payload = await request.json()
    except Exception:                                                # noqa: BLE001
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    node_input = payload.get("input") if isinstance(payload.get("input"), dict) else payload
    node_input = {k: v for k, v in node_input.items() if k not in ("options", "idempotency_key")}

    missing = [f for f in svc.required if node_input.get(f) in (None, "")]
    if missing:
        contract = svc.input_contract()
        unbilled = ("Nothing was billed for this call — your authorization was not settled. Send "
                    "the example above with payment to get the real result.")
        if not node_input:
            return JSONResponse(status_code=200, content={
                "endpoint": endpoint, "status": "input_required",
                "what_this_does": svc.summary, "returns": svc.returns, "depth": svc.depth,
                "required": contract["required"], "properties": contract["properties"],
                "example_request": {"input": contract["example"]},
                "not_charged": unbilled})
        return JSONResponse(status_code=422, content={
            "error": f"missing required field(s): {', '.join(missing)}",
            "code": "missing_required_field", "endpoint": endpoint,
            "required": list(svc.required),
            "example_request": {"input": contract["example"]},
            "not_charged": unbilled})

    payment = verify_payment(authorization, SETTINGS, endpoint=endpoint,
                             fee_usdt=_price(endpoint))
    if not payment.ok:
        if payment.malformed:
            # The caller's bug, not a payment problem — paying again changes nothing, so a typed 400
            # naming the field rather than a challenge that invites a pointless retry.
            return JSONResponse(status_code=400, content={
                "error": payment.detail, "code": payment.code, "field": payment.field,
                "endpoint": endpoint,
                "expected": "base64 of the x402 v2 payment payload from your wallet. Call this "
                            "endpoint with no payment header to get a challenge to sign."})
        # A well-formed authorization we declined gets a *fresh* challenge. It is rebuilt rather
        # than reusing the earlier response's headers: copying those carried a stale content-length
        # onto a longer body, so uvicorn truncated the socket and the caller saw "empty reply from
        # server" on the one path where a reason matters most.
        return _challenge_response(endpoint, {"payment_error": payment.detail}, payment.code)

    notes: list[str] = []
    try:
        output = svc.handler(node_input, CTX)
    except ServiceError as e:
        return JSONResponse(status_code=e.status, headers=_headers_for(payment),
                            content={**e.as_dict(), "endpoint": endpoint,
                                     "example_request": {"input": svc.input_contract()["example"]}})
    except MarketDataError as e:
        return JSONResponse(status_code=502, headers=_headers_for(payment),
                            content={**e.as_dict(), "endpoint": endpoint,
                                     "note": "an upstream market data source failed. This is "
                                             "reported rather than answered around."})
    except Exception as e:                                           # noqa: BLE001
        # Never a bare 500. A typed code and the actual exception, because a caller who has paid
        # deserves to know what broke.
        log.exception("handler failed for %s", endpoint)
        return JSONResponse(status_code=500, headers=_headers_for(payment), content={
            "error": f"{type(e).__name__}: {e}", "code": "handler_failed", "endpoint": endpoint,
            "note": "this is a defect in Horos, not in your request. The failure is reported "
                    "rather than disguised as an empty result."})

    if payment.tx:
        notes.append(f"settled on {SETTINGS.x402_network}, transaction {payment.tx}")

    env = envelope(svc, node_input, output, _LEDGER.signer, notes=notes, started=started)
    return JSONResponse(status_code=200, headers=_headers_for(payment), content=env)


@app.on_event("startup")
def _startup() -> None:
    log.info("%s starting: %d services, ledger %s", SERVICE_NAME, len(REGISTRY.all()),
             CTX.counts())
    if SETTINGS.anchor_enabled:
        try:
            a = Anchorer(SETTINGS.xlayer_rpc, SETTINGS.xlayer_chain_id, SETTINGS.anchor_key,
                         SETTINGS.xlayer_explorer)
            log.info("anchor wallet %s holds %.9f OKB", a.address, a.balance_wei() / 1e18)
        except AnchorError as e:
            log.error("anchoring unavailable: %s", e)
    if SETTINGS.pay_to.startswith("0x0000"):
        log.warning("HOROS_PAYTO is unset — payments would go to the zero address. Set it before "
                    "listing.")
