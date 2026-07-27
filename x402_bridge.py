"""x402 v2 for the OKX.AI marketplace — the single biggest review gate.

Every constant and every branch here exists because a real listing was rejected without it. See
`OKX-REVIEW-RULES.md`; the taxonomy below maps to its sections.

**The unpaid call must return HTTP 402.** Not 200, not 400, not 405. The validator probes the
resource URL *with no body at all* and reads the status code. A handler that validates input first
answers that probe with a 400 and the listing is rejected for "not implementing x402" (§1.2 — four
agents).

**The challenge travels in the `PAYMENT-REQUIRED` header**, base64-encoded. It is mirrored into the
body so a human can read it, but the header is what is checked.

**`decimals` must live inside `extra`.** USD₮0 is not in OKX's token list and a top-level `decimals`
is dropped by their canonical re-serialisation, so the amount is then misread by six orders of
magnitude.

**A malformed payment header must return a typed 400, never a dropped socket.** This was *our own*
rejection on Doxa #9626: three malformed `PAYMENT-SIGNATURE` variants each produced "empty reply from
server" with no HTTP response at all (§1.10).

**Both header names are read.** The OKX agentic wallet sends `PAYMENT-SIGNATURE`; `X-PAYMENT` is the
older name. Reading only one hands a 402 to a customer who has already paid.

**Terms come from this server, never from the request.** `paymentRequirements` sent to the
facilitator is built from our price for this endpoint. Echoing back the caller's `accepted` block
would let anyone declare they owed a fraction of a cent and have the facilitator agree.
"""
from __future__ import annotations

import base64
import json
import threading
import time
import uuid
from decimal import ROUND_DOWN, Decimal

from core.config import Settings

PAYMENT_REQUIRED_HEADER = "PAYMENT-REQUIRED"
PAYMENT_HEADER = "X-PAYMENT"
PAYMENT_RESPONSE_HEADER = "X-PAYMENT-RESPONSE"

_ISSUED: dict[str, dict] = {}
_SPENT: set[str] = set()
_LOCK = threading.Lock()


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def fee_to_min_units(fee: str | float, decimals: int) -> str:
    """'0.05' with 6 decimals -> '50000'. Integer minor units, as a string."""
    q = Decimal(str(fee)) * (Decimal(10) ** decimals)
    return str(int(q.quantize(Decimal(1), rounding=ROUND_DOWN)))


def build_challenge(endpoint: str, fee_usdt: str, settings: Settings, description: str,
                    input_contract: dict | None = None) -> tuple[str, dict]:
    """Return (base64 header value, challenge dict) for one paid endpoint."""
    nonce = uuid.uuid4().hex
    amount = fee_to_min_units(fee_usdt, settings.x402_asset_decimals)
    resource_url = f"{settings.public_base_url.rstrip('/')}/a2mcp/{endpoint}"
    challenge = {
        "x402Version": settings.x402_version,
        "resource": {
            "url": resource_url,
            "description": description,
            "mimeType": "application/json",
            # What the call needs. A buying agent reads the challenge to work out how to invoke the
            # service; without it, it sends an empty body, pays, and gets an error. That is not
            # hypothetical — it is what happened on the first real A2A purchase of a sibling ASP.
            **({"inputSchema": input_contract} if input_contract else {}),
        },
        "accepts": [{
            "scheme": settings.x402_scheme,
            "network": settings.x402_network,
            "asset": settings.x402_asset,
            "amount": amount,
            "payTo": settings.pay_to,
            "maxTimeoutSeconds": settings.x402_max_timeout_seconds,
            "extra": {"name": settings.x402_asset_name,
                      "version": settings.x402_asset_version,
                      "decimals": settings.x402_asset_decimals},
        }],
        "nonce": nonce,
    }
    with _LOCK:
        _ISSUED[nonce] = {"endpoint": endpoint, "amount": amount, "url": resource_url,
                          "ts": time.time()}
        if len(_ISSUED) > 4096:
            cutoff = time.time() - 3600
            for k in [k for k, v in _ISSUED.items() if v["ts"] < cutoff]:
                _ISSUED.pop(k, None)
    return base64.b64encode(canonical(challenge)).decode("ascii"), challenge


def decode_challenge(header_value: str) -> dict:
    return json.loads(base64.b64decode(header_value))


class PaymentResult:
    """The outcome of checking a payment and, when it failed, why — in a form a caller can act on.

    `malformed` separates two failures that must be handled differently. A header we cannot decode is
    the caller's bug: paying again changes nothing, so it earns a 400 naming the field. A well-formed
    authorization we decline (spent nonce, short amount) is a payment problem: the caller really
    should pay again, so it earns a 402 with a fresh challenge.
    """

    def __init__(self, ok: bool, detail: str, response_header: str | None = None,
                 code: str = "", field: str = "", malformed: bool = False, tx: str = ""):
        self.ok = ok
        self.detail = detail
        self.response_header = response_header
        self.code = code or ("" if ok else "payment_rejected")
        self.field = field
        self.malformed = malformed
        self.tx = tx


def verify_payment(header_value: str, settings: Settings, endpoint: str = "",
                   fee_usdt: str = "") -> PaymentResult:
    """Verify an authorization and, in facilitator mode, settle it on X Layer."""
    name = "PAYMENT-SIGNATURE"
    raw = (header_value or "").strip()
    if not raw:
        return PaymentResult(False, f"the {name} header was empty.",
                             code="malformed_payment_signature", field=name, malformed=True)
    try:
        decoded = base64.b64decode(raw, validate=False)
    except Exception as e:                                               # noqa: BLE001
        return PaymentResult(
            False, f"the {name} header is not valid base64 ({e}). It must be the base64 encoding of "
                   f"the x402 v2 payment payload your wallet returns.",
            code="malformed_payment_signature", field=name, malformed=True)
    try:
        payload = json.loads(decoded)
    except Exception as e:                                               # noqa: BLE001
        return PaymentResult(
            False, f"the {name} header decoded, but not to JSON ({e}). Expected a JSON object with "
                   f"'x402Version', 'accepted' and 'payload'.",
            code="malformed_payment_signature", field=name, malformed=True)
    if not isinstance(payload, dict):
        return PaymentResult(
            False, f"the {name} header decoded to {type(payload).__name__}, not a JSON object.",
            code="malformed_payment_signature", field=name, malformed=True)

    inner = payload.get("payload")
    if isinstance(inner, dict) and "authorization" in inner:
        return _verify_v2(payload, settings, endpoint, fee_usdt)

    # --- dev / signature mode, used by the offline suite ---------------------------------------
    nonce = payload.get("nonce")
    with _LOCK:
        issued = _ISSUED.get(nonce) if nonce else None
    if not issued:
        return PaymentResult(False, "unknown or expired challenge nonce")
    if payload.get("scheme") != settings.x402_scheme:
        return PaymentResult(False, "scheme mismatch")
    if payload.get("network") != settings.x402_network:
        return PaymentResult(False, "network mismatch (must be X Layer eip155:196)")
    if str(payload.get("amount")) != issued["amount"]:
        return PaymentResult(False, "amount mismatch against the challenge")
    if settings.x402_mode == "facilitator":
        return PaymentResult(False, "this endpoint settles on X Layer; send a signed x402 v2 "
                                    "authorization in the PAYMENT-SIGNATURE header")
    ok, detail = _verify_dev_signature(payload, nonce)
    if not ok:
        return PaymentResult(False, detail)
    with _LOCK:
        _ISSUED.pop(nonce, None)          # single use, so the same payment cannot be replayed
    resp = base64.b64encode(canonical({"success": True, "nonce": nonce,
                                       "settledAt": time.time()})).decode("ascii")
    return PaymentResult(True, "verified", resp)


def _verify_v2(payload: dict, settings: Settings, endpoint: str, fee_usdt: str) -> PaymentResult:
    accepted = payload.get("accepted") or {}
    auth = (payload.get("payload") or {}).get("authorization") or {}
    resource = payload.get("resource") or {}

    required = fee_to_min_units(fee_usdt or "0", settings.x402_asset_decimals)
    expected_url = f"{settings.public_base_url.rstrip('/')}/a2mcp/{endpoint}" if endpoint else ""

    # Without this check, an authorization bought for the cheapest endpoint unlocks the dearest one.
    if expected_url and resource.get("url") and resource["url"] != expected_url:
        return PaymentResult(False, f"this authorization is for {resource['url']}, "
                                    f"not {expected_url}")
    if str(accepted.get("network") or "") != settings.x402_network:
        return PaymentResult(False, "network mismatch (must be X Layer eip155:196)")
    if (accepted.get("asset") or "").lower() != settings.x402_asset.lower():
        return PaymentResult(False, "asset mismatch")
    if (accepted.get("payTo") or "").lower() != settings.pay_to.lower():
        return PaymentResult(False, "payTo mismatch")
    try:
        if int(auth.get("value") or 0) < int(required):
            return PaymentResult(False, f"authorised {auth.get('value')} minor units, "
                                        f"{required} required")
    except (TypeError, ValueError):
        return PaymentResult(False, "unreadable authorised value")

    auth_nonce = auth.get("nonce")
    # The token contract is the real defence against replay, but a replay inside the validity window
    # would still cost us the work before the chain rejected it.
    if auth_nonce:
        with _LOCK:
            if auth_nonce in _SPENT:
                return PaymentResult(False, "this authorization has already been used")

    requirements = {
        "scheme": settings.x402_scheme,
        "network": settings.x402_network,
        "asset": settings.x402_asset,
        "amount": required,
        "payTo": settings.pay_to,
        "maxTimeoutSeconds": settings.x402_max_timeout_seconds,
        "resource": expected_url or settings.public_base_url,
        "extra": {"name": settings.x402_asset_name, "version": settings.x402_asset_version,
                  "decimals": settings.x402_asset_decimals},
    }
    if settings.x402_mode != "facilitator":
        return PaymentResult(False, "x402 settlement is not enabled on this deployment")

    ok, detail, tx = _settle(payload, requirements, settings)
    if not ok:
        return PaymentResult(False, detail)
    if auth_nonce:
        with _LOCK:
            _SPENT.add(auth_nonce)
    resp = base64.b64encode(canonical({"success": True, "network": settings.x402_network,
                                       "transaction": tx,
                                       "settledAt": time.time()})).decode("ascii")
    return PaymentResult(True, detail, resp, tx=tx)


def _settle(payment_payload: dict, requirements: dict,
            settings: Settings) -> tuple[bool, str, str]:
    """Verify then settle through OKX. **Fail closed: no settlement, no service.**"""
    if not settings.facilitator_configured:
        return False, "settlement is unavailable: OKX facilitator credentials are not configured", ""
    try:
        from core.okx_facilitator import FacilitatorError, OkxFacilitator
    except Exception as e:                                               # noqa: BLE001
        return False, f"facilitator client unavailable: {e}", ""

    fac = OkxFacilitator(settings.okx_api_key, settings.okx_api_secret,
                         settings.okx_api_passphrase,
                         base_url=settings.x402_facilitator_url or settings.okx_facilitator_base_url,
                         sync_settle=settings.okx_sync_settle)
    try:
        v = fac.verify(payment_payload, requirements)
    except FacilitatorError as e:
        return False, f"facilitator verify failed: {e}", ""
    if not (v.get("isValid") is True or v.get("valid") is True):
        return False, f"the facilitator rejected the authorization: {v.get('invalidReason') or v}", ""
    try:
        s = fac.settle(payment_payload, requirements)
    except FacilitatorError as e:
        return False, f"facilitator settle failed: {e}", ""
    tx = str(s.get("transaction") or s.get("txHash") or "")
    if not (s.get("success") is True or tx):
        return False, f"settlement did not complete: {str(s)[:180]}", ""
    return True, f"settled on {settings.x402_network}" + (f" tx {tx}" if tx else ""), tx


def _verify_dev_signature(payload: dict, nonce: str) -> tuple[bool, str]:
    """Offline mode: the payer proves authorization by signing the nonce."""
    pub, sig = payload.get("payer_pubkey"), payload.get("signature")
    if not pub or not sig:
        return False, "missing payer_pubkey/signature"
    try:
        from nacl.signing import VerifyKey
        VerifyKey(bytes.fromhex(pub)).verify(nonce.encode(), bytes.fromhex(sig))
        return True, "signature ok"
    except Exception as e:                                               # noqa: BLE001
        return False, f"bad signature: {e}"


def make_dev_payment(challenge: dict) -> str:
    """Forge a valid offline payment for a challenge. Used by the test suite and the self-probe."""
    from nacl.signing import SigningKey
    sk = SigningKey.generate()
    nonce = challenge["nonce"]
    accept = challenge["accepts"][0]
    return base64.b64encode(canonical({
        "x402Version": challenge["x402Version"],
        "scheme": accept["scheme"], "network": accept["network"], "asset": accept["asset"],
        "amount": accept["amount"], "nonce": nonce,
        "payer_pubkey": bytes(sk.verify_key).hex(),
        "signature": sk.sign(nonce.encode()).signature.hex(),
    })).decode("ascii")
