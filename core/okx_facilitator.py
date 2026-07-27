"""A Python client for the OKX x402 facilitator on X Layer.

OKX ship this as a TypeScript package (`@okxweb3/app-x402-core`). Horos is Python, so the two calls it
needs are implemented directly against the same HTTP API. Endpoints, header names and the signing
scheme were read out of that package rather than guessed:

    POST /api/v6/pay/x402/verify     does this authorization actually pay what was asked?
    POST /api/v6/pay/x402/settle     move the funds

    OK-ACCESS-SIGN = base64(HMAC-SHA256(secret, timestamp + METHOD + path + body))

Verify and settle are kept as two calls, in that order, because they answer different questions and
only the second one moves money. A failure in either is returned as a failure — this layer never
reports a payment as accepted unless the facilitator said it settled, because the handler runs
afterwards and would otherwise do paid work for free.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timezone
from typing import Any

import requests

VERIFY_PATH = "/api/v6/pay/x402/verify"
SETTLE_PATH = "/api/v6/pay/x402/settle"
SUPPORTED_PATH = "/api/v6/pay/x402/supported"
DEFAULT_BASE = "https://web3.okx.com"


class FacilitatorError(RuntimeError):
    """The facilitator could not be reached, or refused the request."""


class OkxFacilitator:
    def __init__(self, api_key: str, api_secret: str, passphrase: str, *,
                 base_url: str = DEFAULT_BASE, sync_settle: bool = True,
                 timeout: int = 30):
        if not (api_key and api_secret and passphrase):
            raise FacilitatorError("api key, secret and passphrase are all required")
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url = (base_url or DEFAULT_BASE).rstrip("/")
        self.sync_settle = sync_settle
        self.timeout = timeout

    # --- signing -------------------------------------------------------------------------------

    def _headers(self, method: str, path: str, body: str) -> dict[str, str]:
        # OKX requires milliseconds and a trailing Z; a bare isoformat() is rejected as malformed.
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
            f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"
        prehash = f"{ts}{method.upper()}{path}{body or ''}"
        sign = base64.b64encode(
            hmac.new(self.api_secret.encode("utf-8"), prehash.encode("utf-8"),
                     hashlib.sha256).digest()).decode("ascii")
        return {"OK-ACCESS-KEY": self.api_key,
                "OK-ACCESS-SIGN": sign,
                "OK-ACCESS-TIMESTAMP": ts,
                "OK-ACCESS-PASSPHRASE": self.passphrase,
                "Content-Type": "application/json"}

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        # The signature covers the exact bytes sent, so the body is serialised once and reused.
        body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        try:
            r = requests.post(self.base_url + path, data=body.encode("utf-8"),
                              headers=self._headers("POST", path, body), timeout=self.timeout)
        except Exception as e:  # noqa: BLE001
            raise FacilitatorError(f"{type(e).__name__}: {str(e)[:160]}") from e
        try:
            data = r.json()
        except Exception as e:  # noqa: BLE001
            raise FacilitatorError(f"HTTP {r.status_code}, unparseable body: "
                                   f"{r.text[:200]!r}") from e
        if r.status_code != 200:
            raise FacilitatorError(f"HTTP {r.status_code}: {str(data)[:200]}")
        # OKX wraps results as {"code": "0", "data": [...]} — a non-zero code is a refusal even
        # though the HTTP status is 200, and treating it as success would let unpaid work through.
        code = str(data.get("code", "0"))
        if code not in ("0", "00000"):
            raise FacilitatorError(f"code={code} msg={data.get('msg') or data.get('message')}")
        inner = data.get("data", data)
        if isinstance(inner, list):
            inner = inner[0] if inner else {}
        return inner if isinstance(inner, dict) else {"result": inner}

    # --- the two calls -------------------------------------------------------------------------

    def verify(self, payment_payload: dict[str, Any],
               payment_requirements: dict[str, Any]) -> dict[str, Any]:
        return self._post(VERIFY_PATH, {"x402Version": payment_payload.get("x402Version", 2),
                                        "paymentPayload": payment_payload,
                                        "paymentRequirements": payment_requirements})

    def settle(self, payment_payload: dict[str, Any],
               payment_requirements: dict[str, Any]) -> dict[str, Any]:
        return self._post(SETTLE_PATH, {"x402Version": payment_payload.get("x402Version", 2),
                                        "paymentPayload": payment_payload,
                                        "paymentRequirements": payment_requirements,
                                        "syncSettle": bool(self.sync_settle)})

    def supported(self) -> dict[str, Any]:
        """Which schemes and networks the facilitator will accept. Useful as a credential check.

        A GET with no body, matching OKX's own client — the signature therefore covers an empty body
        string. Sending it as a POST returns a generic `code=-1 unknown error` rather than an error
        that names the problem.
        """
        try:
            r = requests.get(self.base_url + SUPPORTED_PATH,
                             headers=self._headers("GET", SUPPORTED_PATH, ""),
                             timeout=self.timeout)
        except Exception as e:  # noqa: BLE001
            raise FacilitatorError(f"{type(e).__name__}: {str(e)[:160]}") from e
        try:
            data = r.json()
        except Exception as e:  # noqa: BLE001
            raise FacilitatorError(f"HTTP {r.status_code}: {r.text[:200]!r}") from e
        if r.status_code != 200 or str(data.get("code", "0")) not in ("0", "00000"):
            raise FacilitatorError(f"HTTP {r.status_code} code={data.get('code')} "
                                   f"msg={data.get('msg') or data.get('message')}")
        return data.get("data", data)
