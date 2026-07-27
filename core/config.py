"""Horos configuration. Everything from the environment; nothing secret in the tree.

The x402 constants are not adjustable in any meaningful sense — they are what the OKX marketplace
validates against, and getting one of them wrong is a rejected listing rather than a bug. They are
read from the environment anyway so that a test deployment can point at a different wallet without
editing code, but the defaults are the production values and were copied from the four listings that
already passed review.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

SERVICE_NAME = "Horos"
TAGLINE = "Marked before the outcome."

# The stone was planted before the loan was made, in public, where anyone could read it. That is the
# whole product, and it is worth keeping the sentence next to the code.
PROVENANCE = ("In classical Athens a horos was a marble slab set on the pledged property itself, "
              "inscribed with what was owed and to whom, so that anyone could read the claim before "
              "acting on it. The same word means a boundary, and in Aristotle the term of a "
              "proposition.")


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader. Only fills variables that are not already set."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


def _env(*names: str, default: str = "") -> str:
    """First of several environment names that is set.

    Several names per value because the Azure VM already carries the wallet configuration under the
    names the earlier ASPs used, and requiring a new spelling would mean a deployment that silently
    pays into the zero address.
    """
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


class Settings:
    def __init__(self) -> None:
        # ---- identity ----------------------------------------------------------------------
        self.public_base_url: str = _env("HOROS_PUBLIC_BASE_URL",
                                         default="https://horos.ivaronix.xyz")
        self.data_dir: str = _env("HOROS_DATA_DIR", default=".data")
        self.signing_key_path: str = _env("HOROS_SIGNING_KEY",
                                          default=".secrets/horos_ed25519.key")
        self._signing_key_hex: str | None = _env("HOROS_SIGNING_KEY_HEX") or None

        # ---- x402 / X Layer ----------------------------------------------------------------
        # Every constant below is checked by the OKX listing validator. See OKX-REVIEW-RULES.md §2.
        self.x402_version: int = 2
        self.x402_scheme: str = "exact"
        self.x402_network: str = _env("HOROS_NETWORK", default="eip155:196")   # X Layer mainnet
        self.x402_asset: str = _env("HOROS_ASSET",
                                    default="0x779ded0c9e1022225f8e0630b35a9b54be713736")  # USD₮0
        self.x402_asset_name: str = _env("HOROS_ASSET_NAME", default="USD₮0")
        self.x402_asset_version: str = _env("HOROS_ASSET_VERSION", default="1")
        self.x402_asset_decimals: int = int(_env("HOROS_ASSET_DECIMALS", default="6"))
        self.x402_max_timeout_seconds: int = int(_env("HOROS_X402_TIMEOUT", default="300"))
        self.pay_to: str = _env("HOROS_PAYTO", "X402_PAY_TO", "OKX_PAY_TO",
                                default="0x0000000000000000000000000000000000000000")
        self.x402_mode: str = _env("HOROS_X402_MODE", default="signature")
        self.x402_facilitator_url: str = _env("HOROS_FACILITATOR_URL")
        self.okx_facilitator_base_url: str = _env("OKX_FACILITATOR_BASE_URL",
                                                  default="https://web3.okx.com")
        self.okx_sync_settle: bool = _env("OKX_SYNC_SETTLE", default="true").lower() == "true"
        self._okx_api_key: str = _env("OKX_API_KEY")
        self._okx_api_secret: str = _env("OKX_API_SECRET", "OKX_SECRET_KEY")
        self._okx_api_passphrase: str = _env("OKX_API_PASSPHRASE", "OKX_PASSPHRASE")
        if _env("X402_ENABLED") == "1" and self.facilitator_configured:
            self.x402_mode = "facilitator"

        # ---- on-chain anchoring ------------------------------------------------------------
        # Anchoring writes a 32-byte digest into the calldata of a zero-value self-transfer on
        # X Layer. No contract to deploy, no contract to trust, and the calldata is permanent and
        # readable by anyone with the transaction hash.
        self.xlayer_rpc: str = _env("HOROS_XLAYER_RPC", default="https://rpc.xlayer.tech")
        self.xlayer_chain_id: int = int(_env("HOROS_XLAYER_CHAIN_ID", default="196"))
        self.xlayer_explorer: str = _env("HOROS_XLAYER_EXPLORER",
                                         default="https://www.oklink.com/xlayer/tx/")
        self._anchor_key: str = _env("HOROS_ANCHOR_KEY")           # secret: hex private key
        self.anchor_enabled: bool = bool(self._anchor_key)
        self.anchor_interval_seconds: int = int(_env("HOROS_ANCHOR_INTERVAL", default="3600"))

        # ---- market coverage ----------------------------------------------------------------
        # The scorecard is only meaningful per symbol and per horizon, so the served set is fixed
        # and small. See core/markets.py for why these three and why 1h.
        self.scored_symbols: list[str] = [s.strip() for s in _env(
            "HOROS_SCORED_SYMBOLS", default="BTC/USDT:USDT,ETH/USDT:USDT,SOL/USDT:USDT"
        ).split(",") if s.strip()]
        self.scored_timeframe: str = _env("HOROS_SCORED_TIMEFRAME", default="1h")

        # ---- fetch discipline ---------------------------------------------------------------
        self.http_timeout_seconds: float = float(_env("HOROS_HTTP_TIMEOUT", default="20"))
        self.max_response_bytes: int = int(_env("HOROS_MAX_RESPONSE_BYTES", default=str(8 << 20)))
        self.cache_ttl_seconds: int = int(_env("HOROS_CACHE_TTL", default="20"))

    # ---- derived paths ----------------------------------------------------------------------

    @property
    def ledger_path(self) -> Path:
        return Path(self.data_dir) / "ledger.jsonl"

    @property
    def signing_key_hex(self) -> str | None:
        return self._signing_key_hex

    @property
    def anchor_key(self) -> str:
        return self._anchor_key

    @property
    def okx_api_key(self) -> str:
        return self._okx_api_key

    @property
    def okx_api_secret(self) -> str:
        return self._okx_api_secret

    @property
    def okx_api_passphrase(self) -> str:
        return self._okx_api_passphrase

    @property
    def facilitator_configured(self) -> bool:
        return bool(self._okx_api_key and self._okx_api_secret and self._okx_api_passphrase)

    def redacted(self) -> dict:
        """Safe to log and safe to serve: presence of secrets, never their values."""
        return {
            "service": SERVICE_NAME,
            "base_url": self.public_base_url,
            "network": self.x402_network,
            "asset": self.x402_asset,
            "x402_mode": self.x402_mode,
            "facilitator_configured": self.facilitator_configured,
            "anchoring_enabled": self.anchor_enabled,
            "scored_symbols": self.scored_symbols,
            "scored_timeframe": self.scored_timeframe,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
