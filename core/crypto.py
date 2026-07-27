"""Canonical serialisation, hashing and Ed25519 signing.

Everything the ledger promises rests on one property: two people who hold the same forecast must
compute the same bytes for it. If canonicalisation is sloppy — key order, float formatting, unicode
escaping — then a signature verifies for us and fails for a buyer, and the whole record becomes
unfalsifiable in the wrong direction.

So the rules here are strict and boring on purpose:

  * keys sorted, always
  * no insignificant whitespace
  * unicode emitted as unicode, never as \\u escapes, so the bytes are the text
  * floats rendered through ``repr`` semantics, which round-trips exactly in Python and matches the
    shortest representation that reads back identically

The signature covers a *manifest* — a small fixed set of fields — rather than the whole payload, so
that a buyer can verify the claim without us having to promise byte-stability of every incidental
field we might add later.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError


def canonical(obj: Any) -> str:
    """The one true text form of a JSON-compatible value."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def digest(obj: Any) -> str:
    """SHA-256 of the canonical form, hex, prefixed so it is never mistaken for something else."""
    return "sha256:" + hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def digest_hex(obj: Any) -> str:
    """Bare hex digest, for places that need 32 bytes without the prefix — on-chain commits."""
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


class Signer:
    """Ed25519 over the canonical manifest.

    The key is generated on first use and persisted. Losing it invalidates nothing already published
    — old signatures still verify against the old public key — but it does break the continuity a
    buyer relies on, so the path is deliberately outside the repo tree.
    """

    def __init__(self, key_path: str | Path):
        self.key_path = Path(key_path)
        if self.key_path.exists():
            self._key = SigningKey(bytes.fromhex(self.key_path.read_text().strip()))
        else:
            self._key = SigningKey.generate()
            self.key_path.parent.mkdir(parents=True, exist_ok=True)
            self.key_path.write_text(bytes(self._key).hex())
            # Owner-only where the platform supports it; harmless where it does not.
            try:
                self.key_path.chmod(0o600)
            except (OSError, NotImplementedError):
                pass

    @property
    def public_key(self) -> str:
        return bytes(self._key.verify_key).hex()

    def sign(self, manifest: dict) -> dict:
        """Return the signature block that travels with a forecast."""
        manifest_digest = digest(manifest)
        signature = self._key.sign(manifest_digest.encode("utf-8")).signature.hex()
        return {"algorithm": "ed25519",
                "manifest_sha256": manifest_digest,
                "signature": signature,
                "public_key": self.public_key}

    @staticmethod
    def verify(manifest_sha256: str, signature: str, public_key: str) -> bool:
        """Check a signature without needing the private key or asking us anything.

        This is the function published at /verify, and a test executes the published snippet against
        a real receipt — because verification instructions that do not work are worse than none.
        """
        try:
            VerifyKey(bytes.fromhex(public_key)).verify(
                manifest_sha256.encode("utf-8"), bytes.fromhex(signature))
            return True
        except (BadSignatureError, ValueError):
            return False
