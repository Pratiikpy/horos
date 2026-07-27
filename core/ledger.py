"""The hash-chained forecast ledger.

This is the product. Everything else is a way of filling it.

A forecasting service is unfalsifiable at the moment of sale, so the only honest way to sell one is
to record every prediction before its outcome exists and publish how each turned out. That record is
worth exactly as much as it is hard to edit, which is why this is a chain and not a list:

    entry.prev = hash of the previous entry
    entry.hash = hash of (body + prev)

Remove an entry, reorder two, or quietly soften a bad forecast, and every hash after it changes. The
head is anchored on-chain periodically, so history cannot be rebuilt either — an altered chain would
have to reproduce a hash that is already public on X Layer.

Two kinds of entry, and the split is the whole point:

  * ``issued`` — written when the forecast is made, carrying the full distribution. Its digest goes
    on-chain *before the target window closes*, so the prediction provably predates its outcome.
  * ``scored`` — written when the window closes, carrying what actually happened and the scores.
    Scoring is deterministic, so anyone with the ledger and public market data can recompute it.

There is no update path and no delete path. A correction is a new entry referencing the old one.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .crypto import Signer, canonical, digest, digest_hex

GENESIS = "sha256:" + "0" * 64

# The fields the signature covers, per entry kind. Both `record_*` and `verify` build the manifest
# through `manifest_for`, so there is exactly one definition of what was signed.
#
# This indirection is not decoration. The first version of `verify` checked the signature against the
# digest *stored in the entry* without rebuilding it from the body — so editing a field and leaving
# the old digest in place produced a forged entry that verified. A single builder makes that
# impossible to reintroduce: if the body changes, the rebuilt manifest changes, and the digest no
# longer matches what was signed.
_MANIFEST_FIELDS: dict[str, tuple[str, ...]] = {
    "issued": ("symbol", "timeframe", "horizon", "model", "model_version", "issued_for", "window",
               "distribution_sha256", "inputs_sha256"),
    "scored": ("forecast_id", "actual_sha256", "scores", "scorer_version"),
    "void": ("forecast_id", "reason", "voided_at"),
}


def manifest_for(kind: str, body: dict) -> dict | None:
    """The exact object that was signed for an entry of this kind, rebuilt from its body.

    Returns None for kinds that carry no signature — anchors, whose authenticity is established by
    the chain itself rather than by us asserting it.
    """
    fields = _MANIFEST_FIELDS.get(kind)
    if not fields:
        return None
    return {f: body[f] for f in fields if f in body}


def utcnow() -> str:
    """One timestamp format everywhere, to the second, always UTC, always suffixed Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Entry:
    """One line of the ledger, after parsing."""
    seq: int
    kind: str
    at: str
    body: dict
    prev: str
    hash: str

    @property
    def forecast_id(self) -> str:
        return self.body["forecast_id"]


class Ledger:
    """An append-only, hash-chained JSONL file.

    JSONL rather than a database because the ledger is meant to be published, diffed and replayed by
    people who do not have our infrastructure. A buyer should be able to download it, run the scorer
    themselves, and get our numbers.
    """

    def __init__(self, path: str | Path, signer: Signer):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.signer = signer
        self._lock = threading.Lock()

    # ── reading ──────────────────────────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[Entry]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield Entry(**json.loads(line))

    def head(self) -> str:
        """Hash of the last entry, or the genesis constant when the ledger is empty."""
        last = GENESIS
        for entry in self:
            last = entry.hash
        return last

    def count(self) -> int:
        return sum(1 for _ in self)

    def issued(self) -> dict[str, Entry]:
        """Every issued forecast, keyed by id."""
        return {e.forecast_id: e for e in self if e.kind == "issued"}

    def scored_ids(self) -> set[str]:
        return {e.forecast_id for e in self if e.kind == "scored"}

    def pending(self, now: str | None = None) -> list[Entry]:
        """Issued forecasts whose window has closed but which have not been scored yet.

        This is what the scoring job consumes. It is derived from the ledger rather than from a
        queue, so a crashed or restarted scorer cannot lose work — the ledger is the queue.
        """
        now = now or utcnow()
        settled = self.scored_ids() | self.voided_ids()
        return [e for e in self.issued().values()
                if e.forecast_id not in settled and e.body["window"]["close"] <= now]

    # ── writing ──────────────────────────────────────────────────────────────────────────────────

    def append(self, kind: str, body: dict) -> Entry:
        """Add one entry and return it. The only way to write."""
        with self._lock:
            prev = self.head()
            seq = self.count()
            at = utcnow()
            # The hash covers seq and timestamp as well as the body, so an entry cannot be lifted
            # out of one position in the chain and dropped into another.
            entry_hash = digest({"seq": seq, "kind": kind, "at": at, "body": body, "prev": prev})
            entry = Entry(seq=seq, kind=kind, at=at, body=body, prev=prev, hash=entry_hash)
            line = canonical({"seq": seq, "kind": kind, "at": at, "body": body,
                              "prev": prev, "hash": entry_hash})
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())      # a forecast that is not on disk was never made
            return entry

    def record_forecast(self, *, symbol: str, timeframe: str, horizon: str, model: str,
                        model_version: str, issued_for: str, window: dict,
                        distribution: dict, inputs: dict) -> Entry:
        """Write an issued forecast, signed.

        ``window`` states when the prediction is about — open and close, both UTC. ``distribution``
        carries the full fan, not a summary, because the commitment must be to exactly what was
        predicted or the reveal proves nothing.
        """
        signed_fields = {
            "symbol": symbol,
            "timeframe": timeframe,
            "horizon": horizon,
            "model": model,
            "model_version": model_version,
            "issued_for": issued_for,
            "window": window,
            "distribution_sha256": digest(distribution),
            "inputs_sha256": digest(inputs),
        }
        body = {
            "forecast_id": digest_hex(signed_fields)[:32],
            **signed_fields,
            "distribution": distribution,
            "inputs": inputs,
        }
        # Built from the body through the shared builder, so what gets signed is provably the same
        # object `verify` will later reconstruct.
        body["signature"] = self.signer.sign(manifest_for("issued", body))
        return self.append("issued", body)

    def record_score(self, *, forecast_id: str, actual: dict, scores: dict,
                     scorer_version: str) -> Entry:
        """Write the outcome and the scores for a forecast that has already been issued."""
        body = {"forecast_id": forecast_id, "actual_sha256": digest(actual), "actual": actual,
                "scores": scores, "scorer_version": scorer_version}
        body["signature"] = self.signer.sign(manifest_for("scored", body))
        return self.append("scored", body)

    def record_void(self, *, forecast_id: str, reason: str) -> Entry:
        """Withdraw a forecast from scoring, on the record, with the reason attached.

        The only correction mechanism there is. Nothing is ever edited or removed — a void is an
        append like everything else, it is signed like everything else, and the original forecast
        stays in the chain exactly as issued for anyone to read.

        This is deliberately expensive to use. A voided forecast is *visible* on the scorecard,
        permanently, with the stated reason next to it. An operator who voided the ones that went
        badly would be publishing a list of the forecasts they wanted to hide.

        The legitimate use is a defect in how the forecast was recorded rather than in how it turned
        out — a window that did not describe what the model was actually asked to predict, an input
        digest that does not match the candles used. Those cannot be scored meaningfully at all, and
        scoring them anyway would put noise into the record and call it performance.
        """
        body = {"forecast_id": forecast_id, "reason": reason, "voided_at": utcnow()}
        body["signature"] = self.signer.sign(manifest_for("void", body))
        return self.append("void", body)

    def voided_ids(self) -> set[str]:
        return {e.body["forecast_id"] for e in self if e.kind == "void"}

    def record_anchor(self, *, head: str, tx: str, chain: str) -> Entry:
        """Record that a chain head was committed on-chain, and where."""
        return self.append("anchor", {"forecast_id": f"anchor:{head[:16]}",
                                      "head": head, "tx": tx, "chain": chain})

    # ── integrity ────────────────────────────────────────────────────────────────────────────────

    def verify(self, expect_public_key: str | None = None) -> tuple[bool, list[str]]:
        """Recompute the whole chain and every signature, from the entries' own contents.

        Published as a service and run in CI. If this ever returns False the ledger has been edited,
        and the correct response is to say so loudly rather than repair it quietly.

        Four independent checks per entry, because each catches a forgery the others miss:

        1. **Sequence** — no entry has been inserted or removed ahead of this one.
        2. **Chain** — this entry's `prev` really is the previous entry's hash.
        3. **Hash** — the stored hash is what these contents actually hash to.
        4. **Signature over a *rebuilt* manifest** — the digest that was signed is recomputed from
           the body rather than read out of the entry, and the signing key must be the expected one.

        Check 4 is the one that took two attempts. Verifying the signature against the digest stored
        in the entry proves only that *somebody* signed *something*; a forger who edits a field and
        leaves the old digest in place passes it, and a forger with their own key passes it too.
        Rebuilding the manifest and pinning the key closes both.
        """
        problems: list[str] = []
        expected_key = expect_public_key or self.signer.public_key
        prev = GENESIS
        for i, entry in enumerate(self):
            if entry.seq != i:
                problems.append(f"entry {i}: seq is {entry.seq}, expected {i}")
            if entry.prev != prev:
                problems.append(f"entry {i}: prev {entry.prev[:20]} does not chain to {prev[:20]}")
            recomputed = digest({"seq": entry.seq, "kind": entry.kind, "at": entry.at,
                                 "body": entry.body, "prev": entry.prev})
            if recomputed != entry.hash:
                problems.append(f"entry {i}: hash does not match its contents")

            expects_signature = entry.kind in _MANIFEST_FIELDS
            sig = entry.body.get("signature")
            if expects_signature and not sig:
                problems.append(f"entry {i}: a {entry.kind} entry must carry a signature")
            elif sig:
                rebuilt = manifest_for(entry.kind, entry.body)
                if rebuilt is None:
                    problems.append(f"entry {i}: kind {entry.kind!r} carries a signature but has no "
                                    f"defined manifest, so nothing can be checked against it")
                elif digest(rebuilt) != sig.get("manifest_sha256"):
                    problems.append(f"entry {i}: signature does not verify — the manifest rebuilt "
                                    f"from this entry's own fields is not the one that was signed")
                elif sig.get("public_key") != expected_key:
                    problems.append(f"entry {i}: signature does not verify — signed by "
                                    f"{str(sig.get('public_key'))[:16]}…, expected "
                                    f"{expected_key[:16]}…")
                elif not Signer.verify(sig["manifest_sha256"], sig["signature"],
                                       sig["public_key"]):
                    problems.append(f"entry {i}: signature does not verify against its manifest")
            prev = entry.hash
        return (not problems), problems
