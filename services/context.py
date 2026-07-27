"""The handle every service handler is given.

One object holding the data layer, the model, the ledger, the calibrator and a way to call sibling
services. Constructed once at startup so a paid request never pays for a 409 MB checkpoint read or an
exchange handshake.

The two ledger lookups here are what make the committed-first rule work, and both are deliberately
strict about *time*. `latest_committed` will not return a forecast whose window has already closed —
serving one would be selling a prediction about the past. `most_recent_scored` returns the opposite:
the forecast that covered a window which has now closed, which is the only thing a surprise reading
can honestly be measured against.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from core.conformal import Calibrator
from core.crypto import Signer
from core.ledger import Ledger, utcnow
from core.market_data import MarketData


class Context:
    def __init__(self, data: MarketData, runner, forecaster, ledger: Ledger, signer: Signer,
                 calibrator: Calibrator | None = None, registry=None):
        self.data = data
        self.runner = runner
        self.forecaster = forecaster
        self.ledger = ledger
        self.signer = signer
        self.calibrator = calibrator or Calibrator()
        self.registry = registry
        self._registry = registry
        self._index: dict[str, Any] = {}
        self._index_at = 0.0
        self._lock = threading.Lock()

    # ── ledger index ─────────────────────────────────────────────────────────────────────────

    def _fresh_index(self) -> dict:
        """A read-through index of the ledger, rebuilt at most every few seconds.

        The ledger is a file and grows by twelve entries an hour; re-reading it for every field of
        every request would be the dominant cost of a paid call for no benefit.
        """
        with self._lock:
            if self._index and (time.time() - self._index_at) < 5.0:
                return self._index
            issued: dict[str, dict] = {}
            by_key: dict[tuple[str, str], list[dict]] = {}
            anchors: list[dict] = []
            scored: set[str] = set()
            voided: set[str] = set()
            for e in self.ledger:
                if e.kind == "issued":
                    row = {"forecast_id": e.forecast_id, "body": e.body, "issued_at": e.at,
                           "window": e.body.get("window"), "seq": e.seq}
                    issued[e.forecast_id] = row
                    by_key.setdefault((e.body["symbol"], e.body["horizon"]), []).append(row)
                elif e.kind == "scored":
                    scored.add(e.forecast_id)
                elif e.kind == "void":
                    voided.add(e.body["forecast_id"])
                elif e.kind == "anchor":
                    anchors.append({"seq": e.seq, "at": e.at, **e.body})
            # Attach the first anchor that came after each forecast — the evidence that its hash was
            # on chain, and when.
            for row in issued.values():
                nxt = next((a for a in anchors if a["seq"] > row["seq"]), None)
                if nxt:
                    row["anchor"] = {"tx": nxt.get("tx"), "at": nxt["at"], "chain": nxt.get("chain"),
                                     "before_window_close":
                                         bool(row.get("window")
                                              and nxt["at"] < row["window"]["close"])}
            self._index = {"issued": issued, "by_key": by_key, "anchors": anchors,
                           "scored": scored, "voided": voided}
            self._index_at = time.time()
            return self._index

    def invalidate(self) -> None:
        with self._lock:
            self._index = {}

    def latest_committed(self, symbol: str, horizon: str) -> dict | None:
        """The most recent committed forecast whose window has **not** yet closed.

        Returning one whose window had closed would be selling a prediction about something that has
        already happened, dressed as a forecast. The freshness rule is the product, not a nicety.
        """
        idx = self._fresh_index()
        now = utcnow()
        rows = idx["by_key"].get((symbol, horizon), [])
        live = [r for r in rows
                if r["forecast_id"] not in idx["voided"]
                and r.get("window") and r["window"]["close"] > now]
        return max(live, key=lambda r: r["seq"]) if live else None

    def most_recent_scored(self, symbol: str, horizon: str) -> dict | None:
        """The most recent committed forecast whose window has now closed.

        The only honest basis for a surprise reading: a distribution that was fixed and hashed before
        the candle it is being compared against existed.
        """
        idx = self._fresh_index()
        now = utcnow()
        rows = [r for r in idx["by_key"].get((symbol, horizon), [])
                if r["forecast_id"] not in idx["voided"]
                and r.get("window") and r["window"]["close"] <= now]
        return max(rows, key=lambda r: r["seq"]) if rows else None

    def anchors(self) -> list[dict]:
        return self._fresh_index()["anchors"]

    def counts(self) -> dict:
        idx = self._fresh_index()
        return {"issued": len(idx["issued"]), "scored": len(idx["scored"]),
                "voided": len(idx["voided"]), "anchors": len(idx["anchors"]),
                "entries": self.ledger.count()}

    # ── sibling calls ────────────────────────────────────────────────────────────────────────

    def registry_call(self, endpoint: str, payload: dict) -> dict:
        """Run another service's handler in-process.

        Used where one service genuinely builds on another — `market.dislocation` needs the
        cross-venue comparison `market.venues` already performs. Calling the handler rather than
        duplicating the logic means the two can never disagree, and the caller is charged once.
        """
        if self._registry is None:
            raise RuntimeError("no registry is attached to this context")
        svc = self._registry.get(endpoint)
        if svc is None:
            raise RuntimeError(f"unknown internal service {endpoint!r}")
        return svc.handler(payload, self)
