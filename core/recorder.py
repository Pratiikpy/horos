"""The recorder: issue forecasts on a clock, score them when their windows close, anchor the head.

This is the machine that makes the ledger fill itself. Nothing here is triggered by a paid call —
that is the point. If forecasts were only issued when someone bought one, the record would consist
entirely of the questions people happened to ask, and a bad stretch could be made to disappear by
selling nothing during it. The scored record is produced on a schedule, for a fixed set of markets,
whether anyone is buying or not.

Three jobs, deliberately independent so that one failing does not stop the others:

``issue``  — for every scored market and horizon whose window is not already covered, produce a
             forecast and write it to the ledger. Runs on the candle boundary.
``score``  — for every issued forecast whose window has now closed, fetch what happened and append
             the score. Idempotent: it derives its work from the ledger, so a crash mid-run costs
             nothing and a re-run scores nothing twice.
``anchor`` — commit the current chain head to X Layer.

The ordering constraint that matters: **anchor before the window closes.** A forecast whose hash
reached a block only after its outcome was public proves nothing at all, so the anchor job runs far
more often than the longest horizon and the scorecard reports, per forecast, whether its anchor beat
its window. Any that did not are shown as unanchored rather than quietly counted as proven.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Callable, Iterable

from .commit import AnchorError, Anchorer
from .ledger import Entry, Ledger, utcnow
from .market_data import MarketData, MarketDataError
from .markets import HORIZONS, SCORED_MARKETS, SCORED_TIMEFRAME, window_for_anchor
from .scoring import SCORER_VERSION, empirical_return_samples, score_forecast

log = logging.getLogger("horos.recorder")

# How many past returns feed the empirical baseline. 500 hourly observations is ~21 days — the same
# span the model sees, so neither side of the comparison has more history than the other.
BASELINE_SAMPLE_COUNT = 500


class Recorder:
    """Owns the ledger and the three jobs that fill it."""

    def __init__(self, ledger: Ledger, data: MarketData,
                 forecaster: Callable[..., dict] | None = None,
                 anchorer: Anchorer | None = None):
        self.ledger = ledger
        self.data = data
        # Injected rather than imported, so the recorder can be tested against a known distribution
        # and so a model outage cannot take the scoring job down with it.
        self.forecaster = forecaster
        self.anchorer = anchorer

    # ── issuing ──────────────────────────────────────────────────────────────────────────────

    def already_covers(self, symbol: str, horizon: str, window_close: str) -> bool:
        """Is there a *live* forecast for this exact window already.

        Voided forecasts do not count. A withdrawn forecast is one that cannot be scored — that is
        the whole reason it was withdrawn — so treating it as coverage leaves the window permanently
        unforecast. That is exactly what happened after the window-alignment correction: twelve
        voided entries silently blocked every replacement round, and the recorder reported
        'already_issued' while the record stood still.
        """
        voided = self.ledger.voided_ids()
        return any(e.body.get("symbol") == symbol
                   and e.body.get("horizon") == horizon
                   and (e.body.get("window") or {}).get("close") == window_close
                   and e.forecast_id not in voided
                   for e in self.ledger if e.kind == "issued")

    def issue_round(self, now: datetime | None = None) -> list[dict]:
        """Issue every forecast due at this moment. Returns one result row per attempt."""
        if self.forecaster is None:
            raise RuntimeError("no forecaster is configured; the recorder cannot issue")
        results: list[dict] = []
        for market in SCORED_MARKETS:
            for horizon in sorted(HORIZONS, key=lambda h: HORIZONS[h]):
                try:
                    forecast = self.forecaster(symbol=market.symbol, timeframe=SCORED_TIMEFRAME,
                                               horizon=horizon)
                    # The window comes out of the forecast's own anchor candle, so it describes what
                    # the model actually predicted rather than when the job happened to run.
                    window = window_for_anchor(int(forecast["inputs"]["last_candle_ts"]), horizon,
                                               SCORED_TIMEFRAME)
                    if self.already_covers(market.symbol, horizon, window["close"]):
                        results.append({"symbol": market.symbol, "horizon": horizon,
                                        "status": "already_issued", "window": window["close"]})
                        continue
                    entry = self.issue_one(market.symbol, horizon, window, forecast)
                    results.append({"symbol": market.symbol, "horizon": horizon,
                                    "status": "issued", "forecast_id": entry.forecast_id,
                                    "window": window["close"]})
                except (MarketDataError, RuntimeError, ValueError) as e:
                    # An issuing failure is recorded and moved past. Refusing to issue is always
                    # better than issuing something the model did not actually produce.
                    log.warning("issue failed for %s %s: %s", market.symbol, horizon, e)
                    results.append({"symbol": market.symbol, "horizon": horizon,
                                    "status": "failed", "error": str(e)})
        return results

    def issue_one(self, symbol: str, horizon: str, window: dict,
                  forecast: dict | None = None) -> Entry:
        """Commit one forecast to the ledger."""
        if forecast is None:
            forecast = self.forecaster(symbol=symbol, timeframe=SCORED_TIMEFRAME, horizon=horizon)
        return self.ledger.record_forecast(
            symbol=symbol,
            timeframe=SCORED_TIMEFRAME,
            horizon=horizon,
            model=forecast["model"],
            model_version=forecast["model_version"],
            issued_for=window["close"],
            window=window,
            distribution=forecast["distribution"],
            inputs=forecast["inputs"],
        )

    # ── scoring ──────────────────────────────────────────────────────────────────────────────

    def score_round(self, now: str | None = None, limit: int | None = None) -> list[dict]:
        """Score every forecast whose window has closed and which is not yet scored."""
        pending = self.ledger.pending(now)
        pending.sort(key=lambda e: e.body["window"]["close"])
        if limit:
            pending = pending[:limit]
        results: list[dict] = []
        for entry in pending:
            try:
                results.append(self.score_one(entry))
            except MarketDataError as e:
                # The outcome is not available yet, or the exchange is down. Leave the forecast
                # pending; it will be picked up on the next run. Never write a score derived from
                # data we could not fetch.
                log.warning("cannot score %s yet: %s", entry.forecast_id, e)
                results.append({"forecast_id": entry.forecast_id, "status": "deferred",
                                "reason": str(e)})
        return results

    def score_one(self, entry: Entry) -> dict:
        """Fetch the outcome for one issued forecast, score it, append the result."""
        body = entry.body
        symbol, timeframe = body["symbol"], body["timeframe"]
        window = body["window"]

        actual = self.data.window_outcome(symbol, timeframe, window["open"], window["close"])

        distribution = body.get("distribution") or {}
        anchor = distribution.get("anchor_price")

        # The baseline is rebuilt from the candles that existed *at issue time*, not from today's, so
        # it is exactly the forecast a caller could have made for free at the moment we charged them.
        baseline: list[float] | None = None
        if anchor:
            try:
                horizon_steps = HORIZONS.get(body["horizon"], 1)
                issued_ms = _iso_to_ms(window["open"])
                history = self.data.candles(symbol, timeframe,
                                            limit=BASELINE_SAMPLE_COUNT + horizon_steps,
                                            until_ms=issued_ms)
                returns = history.log_returns(step=horizon_steps)
                if returns:
                    baseline = empirical_return_samples(float(anchor), returns,
                                                        count=BASELINE_SAMPLE_COUNT)
            except MarketDataError as e:
                log.info("no baseline for %s: %s", entry.forecast_id, e)

        scores = score_forecast(
            actual_close=float(actual["close"]),
            actual_high=float(actual["high"]),
            actual_low=float(actual["low"]),
            distribution=distribution,
            baseline_samples=baseline,
            persistence=float(anchor) if anchor else None,
        )
        if baseline is None and anchor:
            scores.setdefault("not_scored", []).append(
                "crps_skill_vs_empirical: the baseline history could not be rebuilt for this window")

        written = self.ledger.record_score(forecast_id=entry.forecast_id, actual=actual,
                                           scores=scores, scorer_version=SCORER_VERSION)
        return {"forecast_id": entry.forecast_id, "status": "scored",
                "close": actual["close"], "crps": scores.get("crps"), "seq": written.seq}

    # ── anchoring ────────────────────────────────────────────────────────────────────────────

    def anchor_now(self) -> dict:
        """Commit the current head. Reports why not, when not."""
        if self.anchorer is None:
            return {"status": "disabled",
                    "reason": "no anchoring key is configured; the ledger still hash-chains but is "
                              "not yet committed on X Layer"}
        head = self.ledger.head()
        entries = self.ledger.count()
        if entries == 0:
            return {"status": "skipped", "reason": "the ledger is empty"}
        if self.last_anchor_head() == head:
            return {"status": "skipped", "reason": "this head is already anchored",
                    "head": head}
        try:
            anchor = self.anchorer.anchor(head, entries)
        except AnchorError as e:
            log.error("anchoring failed: %s", e)
            return {"status": "failed", "reason": str(e), "head": head}
        self.ledger.record_anchor(head=anchor.head, tx=anchor.tx, chain=anchor.chain)
        return {"status": "anchored", "head": head, "entries": entries, "tx": anchor.tx,
                "block": anchor.block, "explorer": anchor.explorer_url}

    def last_anchor_head(self) -> str | None:
        last = None
        for e in self.ledger:
            if e.kind == "anchor":
                last = e.body.get("head")
        return last

    def anchors(self) -> list[dict]:
        return [{"at": e.at, **e.body} for e in self.ledger if e.kind == "anchor"]

    def anchor_state(self) -> dict:
        """Everything the scorecard needs to say, truthfully, how anchored the record is.

        ``proven`` counts only the forecasts whose anchor was mined *before* their window closed —
        the only ones for which the on-chain commitment actually rules out backdating. The rest are
        reported separately rather than folded in.
        """
        anchors = self.anchors()
        issued = [e for e in self.ledger if e.kind == "issued"]
        if not anchors:
            return {"anchored": False, "anchor_count": 0, "issued": len(issued),
                    "proven_before_outcome": 0,
                    "note": "no ledger head has been committed on X Layer yet. Until one is, the "
                            "chain is tamper-evident to anyone holding a copy but not provably "
                            "prior to its own outcomes."}
        # An entry is covered by an anchor if that anchor came after it in the chain.
        seq_of_anchor: list[tuple[int, str, str]] = []   # (seq, at, tx)
        for e in self.ledger:
            if e.kind == "anchor":
                seq_of_anchor.append((e.seq, e.at, e.body.get("tx", "")))
        proven = 0
        for e in issued:
            close = (e.body.get("window") or {}).get("close", "")
            first = next((a for a in seq_of_anchor if a[0] > e.seq), None)
            if first and close and first[1] < close:
                proven += 1
        return {"anchored": True, "anchor_count": len(anchors), "issued": len(issued),
                "proven_before_outcome": proven,
                "unproven": len(issued) - proven,
                "latest": anchors[-1],
                "note": ("`proven_before_outcome` counts forecasts whose ledger head reached an "
                         "X Layer block before their window closed. Only those are provably not "
                         "backdated." if proven < len(issued) else
                         "every issued forecast was anchored before its window closed")}

    # ── the loop ─────────────────────────────────────────────────────────────────────────────

    def run_forever(self, timeframe_seconds: int = 3600, score_every: int = 300,
                    anchor_every: int = 900, stop: Callable[[], bool] | None = None,
                    issue_offset: int = 30) -> None:
        """The daemon.

        Issuing is aligned to the candle clock rather than run on a fixed interval, so a round always
        starts just after a bar closes and works from the freshest possible anchor. A plain
        `sleep(3600)` would drift a little further into each hour every cycle until forecasts were
        being made from an anchor most of an hour old.

        **Anchoring runs immediately after every issue round**, not only on its own timer. The whole
        integrity claim is that a forecast's hash reached a block before its window closed, and the
        shortest window is one hour — so waiting up to `anchor_every` for the next scheduled anchor
        would leave the tightest forecasts with the least margin. Anchoring costs about 0.0000005 OKB;
        there is no reason to be sparing with it.

        Each job fails independently. A model outage must not stop scoring, and a chain outage must
        not stop either.
        """
        next_score = next_anchor = 0.0
        next_issue = self._next_boundary(timeframe_seconds, issue_offset)
        log.info("recorder started; first issue round at %s",
                 datetime.fromtimestamp(next_issue, timezone.utc).strftime("%H:%M:%SZ"))
        while not (stop and stop()):
            now = time.time()
            if now >= next_issue:
                next_issue = self._next_boundary(timeframe_seconds, issue_offset)
                if self.forecaster is not None:
                    try:
                        rows = self.issue_round()
                        log.info("issue round: %s", _tally(rows))
                        if any(r["status"] == "issued" for r in rows):
                            log.info("anchor after issue: %s", self.anchor_now())
                            next_anchor = time.time() + anchor_every
                    except Exception as e:                                # noqa: BLE001
                        log.exception("issue round crashed: %s", e)
            if now >= next_score:
                next_score = now + score_every
                try:
                    rows = self.score_round()
                    if rows:
                        log.info("score round: %s", _tally(rows))
                except Exception as e:                                    # noqa: BLE001
                    log.exception("score round crashed: %s", e)
            if now >= next_anchor:
                next_anchor = now + anchor_every
                try:
                    result = self.anchor_now()
                    if result.get("status") != "skipped":
                        log.info("anchor: %s", result)
                except Exception as e:                                    # noqa: BLE001
                    log.exception("anchor crashed: %s", e)
            time.sleep(min(30.0, max(1.0, min(next_issue, next_score, next_anchor) - time.time())))

    @staticmethod
    def _next_boundary(period: int, offset: int) -> float:
        """The next candle close, plus a small offset so the exchange has published the bar."""
        now = time.time()
        return ((now // period) + 1) * period + offset


def _tally(rows: Iterable[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("status", "?")] = counts.get(r.get("status", "?"), 0) + 1
    return counts


def _iso_to_ms(s: str) -> int:
    return int(datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


__all__ = ["Recorder", "BASELINE_SAMPLE_COUNT", "utcnow"]
