"""Group G — the record. Eight services, $0.001 to $0.02.

Deliberately the cheapest tier — most of it sits at a tenth of a cent. The forecasts are the
product; this is the reason to believe them, so it is priced as close to nothing as the
marketplace allows.

**The same evidence is genuinely free on the web.** /scorecard, /ledger and /ledger/verify need
no payment and no account. The marketplace copies carry a nominal fee only because a zero-fee
x402 gate breaks the marketplace's own task-402-pay flow with an amount mismatch (OKX builder
chat, 2026-07-25). Listing at zero would advertise a purchase path that does not work, which is
worse than charging a tenth of a cent.

The move that matters is `commit` and `judge`: the ledger is not restricted to our own forecasts.
Anyone can commit a hash of theirs before their window closes and have it scored by the same code,
against the same data, and published in the same leaderboard — including when they beat us. A
benchmark other people participate in is a moat that compounds and that a later entrant cannot
fabricate, because the record is timestamped forward.

`scorecard` will show bad periods. That is the product working, not a bug in it.
"""
from __future__ import annotations

import re

from core.conformal import Calibrator
from core.markets import HORIZONS, SCORED_MARKETS, coverage_note
from core.scoring import SCORER_VERSION, aggregate

from .registry import Param, ServiceError, service

HEX64 = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")


def _scored_rows(ctx, symbol: str | None = None, horizon: str | None = None,
                 model: str | None = None) -> list[dict]:
    """Every scored forecast, joined back to what was issued. Voided entries are excluded."""
    issued: dict[str, dict] = {}
    voided: set[str] = set()
    rows: list[dict] = []
    for e in ctx.ledger:
        if e.kind == "issued":
            issued[e.forecast_id] = e.body
        elif e.kind == "void":
            voided.add(e.body["forecast_id"])
    for e in ctx.ledger:
        if e.kind != "scored" or e.forecast_id in voided:
            continue
        body = issued.get(e.forecast_id)
        if not body:
            continue
        if symbol and body.get("symbol") != symbol:
            continue
        if horizon and body.get("horizon") != horizon:
            continue
        if model and body.get("model") != model:
            continue
        rows.append({"forecast_id": e.forecast_id, "issued": body, "scored": e.body,
                     "scores": e.body.get("scores", {})})
    return rows


def _empty_note(ctx, symbol: str | None = None, horizon: str | None = None) -> dict:
    """What to return when the filtered record is empty.

    Two things this must never do, both of which it used to. It must not say "nothing has been
    scored" when the answer is only "nothing matching *your* filter" — with one symbol graded and
    another not, that sentence is simply false to the caller who asked about the second. And it must
    not answer a paid call with prose about how to verify a record that it declines to show; a buyer
    who asked for a symbol's track record and received the coverage policy has been charged for
    documentation. So the empty case returns the forecasts that *are* committed for that filter —
    ids, digests, the anchor that put each on chain, and the timestamp each one grades at.
    """
    scored_total = len(_scored_rows(ctx))
    pending = ctx.pending(symbol, horizon)
    asked = {k: v for k, v in (("symbol", symbol), ("horizon", horizon)) if v}

    if scored_total == 0:
        why = ("No forecast has been scored yet, for any symbol. The ledger is live and forecasts "
               "are committed on X Layer, but no window has closed and been graded.")
    else:
        where = " and ".join(f"{k}={v}" for k, v in asked.items()) or "this filter"
        why = (f"{scored_total} forecast(s) have been scored, but none match {where}. The record "
               f"exists; it does not yet cover what you asked for.")

    available: dict[str, list] = {}
    if scored_total:
        every = _scored_rows(ctx)
        available = {
            "symbols": sorted({r["issued"].get("symbol") for r in every if r["issued"].get("symbol")}),
            "horizons": sorted({r["issued"].get("horizon") for r in every if r["issued"].get("horizon")}),
        }

    out = {
        "scored_forecasts": 0,
        "filter": asked or None,
        "note": why + (
            " Horos publishes empty rather than a backtest — Kronos was pre-trained across 45 "
            "exchanges to an unpublished cutoff, so any historical evaluation risks scoring it on "
            "its own training data. The record starts at first commit and accumulates forward."),
        # The substance of the answer when there are no scores yet. Each row was hashed and signed
        # before its window opened, so it can be checked now and graded later against this same id.
        "pending_forecasts": pending,
        "pending_count": len(pending),
        "next_grades_at": pending[0]["grades_after"] if pending else None,
        "ledger": ctx.counts(),
    }
    if available:
        out["record_available_for"] = available
    if not pending:
        out["pending_note"] = ("nothing is currently committed for this filter either. Coverage is "
                               "listed under 'coverage' — a symbol outside it is never forecast.")
    return out


# ── the record ───────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="scorecard", group="G. Accountability", price=0.001,
    title="The running public accuracy record",
    short="The public accuracy record",
    summary="Publishes how every forecast Horos has issued actually turned out — the proper scores, "
            "the empirical coverage of each stated band, and the skill against a free baseline — "
            "generated from the ledger rather than written by hand.",
    returns="Returns the aggregate scores overall and broken down by symbol and horizon, the "
            "coverage each band claimed against what it delivered, the calibration histogram, and "
            "the on-chain anchors backing the record.",
    depth="The identical record is free and unauthenticated at /scorecard; this is the machine-readable "
          "copy at the marketplace minimum price. Every figure is derived from the "
          "hash-chained ledger, so anyone can download it, run the same scorer and reproduce these "
          "numbers exactly. The skill scores are against an empirical random walk a caller gets for "
          "nothing, so a negative number here means the model is not worth paying for on that "
          "horizon — and it will say so.",
    params=(Param("symbol", "string", "Restrict to one symbol.", default=None),
            Param("horizon", "string", "Restrict to one horizon.", default=None)),
    example={"symbol": "BTC/USDT:USDT"},
)
def scorecard(inp: dict, ctx) -> dict:
    rows = _scored_rows(ctx, inp.get("symbol"), inp.get("horizon"))
    counts = ctx.counts()
    anchors = ctx.anchors()

    base = {
        "service": "Horos",
        "coverage": coverage_note(),
        "ledger": counts,
        "scorer_version": SCORER_VERSION,
        "anchoring": {
            "anchors": len(anchors),
            "latest": anchors[-1] if anchors else None,
            "chain": "eip155:196 (X Layer mainnet)",
            "how_to_verify": ("each anchor's transaction calldata is 46 bytes: b'HOROS', a version "
                              "byte, the 32-byte ledger head, and the entry count. Pull the input "
                              "field from any X Layer explorer and decode it."),
        },
        "reproduce": ("download the ledger, recompute every score with core/scoring.py, and compare. "
                      "Nothing here depends on trusting Horos."),
    }
    if not rows:
        return {**base, **_empty_note(ctx, inp.get("symbol"), inp.get("horizon"))}

    overall = aggregate([r["scores"] for r in rows])
    by_symbol: dict[str, dict] = {}
    by_horizon: dict[str, dict] = {}
    for m in SCORED_MARKETS:
        subset = [r["scores"] for r in rows if r["issued"].get("symbol") == m.symbol]
        if subset:
            by_symbol[m.symbol] = aggregate(subset)
    for h in HORIZONS:
        subset = [r["scores"] for r in rows if r["issued"].get("horizon") == h]
        if subset:
            by_horizon[h] = aggregate(subset)

    return {
        **base,
        "scored_forecasts": len(rows),
        "overall": overall,
        "by_symbol": by_symbol,
        "by_horizon": by_horizon,
        "how_to_read": {
            "crps": "lower is better. It is in price units, so it is not comparable across symbols.",
            "crps_skill_vs_empirical": ("the number that matters. Above 0 the model beats a random "
                                        "walk drawn from recent returns; at or below 0 it does not, "
                                        "and on that horizon you should not pay for it."),
            "coverage": ("`empirical` against `nominal`. A calibrated 0.9 band lands near 0.90. "
                         "Below it the bands are too narrow; above it they are too wide."),
            "pit": ("a flat histogram means calibrated. A U shape means over-confident, a hump in "
                    "the middle means over-cautious."),
        },
        "honesty": ("this page shows losses as readily as wins, because the scoring is automatic and "
                    "the commitments are on chain. We could not hide a miss if we wanted to."),
    }


@service(
    endpoint="scorecard.regime", group="G. Accountability", price=0.001,
    title="Accuracy broken down by market regime",
    short="Accuracy by market regime",
    summary="Splits the accuracy record by the volatility regime each forecast was made in, because "
            "a single average hides the thing a caller most needs to know — when to believe the "
            "model and when not to.",
    returns="Returns the scores separated into calm, normal and volatile regimes, the number of "
            "forecasts in each, and where the model's advantage over the baseline actually comes "
            "from.",
    depth="Regime is assigned from the volatility of the context window at issue time, so it is "
          "fixed before the outcome is known and cannot be chosen afterwards to flatter a bucket. "
          "Forecasting models typically do well in calm conditions and badly in violent ones; if "
          "that is true here, this is where it shows.",
    params=(Param("symbol", "string", "Restrict to one symbol.", default=None),),
    example={"symbol": "BTC/USDT:USDT"},
)
def scorecard_regime(inp: dict, ctx) -> dict:
    rows = _scored_rows(ctx, inp.get("symbol"))
    if not rows:
        return {**_empty_note(ctx, inp.get("symbol")),
                "note_regime": "regime breakdown needs scored forecasts to break down."}

    # Regime from the dispersion the model itself implied at issue time — known before the outcome.
    spreads = []
    for r in rows:
        d = r["issued"].get("distribution") or {}
        sd = d.get("per_candle_log_return_sd")
        spreads.append(float(sd) if sd is not None else None)
    known = sorted(s for s in spreads if s is not None)
    if len(known) < 3:
        return {"scored_forecasts": len(rows),
                "note": ("too few forecasts carry a dispersion measure to split into regimes. "
                         "Buckets appear once the record is deeper."),
                "ledger": ctx.counts()}

    low, high = known[len(known) // 3], known[2 * len(known) // 3]
    buckets: dict[str, list] = {"calm": [], "normal": [], "volatile": []}
    for r, s in zip(rows, spreads):
        if s is None:
            continue
        name = "calm" if s <= low else "volatile" if s > high else "normal"
        buckets[name].append(r["scores"])

    return {
        "scored_forecasts": len(rows),
        "regime_thresholds": {"calm_below": low, "volatile_above": high,
                              "measure": "per-candle log-return standard deviation implied by the "
                                         "forecast distribution at issue time"},
        "by_regime": {k: aggregate(v) for k, v in buckets.items() if v},
        "assignment_note": ("regime is fixed from information available when the forecast was made, "
                            "so no bucket can be chosen after the outcome to make it look better"),
        "how_to_read": ("compare crps_skill_vs_empirical across the three. Where it is negative, the "
                        "model is not beating a free baseline in those conditions and Horos says so "
                        "rather than averaging it away."),
        "ledger": ctx.counts(),
    }


@service(
    endpoint="scorecard.model", group="G. Accountability", price=0.001,
    title="Accuracy per model and per checkpoint",
    short="Accuracy per model version",
    summary="Reports the record separately for every model and checkpoint that has ever issued a "
            "forecast, so a change of model starts a visibly new record instead of quietly "
            "inheriting the old one's reputation.",
    returns="Returns each model version with the forecasts it issued, its scores, the window it was "
            "active over, and whether it improved on its predecessor.",
    depth="Every forecast records the SHA-256 of the weights that produced it, not just a model "
          "name. Upgrading the checkpoint therefore cannot be hidden inside an existing track "
          "record — the new weights appear as a new row starting from zero, which is the only "
          "honest way to publish a model change.",
    params=(),
    example={},
)
def scorecard_model(inp: dict, ctx) -> dict:
    rows = _scored_rows(ctx)
    if not rows:
        return {**_empty_note(ctx),
                "note_model": "per-model breakdown needs scored forecasts."}
    by_version: dict[str, list] = {}
    meta: dict[str, dict] = {}
    for r in rows:
        v = r["issued"].get("model_version", "unknown")
        by_version.setdefault(v, []).append(r["scores"])
        m = meta.setdefault(v, {"model": r["issued"].get("model"), "first": None, "last": None})
        w = (r["issued"].get("window") or {}).get("close")
        if w:
            m["first"] = min(m["first"], w) if m["first"] else w
            m["last"] = max(m["last"], w) if m["last"] else w
    return {
        "scored_forecasts": len(rows),
        "models": {v: {**meta[v], "scores": aggregate(s)} for v, s in by_version.items()},
        "versioning_note": ("a model version is 'repo@sha256-prefix-of-the-weights'. Changing the "
                            "checkpoint starts a new record; it does not inherit the old one."),
        "ledger": ctx.counts(),
    }


@service(
    endpoint="forecast.reliability", group="C. Forecast", price=0.02,
    title="At which horizons this model is worth believing",
    short="Where the model is useful",
    summary="Reads the accuracy record and answers the question that decides whether to use a "
            "forecast at all — which symbols and horizons the model has actually beaten a free "
            "baseline on, and by how much.",
    returns="Returns per horizon the skill against the baseline, the empirical coverage of each "
            "band, the number of scored observations behind the figure, and an explicit "
            "recommendation on whether to rely on it.",
    depth="This is the service most likely to tell a buyer not to buy the others, which is exactly "
          "why it exists. Its verdicts come from the same ledger the scorecard publishes, so they "
          "cannot be more optimistic than the public record. Where the sample is too small to "
          "conclude anything, it says that rather than reporting a number.",
    params=(Param("symbol", "string", "Symbol to assess.", default=None),),
    example={"symbol": "BTC/USDT:USDT"},
)
def forecast_reliability(inp: dict, ctx) -> dict:
    symbol = inp.get("symbol")
    rows = _scored_rows(ctx, symbol)
    if not rows:
        empty = _empty_note(ctx, symbol)
        scored_elsewhere = empty.get("record_available_for") or {}
        return {
            "symbol": symbol,
            "verdict": "no_record_yet",
            # Says "for this symbol", not "at all" — the previous wording claimed nothing had ever
            # been scored even when another symbol's record was already published.
            "recommendation": (
                f"There is no scored record for {symbol or 'any symbol'} yet, so no claim is made "
                f"either way about reliability here. Horos publishes no backtest because the "
                f"model's pre-training contaminates one. Treat the forecast bands as the model's "
                f"own claim until the record fills."
                + (f" Scored coverage does exist for: {', '.join(scored_elsewhere.get('symbols', []))}."
                   if scored_elsewhere.get("symbols") else "")
                + (f" {empty['pending_count']} forecast(s) are already committed and hashed for this "
                   f"filter; the first grades at {empty['next_grades_at']}."
                   if empty.get("pending_count") else "")),
            "pending_forecasts": empty["pending_forecasts"],
            "pending_count": empty["pending_count"],
            "next_grades_at": empty["next_grades_at"],
            "record_available_for": scored_elsewhere or None,
            "free_alternative": "the same record is free and unauthenticated at /scorecard",
            "ledger": ctx.counts(),
        }

    out: dict[str, dict] = {}
    for h in HORIZONS:
        subset = [r["scores"] for r in rows if r["issued"].get("horizon") == h]
        if not subset:
            continue
        agg = aggregate(subset)
        skill = agg.get("crps_skill_vs_empirical")
        n = agg["count"]
        # 30 is where a skill estimate stops being dominated by which particular candles landed in
        # the sample. Below it the honest answer is "not enough evidence", not a number.
        if n < 30:
            verdict, rec = "insufficient_evidence", (
                f"only {n} scored forecast(s) at this horizon. That is too few to distinguish skill "
                f"from luck; no claim is made either way.")
        elif skill is None:
            verdict, rec = "unmeasurable", "the baseline could not be reconstructed for this horizon."
        elif skill > 0.05:
            verdict, rec = "reliable", (
                f"the model beats an empirical random walk by {skill:.1%} on CRPS across {n} scored "
                f"forecasts. Worth paying for at this horizon.")
        elif skill > 0:
            verdict, rec = "marginal", (
                f"the model beats the baseline by only {skill:.1%} across {n} forecasts. The edge is "
                f"real but small; weigh it against the price.")
        else:
            verdict, rec = "not_reliable", (
                f"the model does NOT beat a free empirical random walk at this horizon "
                f"({skill:.1%} across {n} forecasts). Do not pay for a forecast here — resample "
                f"recent returns yourself and you will do at least as well.")
        out[h] = {
            "scored_forecasts": n,
            "crps_skill_vs_empirical": skill,
            "crps_skill_vs_persistence": agg.get("crps_skill_vs_persistence"),
            "coverage": agg.get("coverage"),
            "verdict": verdict,
            "recommendation": rec,
        }

    best = max((h for h in out if out[h]["verdict"] == "reliable"),
               key=lambda h: out[h]["crps_skill_vs_empirical"] or -9, default=None)
    return {
        "symbol": symbol or "all scored symbols",
        "by_horizon": out,
        "best_horizon": best,
        "summary": (f"the strongest measured horizon is {best}" if best else
                    "no horizon currently shows measured skill above the baseline"),
        "source": "derived from the same ledger the free scorecard publishes; it cannot be more "
                  "favourable than the public record",
        "ledger": ctx.counts(),
    }


# ── third parties ────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="commit", group="G. Accountability", price=0.01,
    title="Timestamp your own forecast, on chain",
    short="Timestamp your forecast",
    summary="Writes the hash of any forecast — yours, not just ours — into the same hash-chained "
            "ledger, so it is provably fixed before its window closes and can be scored afterwards "
            "by the same code that scores ours.",
    returns="Returns the commitment id, the ledger entry, the chain head it joined, and the "
            "transaction that anchors it on X Layer once the next anchor runs.",
    depth="This is what makes the ledger a benchmark rather than a marketing page. You send a "
          "digest, a symbol, a window and a label; we never see your forecast until you reveal it, "
          "and we cannot alter what you committed. Reveal before the window closes and `judge` "
          "scores it with the identical scorer, publishing the result in the leaderboard — "
          "including when it beats us.",
    params=(Param("digest", "string", "SHA-256 of your canonical forecast, 64 hex characters.",
                  required=True),
            Param("symbol", "string", "What the forecast is about.", required=True),
            Param("window_close", "string", "When the window closes, UTC 'YYYY-MM-DDTHH:MM:SSZ'.",
                  required=True),
            Param("label", "string", "Your name in the leaderboard.", required=True),
            Param("horizon", "string", "Horizon label.", default="1h"),
            Param("note", "string", "Anything you want recorded alongside it.", default=None)),
    example={"digest": "3b1f...64 hex chars...", "symbol": "BTC/USDT:USDT",
             "window_close": "2026-07-28T12:00:00Z", "label": "acme-forecaster", "horizon": "24h"},
)
def commit(inp: dict, ctx) -> dict:
    from core.ledger import utcnow

    digest = str(inp.get("digest", "")).strip()
    if not HEX64.match(digest):
        raise ServiceError(
            "digest must be a SHA-256 as 64 hex characters, optionally 0x-prefixed. Hash the "
            "canonical JSON of your forecast — sorted keys, no whitespace.", code="bad_digest")
    digest = digest[2:] if digest.startswith("0x") else digest
    symbol = str(inp.get("symbol", "")).strip()
    label = str(inp.get("label", "")).strip()
    if not symbol or not label:
        raise ServiceError("symbol and label are both required.", code="missing_field")
    if len(label) > 64:
        raise ServiceError("label must be 64 characters or fewer.", code="bad_label")
    close = str(inp.get("window_close", "")).strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", close):
        raise ServiceError("window_close must look like '2026-07-28T12:00:00Z'.",
                           code="bad_window")

    now = utcnow()
    if close <= now:
        raise ServiceError(
            f"window_close {close} is not in the future (now {now}). A commitment made after its "
            f"own window has closed proves nothing, so it is refused rather than recorded.",
            code="window_already_closed")

    entry = ctx.ledger.append("third_party_commit", {
        "forecast_id": f"tp:{digest[:24]}",
        "digest": digest, "symbol": symbol, "horizon": str(inp.get("horizon", "1h")),
        "label": label, "window_close": close, "note": inp.get("note"),
        "committed_at": now,
    })
    ctx.invalidate()
    return {
        "committed": True,
        "commitment_id": entry.body["forecast_id"],
        "digest": digest, "symbol": symbol, "window_close": close, "label": label,
        "ledger": {"seq": entry.seq, "entry_hash": entry.hash, "chain_head": ctx.ledger.head()},
        "anchoring": ("this entry joins the hash chain immediately and its head is committed to "
                      "X Layer on the next anchor, which runs at least every 15 minutes and "
                      "always right after a forecast round"),
        "next_step": ("call `judge` with the same commitment id and your revealed forecast once the "
                      "window has closed. The reveal is checked against this digest before it is "
                      "scored, so a forecast that does not hash to what you committed is rejected."),
    }


@service(
    endpoint="judge", group="G. Accountability", price=0.02,
    title="Score a committed forecast against what happened",
    short="Score a committed forecast",
    summary="Takes a forecast you committed earlier, checks it hashes to the digest you registered, "
            "fetches what the market actually did, and scores it with the identical code that scores "
            "ours.",
    returns="Returns the digest check, the realised outcome, the full score — CRPS, pinball, "
            "interval scores, coverage and Brier where applicable — and the skill against the same "
            "baseline we are measured on.",
    depth="The reveal is verified before anything is scored: your forecast is canonicalised, hashed, "
          "and compared against what you committed. A mismatch is refused outright, so nothing can "
          "be adjusted after the fact. The scorer is `core/scoring.py` unchanged — the same "
          "functions, the same fair CRPS estimator, the same baseline.",
    params=(Param("commitment_id", "string", "The id returned by `commit`.", required=True),
            Param("forecast", "object", "Your revealed forecast: samples or quantiles, plus bands.",
                  required=True)),
    example={"commitment_id": "tp:3b1f0a...", "forecast": {"samples": [65000, 65500, 66000],
                                                           "bands": {"0.9": [64000, 67000]}}},
)
def judge(inp: dict, ctx) -> dict:
    from core.crypto import digest as sha
    from core.ledger import utcnow
    from core.market_data import MarketDataError
    from core.markets import TIMEFRAME_SECONDS
    from core.scoring import empirical_return_samples, score_forecast

    cid = str(inp.get("commitment_id", "")).strip()
    revealed = inp.get("forecast")
    if not cid or not isinstance(revealed, dict):
        raise ServiceError("commitment_id and a forecast object are both required.",
                           code="missing_field")

    record = None
    for e in ctx.ledger:
        if e.kind == "third_party_commit" and e.body.get("forecast_id") == cid:
            record = e.body
    if record is None:
        raise ServiceError(f"no commitment with id {cid!r} exists in the ledger.",
                           code="unknown_commitment")

    # The reveal must hash to what was committed. This check is the whole guarantee.
    revealed_digest = sha(revealed).split(":", 1)[-1]
    if revealed_digest != record["digest"]:
        raise ServiceError(
            f"the revealed forecast does not hash to the committed digest. Committed "
            f"{record['digest'][:16]}…, revealed {revealed_digest[:16]}…. Hash the canonical JSON "
            f"of exactly the object you commit: sorted keys, separators (',',':'), no whitespace, "
            f"ensure_ascii false.", code="digest_mismatch")

    horizon = record.get("horizon", "1h")
    now = utcnow()
    if record["window_close"] > now:
        # A 200, not an error. Payment settles before this handler runs, so raising here would
        # charge the caller and hand them a 4xx — paying for nothing. The digest has been verified,
        # which is real work and the thing that cannot be redone later, so the answer is the
        # verification plus exactly when to come back.
        return {
            "commitment_id": cid, "label": record["label"], "symbol": record["symbol"],
            "horizon": horizon,
            "status": "not_yet_scoreable",
            "digest_verified": True,
            "committed_at": record["committed_at"],
            "window_close": record["window_close"],
            "now": now,
            "reason": (f"this window does not close until {record['window_close']} (it is {now}). "
                       f"Scoring early would grade a forecast against an outcome that has not "
                       f"happened yet."),
            "next_step": (f"call judge again with the same commitment_id and forecast after "
                          f"{record['window_close']}. Your reveal has already been checked against "
                          f"the committed digest and matches."),
        }

    steps = HORIZONS.get(horizon, 1)
    step_s = TIMEFRAME_SECONDS["1h"]
    from datetime import datetime, timedelta, timezone
    close_dt = datetime.strptime(record["window_close"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)
    open_iso = (close_dt - timedelta(seconds=step_s * steps)).strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        actual = ctx.data.window_outcome(record["symbol"], "1h", open_iso, record["window_close"])
    except MarketDataError as e:
        raise ServiceError(f"the outcome could not be fetched: {e}", code=e.code) from e

    baseline = None
    anchor = revealed.get("anchor_price")
    if anchor:
        try:
            hist = ctx.data.candles(record["symbol"], "1h", limit=500 + steps,
                                    until_ms=int(close_dt.timestamp() * 1000) - step_s * steps * 1000)
            rets = hist.log_returns(step=steps)
            if rets:
                baseline = empirical_return_samples(float(anchor), rets, count=500)
        except MarketDataError:
            baseline = None

    scores = score_forecast(
        actual_close=float(actual["close"]), actual_high=float(actual["high"]),
        actual_low=float(actual["low"]), distribution=revealed,
        baseline_samples=baseline,
        persistence=float(anchor) if anchor else None)

    entry = ctx.ledger.append("third_party_score", {
        "forecast_id": cid, "label": record["label"], "symbol": record["symbol"],
        "horizon": horizon, "actual_sha256": sha(actual), "scores": scores,
        "scorer_version": SCORER_VERSION, "judged_at": now})
    ctx.invalidate()

    return {
        "commitment_id": cid, "label": record["label"], "symbol": record["symbol"],
        "horizon": horizon,
        "digest_verified": True,
        "committed_at": record["committed_at"],
        "window": {"open": open_iso, "close": record["window_close"]},
        "actual": actual,
        "scores": scores,
        "scorer_version": SCORER_VERSION,
        "scored_with": ("core/scoring.py, byte-identical to the code that scores Horos's own "
                        "forecasts — same fair CRPS estimator, same empirical baseline"),
        "baseline_available": baseline is not None,
        "ledger": {"seq": entry.seq, "entry_hash": entry.hash},
        "leaderboard": "this result now appears in the free `leaderboard` endpoint",
    }


@service(
    endpoint="leaderboard", group="G. Accountability", price=0.001,
    title="Every participant ranked, including us",
    short="Forecaster leaderboard",
    summary="Ranks everyone who has committed and revealed forecasts through this ledger by the same "
            "proper scores, with Horos in the table on identical terms rather than presenting itself "
            "as the benchmark.",
    returns="Returns each participant with their scored forecast count, mean CRPS, skill against the "
            "baseline and coverage, ranked by skill, plus how to enter.",
    depth="Horos appears as one row among others, on identical terms. Ranking is by skill against the "
          "empirical baseline rather than by raw CRPS, because CRPS is in price units and a "
          "participant forecasting a cheaper asset would otherwise top the table automatically. "
          "Participants with too few scored forecasts are listed but marked as provisional.",
    params=(Param("symbol", "string", "Restrict the table to one symbol.", default=None),),
    example={"symbol": "BTC/USDT:USDT"},
)
def leaderboard(inp: dict, ctx) -> dict:
    symbol = inp.get("symbol")
    by_label: dict[str, list] = {}

    ours = _scored_rows(ctx, symbol)
    if ours:
        by_label["Horos"] = [r["scores"] for r in ours]
    for e in ctx.ledger:
        if e.kind == "third_party_score":
            if symbol and e.body.get("symbol") != symbol:
                continue
            by_label.setdefault(e.body.get("label", "unknown"), []).append(e.body.get("scores", {}))

    if not by_label:
        # Scoped to the filter. Saying "nobody has a scored forecast" while another symbol's table
        # is already populated would be false to whoever asked about this one.
        anywhere = len(_scored_rows(ctx))
        empty = _empty_note(ctx, symbol)
        return {
            "participants": [],
            "filter": {"symbol": symbol} if symbol else None,
            "note": (f"No participant has a scored forecast for {symbol} yet — including Horos."
                     if symbol and anywhere else
                     "Nobody has a scored forecast in this ledger yet — including Horos.")
                    + " The table fills as windows close and forecasts are graded.",
            "record_available_for": empty.get("record_available_for"),
            "pending_forecasts": empty["pending_forecasts"],
            "pending_count": empty["pending_count"],
            "next_grades_at": empty["next_grades_at"],
            "how_to_enter": _how_to_enter(),
            "ledger": ctx.counts(),
        }

    rows = []
    for label, scores in by_label.items():
        agg = aggregate(scores)
        rows.append({
            "label": label,
            "scored_forecasts": agg["count"],
            "mean_crps": agg.get("mean_crps"),
            "crps_skill_vs_empirical": agg.get("crps_skill_vs_empirical"),
            "coverage": agg.get("coverage"),
            "provisional": agg["count"] < 30,
        })
    rows.sort(key=lambda r: (r["crps_skill_vs_empirical"] is None,
                             -(r["crps_skill_vs_empirical"] or 0)))
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    return {
        "participants": rows,
        "ranked_by": ("skill against an empirical random-walk baseline, not raw CRPS — CRPS is in "
                      "price units, so ranking on it would reward whoever forecasts the cheapest "
                      "asset"),
        "provisional_threshold": 30,
        "horos_rank": next((r["rank"] for r in rows if r["label"] == "Horos"), None),
        "how_to_enter": _how_to_enter(),
        "ledger": ctx.counts(),
    }


def _how_to_enter() -> list[str]:
    return [
        "1. Build your forecast as a JSON object with 'samples' (or 'quantiles' plus "
        "'quantile_levels'), optionally 'bands' and 'anchor_price'.",
        "2. Canonicalise it — sorted keys, separators (',',':'), ensure_ascii false — and take its "
        "SHA-256.",
        "3. Call `commit` with that digest, the symbol, the window close and your label, before the "
        "window closes.",
        "4. After the window closes, call `judge` with the commitment id and the revealed forecast.",
        "5. Your result appears here, scored by the same code that scores Horos.",
    ]


@service(
    endpoint="receipt.verify", group="G. Accountability", price=0.001,
    title="Check a Horos receipt without trusting Horos",
    short="Verify a Horos receipt",
    summary="Verifies the Ed25519 signature on any Horos response by rebuilding its manifest from "
            "the fields the response itself carries, and confirms the signing key is the one this "
            "service publishes.",
    returns="Returns whether the signature is valid, whether the manifest rebuilt from the response "
            "matches what was signed, whether the key is ours, and the exact steps to run the same "
            "check offline.",
    depth="Priced at a tenth of a cent so there is no reason not to check, and the offline "
          "instructions are published so there is no reason to keep paying for it. The signature "
          "covers a manifest of six fields, all echoed in every response, so verification needs "
          "nothing from us but the public key — which is served free at /verify.",
    params=(Param("receipt", "object", "The 'receipt' block from a Horos response.", required=True),
            Param("envelope", "object", "The full response, to rebuild the manifest from.",
                  default=None)),
    example={"receipt": {"algorithm": "ed25519", "manifest_sha256": "sha256:...",
                         "signature": "...", "public_key": "..."}},
)
def receipt_verify(inp: dict, ctx) -> dict:
    from core.crypto import Signer, digest

    receipt = inp.get("receipt")
    if not isinstance(receipt, dict):
        raise ServiceError("'receipt' must be the receipt object from a Horos response.",
                           code="bad_receipt")
    for field in ("manifest_sha256", "signature", "public_key"):
        if field not in receipt:
            raise ServiceError(f"the receipt is missing '{field}'.", code="bad_receipt")

    ours = ctx.signer.public_key
    is_ours = receipt["public_key"] == ours
    valid = Signer.verify(receipt["manifest_sha256"], receipt["signature"], receipt["public_key"])

    rebuilt = None
    matches = None
    env = inp.get("envelope")
    if isinstance(env, dict):
        try:
            manifest = {
                "endpoint": env["endpoint"],
                "input_sha256": env["input_sha256"],
                "output_sha256": env["output_sha256"],
                "tool": "horos",
                "price_usdt": env["price_usdt"],
                "job_id": env["job_id"],
            }
            rebuilt = digest(manifest)
            matches = rebuilt == receipt["manifest_sha256"]
        except KeyError as e:
            rebuilt = f"the envelope is missing {e}, so the manifest could not be rebuilt"

    verdict = ("valid and issued by Horos" if valid and is_ours and matches is not False else
               "the signature is valid but the key is not the one Horos publishes"
               if valid and not is_ours else
               "the signature is valid but the envelope does not rebuild to the signed manifest — "
               "the response body has been altered since it was signed"
               if valid and matches is False else
               "the signature does not verify")

    return {
        "signature_valid": valid,
        "signed_by_horos": is_ours,
        "horos_public_key": ours,
        "manifest_rebuilt_from_envelope": rebuilt,
        "manifest_matches": matches,
        "verdict": verdict,
        "offline_check": {
            "algorithm": "Ed25519",
            "signed_bytes": "the UTF-8 bytes of receipt.manifest_sha256, verbatim",
            "manifest_fields": ["endpoint", "input_sha256", "output_sha256", "tool", "price_usdt",
                                "job_id"],
            "canonical_json": "sorted keys, separators (',',':'), ensure_ascii false, "
                              "prefixed 'sha256:'",
            "python": (
                "import hashlib, json\n"
                "from nacl.signing import VerifyKey\n"
                "m = {'endpoint': env['endpoint'], 'input_sha256': env['input_sha256'],\n"
                "     'output_sha256': env['output_sha256'], 'tool': 'horos',\n"
                "     'price_usdt': env['price_usdt'], 'job_id': env['job_id']}\n"
                "blob = json.dumps(m, sort_keys=True, separators=(',', ':'), ensure_ascii=False)\n"
                "d = 'sha256:' + hashlib.sha256(blob.encode()).hexdigest()\n"
                "assert d == env['receipt']['manifest_sha256']\n"
                "VerifyKey(bytes.fromhex(env['receipt']['public_key'])).verify(\n"
                "    d.encode(), bytes.fromhex(env['receipt']['signature']))"),
        },
        "note": "run the snippet above and you never need this endpoint again. That is the point.",
    }
