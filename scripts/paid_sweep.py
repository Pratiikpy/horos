"""Buy every service as a customer would, then judge what came back.

OKX asks for exactly this before they will review: *"please register a user agent ID and use your
agents as if you were a user. If your agent passes successfully (i.e. you as a user makes a payment
and then receive a deliverable) please let me know."*

A 200 is the entry requirement, not the result. Five separate times on a sibling service an endpoint
returned 200 while the deliverable was wrong, so every response here is put through outcome checks
that know what the service was supposed to produce:

  * did it return the *shape* the description promises, or a plausible-looking husk
  * are the numbers finite, in range, and internally consistent
  * did anything come back as a silent null that should have been a stated failure
  * is the signed receipt verifiable from the response alone

Each check is a predicate with a sentence explaining what a failure would mean. The report separates
"paid and delivered" from "paid and delivered *something useful*", because they are different claims
and only the second one is worth anything.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Importing the registry alone yields an empty catalogue — the services register themselves as a
# side effect of their own module import. Without these the sweep silently tests nothing and reports
# a clean 0/0, which is the most dangerous possible result for a verification script.
from services import a2a, accountability, forecast, market, metrics, risk  # noqa: F401,E402
from services.registry import REGISTRY  # noqa: E402
from x402_bridge import PAYMENT_REQUIRED_HEADER  # noqa: E402

# The forecast used by the commit/judge pair. Committed as a digest, then revealed verbatim, so the
# digest check passes and the *window* check is what refuses — which is the guard being tested.
SELFTEST_FORECAST = {"anchor_price": 65000.0,
                     "samples": [64000.0, 64500.0, 65000.0, 65500.0, 66000.0],
                     "bands": {"0.9": [63000.0, 67000.0]}}

# Realistic inputs. Not minimal ones — a sweep that only ever sends the happy-path example proves
# the example works, not that the service does.
INPUTS: dict[str, dict] = {
    "market.candles": {"symbol": "BTC/USDT:USDT", "timeframe": "1h", "limit": 300},
    "market.orderbook": {"symbol": "ETH/USDT:USDT", "depth": 400},
    "market.funding": {"symbol": "BTC/USDT:USDT", "history": 200},
    "market.openinterest": {"symbol": "ETH/USDT:USDT", "periods": 24},
    "market.basis": {"symbol": "BTC/USDT:USDT"},
    "market.venues": {"symbol": "BTC/USDT:USDT", "exchanges": ["okx", "gate", "bitget", "kucoin"]},
    "metrics.indicators": {"symbol": "BTC/USDT:USDT", "lookback": 500},
    "metrics.risk": {"symbol": "BTC/USDT:USDT", "lookback": 720, "confidence": 0.95},
    "metrics.performance": {"symbol": "ETH/USDT:USDT", "benchmark": "BTC/USDT:USDT",
                            "lookback": 720},
    "metrics.volatility": {"symbol": "BTC/USDT:USDT", "lookback": 720},
    "metrics.distribution": {"symbol": "BTC/USDT:USDT", "lookback": 1000},
    "metrics.correlation": {"symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
                            "lookback": 500},
    "metrics.liquidity": {"symbol": "BTC/USDT:USDT", "lookback": 500},
    "metrics.microstructure": {"symbol": "BTC/USDT:USDT", "lookback": 500},
    "metrics.drawdown": {"symbol": "BTC/USDT:USDT", "lookback": 2000, "threshold_pct": 2.0},
    "forecast.path": {"symbol": "BTC/USDT:USDT", "horizon": "4h"},
    "forecast.range": {"symbol": "ETH/USDT:USDT", "horizon": "24h"},
    "forecast.touch": {"symbol": "BTC/USDT:USDT", "horizon": "12h", "levels": [70000, 60000]},
    "forecast.volatility": {"symbol": "ETH/USDT:USDT", "horizon": "24h", "lookback": 168},
    "forecast.reliability": {"symbol": "BTC/USDT:USDT"},
    "market.surprise": {"symbol": "BTC/USDT:USDT"},
    "market.regime": {"symbol": "BTC/USDT:USDT", "lookback": 500},
    "market.anomaly": {"symbol": "BTC/USDT:USDT", "periods": 24},
    "market.dislocation": {"symbol": "BTC/USDT:USDT", "size_usd": 50000},
    "risk.size": {"symbol": "BTC/USDT:USDT", "horizon": "24h", "account_equity": 100000,
                  "max_loss_pct": 1.0, "confidence": 0.95},
    "risk.stop": {"symbol": "BTC/USDT:USDT", "horizon": "24h", "direction": "long",
                  "stopout_tolerance": 0.1},
    "risk.portfolio": {"positions": [{"symbol": "BTC/USDT:USDT", "quantity": 1.5},
                                     {"symbol": "ETH/USDT:USDT", "quantity": -10}],
                       "confidence": 0.95},
    "risk.stress": {"positions": [{"symbol": "BTC/USDT:USDT", "quantity": 1},
                                  {"symbol": "SOL/USDT:USDT", "quantity": 100}],
                    "shocks_pct": [-30, -10, 10], "loss_limit": 20000},
    "portfolio.optimise": {"symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
                           "lookback": 1000},
    "exec.impact": {"symbol": "BTC/USDT:USDT", "side": "buy", "notional_usd": 250000},
    "exec.quality": {"symbol": "BTC/USDT:USDT", "fill_price": 65010.5, "side": "buy",
                     "timestamp_ms": 0, "quantity": 0.5},          # timestamp filled at runtime
    "exec.venue": {"symbol": "BTC/USDT:USDT", "side": "buy", "notional_usd": 500000,
                   "exchanges": ["okx", "gate", "bitget", "kucoin"]},
    "scorecard": {},
    "scorecard.regime": {},
    "scorecard.model": {},
    "leaderboard": {},
    "commit": {},           # filled at runtime: needs a live digest and a future window
    "judge": {},            # exercised separately, needs a matching commitment
    "receipt.verify": {},   # filled at runtime from a real receipt
    "analysis.task": {"request": "I hold 2 BTC perp. Summarise my next-day risk and tell me "
                                 "whether funding is unusual.",
                      "symbols": ["BTC/USDT:USDT"], "max_steps": 3},
}


def _n(x) -> bool:
    """A real, finite number."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) != float("inf")


def _nested(d: dict, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


# What each service must actually deliver. A predicate and the sentence a failure would mean.
CHECKS: dict[str, list[tuple[str, callable]]] = {
    "market.candles": [
        ("returned the requested candles", lambda o: len(o.get("candles", [])) > 250),
        ("every candle is closed", lambda o: o.get("open_candle_excluded") is True),
        ("series is contiguous", lambda o: o.get("contiguous") is True),
        ("summary prices are real", lambda o: _n(_nested(o, "summary", "last_close"))),
    ],
    "market.orderbook": [
        ("both sides present", lambda o: o.get("bids") and o.get("asks")),
        ("spread is positive", lambda o: _n(o.get("spread_bps")) and o["spread_bps"] > 0),
        ("depth buckets flag truncation honestly",
         lambda o: all("note" in v or "incomplete" not in v
                       for v in o.get("depth_notional_usd", {}).values())),
    ],
    "market.funding": [
        ("rate is a real number", lambda o: _n(o.get("funding_rate"))),
        ("percentile is on a named scale",
         lambda o: "percentile_0_100" in o or "history_unavailable" in o),
        ("reading is a sentence, not a stub",
         lambda o: isinstance(o.get("reading"), str) and len(o["reading"]) > 30),
    ],
    "market.openinterest": [
        ("open interest present", lambda o: o.get("open_interest") or o.get("open_interest_value")),
        ("interpretation given or absence explained",
         lambda o: isinstance(o.get("reading"), str) or "change_unavailable" in o),
    ],
    "market.basis": [
        ("both legs priced", lambda o: _n(o.get("perpetual_price")) and _n(o.get("spot_price"))),
        ("basis computed", lambda o: _n(o.get("basis_pct"))),
    ],
    "market.venues": [
        ("compared at least two venues", lambda o: o.get("venues_compared", 0) >= 2),
        ("spread computed", lambda o: _n(o.get("spread_bps"))),
        ("stale feeds reported separately", lambda o: "stale_excluded" in o),
    ],
    "metrics.indicators": [
        ("computed a real indicator set", lambda o: o.get("total", 0) > 50),
        ("unavailable ones are named, not nulled",
         lambda o: isinstance(o.get("unavailable"), list)),
        ("candle digest published", lambda o: _nested(o, "provenance", "candles_sha256")),
    ],
    "metrics.risk": [
        ("all four VaR estimators computed",
         lambda o: sum(1 for v in o.get("value_at_risk", {}).values() if _n(v)) == 4),
        ("all five CVaR estimators computed",
         lambda o: sum(1 for v in o.get("conditional_value_at_risk", {}).values() if _n(v)) == 5),
        ("CVaR is worse than VaR, as it must be",
         lambda o: _nested(o, "conditional_value_at_risk", "historic")
                   <= _nested(o, "value_at_risk", "historic")),
        ("failures are explained, never silent",
         lambda o: isinstance(o.get("not_computed"), dict)),
    ],
    "metrics.performance": [
        ("sharpe computed", lambda o: _n(_nested(o, "ratios", "sharpe"))),
        ("beta against the benchmark", lambda o: _n(_nested(o, "against_benchmark", "beta"))),
        ("short-window annualisation is disclosed",
         lambda o: _nested(o, "reliability", "sharpe_standard_error") is not None),
        ("annualised on hours, not 252 days",
         lambda o: _nested(o, "annualisation", "periods_per_year") == 8760),
    ],
    "metrics.volatility": [
        ("all four estimators computed",
         lambda o: sum(1 for v in o.get("estimates_per_candle", {}).values() if _n(v)) == 4),
        ("GARCH fitted or its failure explained",
         lambda o: _nested(o, "garch", "fitted") is True or _nested(o, "garch", "reason")),
        ("reading interprets the disagreement", lambda o: len(str(o.get("reading", ""))) > 25),
    ],
    "metrics.distribution": [
        ("moments computed", lambda o: _n(_nested(o, "moments", "excess_kurtosis"))),
        ("normality tested with a p-value", lambda o: _n(_nested(o, "normality", "p_value"))),
        ("verdict says whether gaussian risk models apply",
         lambda o: len(str(_nested(o, "normality", "verdict"))) > 30),
    ],
    "metrics.correlation": [
        ("matrix returned", lambda o: len(o.get("correlation_matrix", {})) >= 2),
        ("self-correlation is exactly 1",
         lambda o: all(abs(o["correlation_matrix"][s][s] - 1.0) < 1e-9
                       for s in o["correlation_matrix"])),
        ("aligned on timestamps, count reported",
         lambda o: o.get("aligned_observations", 0) > 50),
    ],
    "metrics.liquidity": [
        ("realised measures computed",
         lambda o: _n(_nested(o, "realised", "amihud_illiquidity_x1e9"))),
        ("verdict is a usable sentence", lambda o: len(str(o.get("verdict", ""))) > 25),
        ("Roll's undefined case is explained, not nulled",
         lambda o: _n(_nested(o, "realised", "roll_effective_spread"))
                   or _nested(o, "realised", "roll_note")),
    ],
    "metrics.microstructure": [
        ("body/wick profile computed", lambda o: _n(_nested(o, "body", "mean_body_to_range"))),
        ("runs compared against a coin flip",
         lambda o: _n(_nested(o, "runs", "expected_longest_under_coin_flip"))),
    ],
    "metrics.drawdown": [
        ("max drawdown reported", lambda o: _n(o.get("max_drawdown_pct"))),
        ("summary states the real situation", lambda o: len(str(o.get("summary", ""))) > 40),
        ("an open drawdown is not reported as zero episodes",
         lambda o: not (o.get("in_drawdown_now") and o.get("worst") is None)),
    ],
    "forecast.path": [
        ("full quantile grid", lambda o: len(o.get("quantiles", [])) == 9),
        ("quantiles are monotone",
         lambda o: all(a <= b for a, b in zip(o["quantiles"], o["quantiles"][1:]))),
        ("the raw ensemble is published", lambda o: len(o.get("terminal_prices", [])) >= 32),
        ("says where it came from", lambda o: _nested(o, "provenance", "served_from") in
                                              ("ledger", "live")),
        ("names what the model cannot see", lambda o: len(o.get("model_blind_spots", [])) >= 4),
    ],
    "forecast.range": [
        ("bands returned", lambda o: len(o.get("bands", {})) == 3),
        ("wider coverage means a wider band",
         lambda o: _nested(o, "bands", "0.5", "width_pct")
                   <= _nested(o, "bands", "0.9", "width_pct")),
        ("every band states whether it is calibrated",
         lambda o: all(isinstance(_nested(b, "calibration", "calibrated"), bool)
                       for b in o.get("bands", {}).values())),
    ],
    "forecast.touch": [
        ("each level answered", lambda o: len(o.get("levels", [])) == 2),
        ("probabilities are probabilities",
         lambda o: all(0.0 <= l["probability"] <= 1.0 for l in o.get("levels", []))),
        ("resolved on path extremes", lambda o: "path" in str(o.get("resolution", "")).lower()),
    ],
    "forecast.volatility": [
        ("expected volatility computed",
         lambda o: _n(_nested(o, "expected", "annualised_pct"))),
        ("compared against realised", lambda o: _n(_nested(o, "realised", "annualised_pct"))),
        ("reading interprets the ratio", lambda o: len(str(o.get("reading", ""))) > 20),
    ],
    "forecast.reliability": [
        ("gives a verdict or says there is no record",
         lambda o: o.get("verdict") == "no_record_yet" or o.get("by_horizon")),
        ("recommendation is actionable prose",
         lambda o: len(str(o.get("recommendation") or o.get("summary") or "")) > 30),
    ],
    "market.surprise": [
        ("percentile computed", lambda o: _n(o.get("percentile_rank_0_1"))),
        ("says what it measured against",
         lambda o: len(str(o.get("measured_against", ""))) > 15),
        ("degraded mode is disclosed",
         lambda o: o.get("forecast_id") or "note" in o),
    ],
    "market.regime": [
        ("volatility regime classified",
         lambda o: _nested(o, "volatility", "regime") in ("high", "low", "normal")),
        ("hurst estimated or refused", lambda o: _n(_nested(o, "hurst", "exponent"))
                                                 or "could not" in str(_nested(o, "hurst", "interpretation"))),
        ("state is one readable line", lambda o: len(str(o.get("state", ""))) > 20),
    ],
    "market.anomaly": [
        ("price change measured", lambda o: _n(_nested(o, "signals", "price_change_pct"))),
        ("divergences is a list", lambda o: isinstance(o.get("divergences"), list)),
        ("unavailable signals are named",
         lambda o: isinstance(o.get("signals_unavailable"), list)),
        ("liquidation gap disclosed", lambda o: "liquidation" in str(o.get("caveat", "")).lower()),
    ],
    "market.dislocation": [
        ("spread measured", lambda o: _n(o.get("spread_bps"))),
        ("depth actually checked", lambda o: len(o.get("depth_checks", {})) >= 1),
        ("verdict distinguishes real from artefact",
         lambda o: len(str(o.get("verdict", ""))) > 25),
    ],
    "risk.size": [
        ("a size was produced or refusal explained",
         lambda o: _n(_nested(o, "position", "units")) or o.get("reason")),
        ("loss at confidence equals the stated budget",
         lambda o: not o.get("sizeable") or
                   abs(_nested(o, "at_this_size", "loss_at_confidence")
                       - _nested(o, "loss_budget", "max_loss_absolute")) < 1.0),
        ("shortfall beyond the limit is worse than the limit",
         lambda o: not o.get("sizeable") or
                   _nested(o, "at_this_size", "expected_shortfall_beyond_it")
                   >= _nested(o, "at_this_size", "loss_at_confidence")),
        ("explicitly not advice", lambda o: "not_advice" in o or "reason" in o),
    ],
    "risk.stop": [
        ("stop level produced", lambda o: _n(_nested(o, "stop", "level"))),
        ("stop-out probability near the tolerance asked for",
         lambda o: abs(_nested(o, "stop", "stopout_probability") - 0.1) < 0.08),
        ("whipsaw cost quantified",
         lambda o: _n(_nested(o, "stop", "whipsaw_probability"))),
    ],
    "risk.portfolio": [
        ("VaR computed", lambda o: _n(_nested(o, "risk", "value_at_risk"))),
        ("shortfall is worse than VaR",
         lambda o: _nested(o, "risk", "expected_shortfall") <= _nested(o, "risk", "value_at_risk")),
        ("component risks sum to portfolio risk",
         lambda o: abs(sum(c["component_risk"] for c in o["contribution"].values())
                       - _nested(o, "risk", "portfolio_sd_per_candle")) < 1.0),
    ],
    "risk.stress": [
        ("scenarios computed", lambda o: len(o.get("scenarios", [])) >= 3),
        ("a bigger shock hurts more",
         lambda o: min(s["total_pnl"] for s in o["scenarios"]) ==
                   next(s["total_pnl"] for s in o["scenarios"]
                        if s["shock_pct"] == min(x["shock_pct"] for x in o["scenarios"]))),
        ("reverse stress answers the real question",
         lambda o: o.get("reverse_stress") is not None),
        ("correlation caveat stated", lambda o: "correlation" in str(o.get("caveat", "")).lower()),
    ],
    "portfolio.optimise": [
        ("mean-variance produced weights",
         lambda o: _nested(o, "results", "mean_variance", "weights")),
        ("HRP produced weights",
         lambda o: _nested(o, "results", "hierarchical_risk_parity", "weights")),
        ("weights sum to one",
         lambda o: abs(sum(_nested(o, "results", "mean_variance", "weights").values()) - 1) < 0.02),
        ("method disagreement is surfaced", lambda o: o.get("method_agreement") is not None),
    ],
    "exec.impact": [
        ("walked the book", lambda o: _n(_nested(o, "impact", "average_fill_price"))),
        ("slippage computed", lambda o: _n(_nested(o, "impact", "slippage_bps"))),
        ("a partial fill would be disclosed",
         lambda o: _nested(o, "impact", "fully_filled") is True
                   or "warning" in o.get("impact", {})),
        ("size ladder returned", lambda o: len(o.get("by_size", {})) == 4),
    ],
    "exec.quality": [
        ("graded against three benchmarks", lambda o: len(o.get("slippage_bps", {})) >= 3),
        ("verdict given", lambda o: len(str(o.get("verdict", ""))) > 15),
        ("vwap proxy is labelled a proxy",
         lambda o: "proxy" in str(o.get("benchmark_caveat", "")).lower()),
    ],
    "exec.venue": [
        ("ranked at least two venues", lambda o: len(o.get("ranked", [])) >= 2),
        ("best is genuinely the cheapest fill",
         lambda o: o["ranked"][0]["exchange"] == o["best"]),
        ("saving quantified", lambda o: _n(o.get("saving_vs_worst_usd"))),
    ],
    "scorecard": [
        ("states the ledger position", lambda o: _nested(o, "ledger", "entries") > 0),
        ("publishes coverage provenance", lambda o: _nested(o, "coverage", "selection", "basis")),
        ("empty record says so rather than faking one",
         lambda o: o.get("scored_forecasts", 0) > 0 or "note" in o),
        ("anchoring evidence present", lambda o: _nested(o, "anchoring", "anchors") is not None),
    ],
    "scorecard.regime": [("returns a breakdown or explains why not",
                          lambda o: o.get("by_regime") or o.get("note"))],
    "scorecard.model": [("returns per-model rows or explains why not",
                         lambda o: o.get("models") or o.get("note"))],
    "leaderboard": [
        ("participants is a list", lambda o: isinstance(o.get("participants"), list)),
        ("explains how to enter", lambda o: len(o.get("how_to_enter", [])) >= 4),
        ("ranks on skill, not raw CRPS",
         lambda o: "skill" in str(o.get("ranked_by", "")).lower() or not o["participants"]),
    ],
    "commit": [
        ("commitment recorded", lambda o: o.get("committed") is True),
        ("ledger position returned", lambda o: _nested(o, "ledger", "entry_hash")),
        ("tells the caller the next step", lambda o: "judge" in str(o.get("next_step", ""))),
    ],
    "judge": [
        ("verifies the reveal against the commitment",
         lambda o: o.get("digest_verified") is True),
        ("does not score a window that has not closed",
         lambda o: o.get("status") == "not_yet_scoreable" or o.get("scores")),
        ("tells a paying caller exactly when to come back",
         lambda o: o.get("status") != "not_yet_scoreable" or "judge again" in str(o.get("next_step",""))),
    ],
    "receipt.verify": [
        ("signature checked", lambda o: isinstance(o.get("signature_valid"), bool)),
        ("verdict given", lambda o: len(str(o.get("verdict", ""))) > 10),
        ("publishes the offline method", lambda o: _nested(o, "offline_check", "python")),
    ],
    "analysis.task": [
        ("ran at least one real service", lambda o: len(o.get("services_run", [])) >= 1),
        ("returned raw evidence alongside prose",
         lambda o: isinstance(o.get("raw_results"), list) and o["raw_results"]),
        ("brief produced or its failure explained",
         lambda o: o.get("brief") or o.get("brief_unavailable")),
        ("every finding cites an endpoint",
         lambda o: not _nested(o, "brief", "findings")
                   or all(f.get("endpoint") for f in _nested(o, "brief", "findings"))),
        ("states it is not advice", lambda o: "not_advice" in o),
    ],
}


def real_payment(challenge_b64: str) -> str:
    """Sign a real x402 authorization through the OKX wallet, settling in USD₮0 on X Layer.

    This is the difference between "the payment path is cryptographically valid" and "money moved".
    OKX reviews on the second, and so should we — a sweep that only ever uses dev-mode signatures
    proves the handler works, not that a customer can buy anything.
    """
    import shutil
    import subprocess

    cli = shutil.which("onchainos")
    if not cli:
        raise RuntimeError("the onchainos CLI is not on PATH; real settlement cannot be tested")
    out = subprocess.run([cli, "payment", "pay", "--payload", challenge_b64],
                         capture_output=True, text=True, timeout=300)
    line = (out.stdout or out.stderr).strip().splitlines()[-1]
    data = json.loads(line)
    if not data.get("ok"):
        raise RuntimeError(f"wallet refused to sign: {str(data)[:200]}")
    return data["data"]["authorization_header"]


def sweep(base: str, only: list[str] | None = None, timeout: float = 300.0,
          real: bool = False) -> dict:
    client = httpx.Client(timeout=timeout, follow_redirects=True)
    endpoints = [s.endpoint for s in REGISTRY.all()]
    if only:
        endpoints = [e for e in endpoints if e in only]
    # A sweep that tests nothing must fail loudly, not report a clean zero.
    if not endpoints:
        raise SystemExit("no services to sweep — the registry is empty, which means the service "
                         "modules were not imported. Refusing to report a vacuous pass.")

    results = []
    receipt_sample = None
    commitment = None

    for ep in endpoints:
        svc = REGISTRY.get(ep)
        payload = dict(INPUTS.get(ep, svc.example))
        # Runtime-dependent inputs.
        if ep == "exec.quality":
            payload["timestamp_ms"] = int((time.time() - 7200) // 60 * 60) * 1000
        if ep == "commit":
            from core.crypto import digest
            payload = {"digest": digest(SELFTEST_FORECAST).split(":", 1)[1],
                       "symbol": "BTC/USDT:USDT", "horizon": "1h", "label": "horos-selftest",
                       "window_close": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                     time.gmtime(time.time() + 7200))}
        if ep == "judge":
            # The *matching* forecast, so the digest check passes and the window check is what
            # refuses. Sending a different one would test the tamper path instead — also a correct
            # refusal, but not the one this row is meant to exercise.
            payload = {"commitment_id": commitment or "tp:none",
                       "forecast": SELFTEST_FORECAST}
        if ep == "receipt.verify" and receipt_sample:
            payload = {"receipt": receipt_sample["receipt"], "envelope": receipt_sample}

        row = {"endpoint": ep, "price": svc.price_str()}
        t0 = time.time()
        try:
            r0 = client.post(f"{base}/a2mcp/{ep}", json={})
            row["challenge_status"] = r0.status_code
            if r0.status_code != 402:
                row["error"] = f"unpaid call returned {r0.status_code}, expected 402"
                results.append(row)
                continue
            challenge = json.loads(base64.b64decode(r0.headers[PAYMENT_REQUIRED_HEADER]))
            row["challenge_amount"] = challenge["accepts"][0]["amount"]

            if real:
                header = real_payment(r0.headers[PAYMENT_REQUIRED_HEADER])
            else:
                from x402_bridge import make_dev_payment
                header = make_dev_payment(challenge)
            r = client.post(f"{base}/a2mcp/{ep}", json={"input": payload},
                            headers={"PAYMENT-SIGNATURE": header})
            settle = r.headers.get("x-payment-response") or r.headers.get("X-PAYMENT-RESPONSE")
            if settle:
                try:
                    s = json.loads(base64.b64decode(settle))
                    row["settlement_tx"] = s.get("transaction")
                    row["settled"] = bool(s.get("success"))
                except Exception:                                    # noqa: BLE001
                    row["settled"] = None
            row["status"] = r.status_code
            row["seconds"] = round(time.time() - t0, 1)
            body = r.json()

            if r.status_code != 200:
                row["error"] = f"{body.get('code')}: {str(body.get('error'))[:150]}"
                # judge is expected to refuse an open window; that is a pass, not a failure.
                results.append(row)
                continue

            row["delivered"] = True
            output = body.get("output", {})
            # Keep the whole exchange. A proof page that shows only a tick has proved nothing —
            # a reader has to be able to see the actual answer that was bought.
            row["request"] = payload
            row["response"] = body
            row["challenge"] = challenge
            if ep == "commit":
                commitment = output.get("commitment_id")
            if receipt_sample is None and body.get("receipt"):
                receipt_sample = body

            # The receipt must verify from the response alone.
            import hashlib

            from nacl.signing import VerifyKey
            manifest = {"endpoint": body["endpoint"], "input_sha256": body["input_sha256"],
                        "output_sha256": body["output_sha256"], "tool": "horos",
                        "price_usdt": body["price_usdt"], "job_id": body["job_id"]}
            blob = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            d = "sha256:" + hashlib.sha256(blob.encode()).hexdigest()
            ok_sig = d == body["receipt"]["manifest_sha256"]
            if ok_sig:
                try:
                    VerifyKey(bytes.fromhex(body["receipt"]["public_key"])).verify(
                        d.encode(), bytes.fromhex(body["receipt"]["signature"]))
                except Exception:                                    # noqa: BLE001
                    ok_sig = False
            row["receipt_verifies"] = ok_sig

            checks = []
            for name, fn in CHECKS.get(ep, []):
                try:
                    checks.append((name, bool(fn(output))))
                except Exception as e:                               # noqa: BLE001
                    checks.append((name, False, f"{type(e).__name__}: {e}"))
            row["outcome"] = checks
            row["outcome_passed"] = sum(1 for c in checks if c[1])
            row["outcome_total"] = len(checks)
        except Exception as e:                                       # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:180]}"
            row["seconds"] = round(time.time() - t0, 1)
        results.append(row)

    return {"base": base, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": results}


def report(data: dict) -> int:
    rows = data["results"]
    delivered = [r for r in rows if r.get("delivered")]
    failed = [r for r in rows if not r.get("delivered")]
    checks_run = sum(r.get("outcome_total", 0) for r in rows)
    checks_ok = sum(r.get("outcome_passed", 0) for r in rows)
    bad_receipts = [r for r in rows if r.get("delivered") and r.get("receipt_verifies") is False]

    print(f"\n{'endpoint':<24} {'$':>7} {'code':>5} {'s':>6}  outcome   receipt")
    print("-" * 74)
    for r in rows:
        oc = (f"{r.get('outcome_passed', 0)}/{r.get('outcome_total', 0)}"
              if r.get("outcome_total") else "  -")
        rec = "ok" if r.get("receipt_verifies") else ("BAD" if r.get("delivered") else "-")
        print(f"{r['endpoint']:<24} {r['price']:>7} {str(r.get('status', '-')):>5} "
              f"{str(r.get('seconds', '-')):>6}  {oc:>7}   {rec}")
        if r.get("error"):
            print(f"    ERROR: {r['error']}")
        for c in r.get("outcome", []):
            if not c[1]:
                extra = f" ({c[2]})" if len(c) > 2 else ""
                print(f"    FAILED CHECK: {c[0]}{extra}")

    print("-" * 74)
    print(f"delivered      : {len(delivered)}/{len(rows)}")
    print(f"outcome checks : {checks_ok}/{checks_run}")
    print(f"receipts valid : {sum(1 for r in delivered if r.get('receipt_verifies'))}/"
          f"{len(delivered)}")
    settled = [r for r in rows if r.get("settlement_tx")]
    if settled:
        print(f"settled onchain: {len(settled)}/{len(rows)}  "
              f"({sum(1 for r in settled if r.get('settled'))} confirmed success)")
        print(f"  first tx: {settled[0]['settlement_tx']}")
    if failed:
        print(f"NOT DELIVERED  : {', '.join(r['endpoint'] for r in failed)}")
    if bad_receipts:
        print(f"BAD RECEIPTS   : {', '.join(r['endpoint'] for r in bad_receipts)}")
    return 0 if (not failed and checks_ok == checks_run and not bad_receipts) else 1


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="https://horos.ivaronix.xyz")
    p.add_argument("--only", nargs="*")
    p.add_argument("--out", default=".data/sweep.json")
    p.add_argument("--real", action="store_true",
                   help="pay with real USD₮0 through the OKX wallet and settle on X Layer")
    a = p.parse_args()
    data = sweep(a.base, a.only, real=a.real)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    raise SystemExit(report(data))
