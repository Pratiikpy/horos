"""The one A2A service — hire Horos over chat for work that does not fit a fixed call.

Thirty-nine services are bounded request/response with a declared schema, which is what per-call
x402 pricing is for. Some work genuinely is not: a multi-asset review, a bespoke horizon, an ongoing
watch on a position. Those need scope agreed in conversation before a price makes sense.

It also matters for listing. OKX's second gate is a live A2A probe — a reviewer registers, says "I
would like to use the services of agent ID X", and the agent must answer. An ASP with no A2A route
has nothing to answer with.

**This is the only place a language model touches the product, and it is deliberate.** The model
plans and narrates; it never produces a number. Every figure in a deliverable comes from one of the
39 deterministic or committed services, is tagged with the endpoint that produced it, and can be
bought separately and compared. A language model writing the numbers would attach unfalsifiable prose
to a falsifiable record, which is the exact thing this product exists to be the opposite of.
"""
from __future__ import annotations

import json
import time

from core.llm import LLMError, shared as llm

from .registry import Param, ServiceError, service

# What the planner is allowed to call. A closed list, so a hallucinated endpoint cannot be invoked
# and cannot silently return nothing.
PLANNABLE = [
    "market.candles", "market.orderbook", "market.funding", "market.openinterest",
    "market.basis", "market.venues", "market.surprise", "market.regime", "market.anomaly",
    "market.dislocation", "metrics.indicators", "metrics.risk", "metrics.performance",
    "metrics.volatility", "metrics.distribution", "metrics.correlation", "metrics.liquidity",
    "metrics.microstructure", "metrics.drawdown", "forecast.path", "forecast.range",
    "forecast.touch", "forecast.volatility", "forecast.reliability", "risk.size", "risk.stop",
    "risk.portfolio", "risk.stress", "portfolio.optimise", "exec.impact", "exec.venue",
    "scorecard", "scorecard.regime", "scorecard.model", "leaderboard", "exec.quality",
]

# Deliberately NOT plannable: commit, judge and receipt.verify all require something only the
# caller can supply — a digest of their own forecast, a matching reveal, or a receipt to check.
# A planner cannot invent those, so exposing them would produce steps that always fail.

MAX_STEPS = 8

# Punctuation that does not survive the trip to the buyer.
#
# A delivered A2A file was read back and found to contain the literal characters "â€”" where an em
# dash had been written: the UTF-8 bytes were decoded as cp1252 somewhere between this service and
# the file the buyer opens, then re-encoded. The buyer sees corrupted text in the artifact they paid
# for. The encoding step is not ours to fix, so the durable answer is to stop emitting characters
# that depend on it being right. Asking the model for ASCII is not enough on its own — it will
# occasionally reach for a nicer dash — so the substitution is applied to what it returns.
_ASCII_PUNCTUATION = {
    "—": "--", "–": "-", "‒": "-", "−": "-",
    "‘": "'", "’": "'", "‚": "'",
    "“": '"', "”": '"', "„": '"',
    "…": "...", " ": " ", " ": " ", " ": " ",
    "×": "x", "→": "->", "≤": "<=", "≥": ">=",
}


def to_ascii_punctuation(value):
    """Replace typographic punctuation with ASCII, recursively through the brief."""
    if isinstance(value, str):
        for bad, good in _ASCII_PUNCTUATION.items():
            value = value.replace(bad, good)
        return value
    if isinstance(value, dict):
        return {k: to_ascii_punctuation(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_ascii_punctuation(v) for v in value]
    return value

PLANNER_SYSTEM = """You plan analysis for Horos, a market-analysis agent.

You do NOT produce numbers, prices, forecasts or opinions. You choose which of Horos's services to
call and with what arguments. Every figure in the final answer comes from those services.

Return ONLY a JSON object:
{"understanding": "one sentence restating what the user actually wants",
 "steps": [{"endpoint": "<one of the allowed endpoints>", "input": {...}, "why": "one short clause"}],
 "out_of_scope": ["anything they asked for that Horos cannot answer"]}

Rules:
- At most 8 steps. Choose the fewest that genuinely answer the question.
- Only use endpoints from the allowed list. Never invent one.
- Symbols are ccxt unified: "BTC/USDT:USDT" for the perpetual, "BTC/USDT" for spot.
- If they ask whether to buy, sell or hold, put that in out_of_scope. Horos measures; it does not
  advise, and it publishes no directional view.
- If they ask for a backtest or historical performance of the model, put it in out_of_scope: the
  model's pre-training contaminates any backtest, so Horos publishes none."""

WRITER_SYSTEM = """You write the summary for a Horos analysis.

You are given the user's request and the actual JSON output of services that were run. Write a clear
brief for a professional.

Absolute rules:
- Use ONLY numbers that appear in the provided results. Never compute, estimate, round differently
  or infer a figure that is not there. If something was not measured, say it was not measured.
- Cite the endpoint after each figure, like: funding is at the 97th percentile (market.funding).
- No buy/sell/hold. No directional prediction of your own. No financial advice.
- Where a result carries a caveat, warning or "not_computed" entry, surface it rather than omitting
  it. Bad news comes first.
- Plain professional English. No hype, no filler, no bullet-point padding.
- ASCII punctuation only: use "--" not an em dash, straight quotes not curly ones, "..." not an
  ellipsis character. The brief is written to a plain text file for the buyer, and non-ASCII
  punctuation has been observed arriving corrupted in a delivered file.

Return ONLY a JSON object:
{"summary": "2-4 sentences answering the question directly",
 "findings": [{"finding": "...", "evidence": "...", "endpoint": "..."}],
 "caveats": ["what could not be measured or should not be over-read"],
 "suggested_next": ["specific follow-up calls worth making, if any"]}"""


@service(
    endpoint="analysis.task", group="H. Agent to agent", price=0.25,
    title="Hire Horos for analysis that does not fit a fixed call",
    short="Custom market analysis",
    summary="Describe a market question in plain language and Horos plans which of its data, "
            "forecast, risk and execution services answer it, runs them, and returns a signed "
            "brief with every figure traced to the endpoint that produced it.",
    returns="Returns the plan it chose and why, the full raw output of every service it ran, a "
            "written brief citing each figure to its source, the caveats, and anything you asked "
            "for that is out of scope.",
    depth="A language model chooses and narrates; it never produces a number. Every figure comes "
          "from a deterministic or committed service, is tagged with its endpoint, and can be "
          "bought separately and compared against this answer. Raw results are returned in full "
          "alongside the prose so the brief can be checked against its own evidence rather than "
          "taken on trust.",
    params=(Param("request", "string", "What you want analysed, in plain language.", required=True),
            Param("symbols", "array", "Symbols to focus on, if you already know them.",
                  default=None),
            Param("max_steps", "integer", "Cap on how many services are run, 1-8.", default=6)),
    example={"request": "I'm long 2 BTC perp. How exposed am I over the next day, and is the "
                        "current funding unusual?",
             "symbols": ["BTC/USDT:USDT"], "max_steps": 6},
)
def analysis_task(inp: dict, ctx) -> dict:
    started = time.time()
    request = str(inp.get("request", "")).strip()
    if not request:
        raise ServiceError("a 'request' describing what you want analysed is required.",
                           code="missing_request")
    if len(request) > 4000:
        raise ServiceError("the request is too long; keep it under 4000 characters.",
                           code="request_too_long")
    max_steps = max(1, min(int(inp.get("max_steps", 6)), MAX_STEPS))

    model = llm()
    if not model.configured:
        raise ServiceError(
            "the planning model is not configured on this deployment, so a free-form analysis "
            "cannot be planned. Every underlying measurement is still available as a direct "
            "service — see GET /services.", code="planner_unavailable", status=503)

    # One deadline for planning, execution and writing together. x402 declares 300 seconds; the
    # services in the middle need most of it, so the two model calls get a hard 150 between them.
    deadline = started + 150

    # The catalogue carries each endpoint's real parameter contract, not just its summary. With only
    # summaries the planner guesses argument values — it asked for horizon "1d" when the valid set
    # is 1h/4h/12h/24h, which burned one of the user's steps on a guaranteed failure. Handing it the
    # required fields and the worked example removes the guessing.
    lines = []
    for e in PLANNABLE:
        svc = ctx.registry.get(e)
        if svc is None:
            continue
        contract = svc.input_contract()
        allowed = []
        for p in svc.params:
            hint = p.description
            allowed.append(f"{p.name}{'*' if p.required else ''}: {hint[:70]}")
        lines.append(f"- {e}: {svc.summary[:100]}\n    args: {'; '.join(allowed) or 'none'}\n"
                     f"    example: {json.dumps(contract['example'], separators=(',', ':'))}")
    catalogue = "\n".join(lines)
    focus = f"\nSymbols the user named: {inp['symbols']}" if inp.get("symbols") else ""

    try:
        plan, planner_model = model.json_complete(
            PLANNER_SYSTEM,
            f"Allowed endpoints:\n{catalogue}\n\nUser request:\n{request}{focus}\n\n"
            f"Use at most {max_steps} steps.",
            required=["understanding", "steps"], max_tokens=1200, deadline=deadline)
    except LLMError as e:
        raise ServiceError(f"the analysis could not be planned: {e}", code=e.code,
                           status=503) from e

    steps = plan.get("steps") or []
    if not isinstance(steps, list):
        raise ServiceError("the planner returned an unusable plan.", code="bad_plan", status=503)

    results, ran, refused = [], [], []
    for step in steps[:max_steps]:
        if not isinstance(step, dict):
            continue
        endpoint = str(step.get("endpoint", ""))
        if endpoint not in PLANNABLE:
            refused.append({"endpoint": endpoint,
                            "reason": "not an endpoint Horos offers, so it was not run"})
            continue
        svc = ctx.registry.get(endpoint)
        if svc is None:
            refused.append({"endpoint": endpoint, "reason": "unknown service"})
            continue
        payload = step.get("input") if isinstance(step.get("input"), dict) else {}
        try:
            out = svc.handler(payload, ctx)
            results.append({"endpoint": endpoint, "input": payload,
                            "why": step.get("why"), "output": out})
            ran.append(endpoint)
        except Exception as e:                                     # noqa: BLE001
            # A step that failed is reported as failed. Dropping it would let the brief be written
            # as though the measurement had simply not been needed.
            results.append({"endpoint": endpoint, "input": payload, "why": step.get("why"),
                            "failed": f"{type(e).__name__}: {e}"})

    if not ran:
        raise ServiceError(
            "no service in the plan produced a result. Nothing was measured, so no brief is "
            "written — see 'attempted' for what was tried and why each failed.",
            code="no_results", status=502, attempted=results, plan=plan)

    evidence = json.dumps(
        [{"endpoint": r["endpoint"], "output": r.get("output"), "failed": r.get("failed")}
         for r in results], separators=(",", ":"), default=str)[:60_000]

    brief, writer_model = None, None
    brief_error = None
    try:
        brief, writer_model = model.json_complete(
            WRITER_SYSTEM,
            f"User request:\n{request}\n\nService results (the ONLY source of numbers):\n{evidence}",
            required=["summary", "findings"], max_tokens=2500, deadline=deadline)
    except LLMError as e:
        brief_error = str(e)

    # Only the prose is normalised. `raw_results` carries the services' own output and is what the
    # brief is checked against, so rewriting characters inside it would break that correspondence.
    brief = to_ascii_punctuation(brief)

    return {
        "request": request,
        "understanding": plan.get("understanding"),
        "plan": [{"endpoint": s.get("endpoint"), "why": s.get("why")}
                 for s in steps[:max_steps] if isinstance(s, dict)],
        "services_run": ran,
        "services_refused": refused,
        "out_of_scope": plan.get("out_of_scope") or [],
        "brief": brief,
        "brief_unavailable": brief_error,
        # Always returned in full. The prose is checkable against this rather than replacing it.
        "raw_results": results,
        "models": {"planner": planner_model, "writer": writer_model,
                   "role": "the model chose which services to run and wrote the prose. It produced "
                           "no numbers — every figure above comes from a named service and can be "
                           "bought separately and compared."},
        "elapsed_seconds": round(time.time() - started, 2),
        "not_advice": ("Horos measures and forecasts with uncertainty attached. It does not tell you "
                       "whether to trade, and publishes no directional view."),
    }
