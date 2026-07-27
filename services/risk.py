"""Groups E and F — sizing, stops, portfolio risk and execution. Eight services.

**These answer "what is consistent with the limit you stated", never "should you trade".** That is
not legal throat-clearing, it is the actual design. `risk.size` takes your loss tolerance as input
and returns the position that satisfies it under the forecast distribution; it has no view on whether
you should hold the position at all. A service that answered the second question would be selling
financial advice on top of a model whose short-horizon direction is close to random.

Sizing is driven off the same committed forecast distribution the scorecard grades, so if the
distribution is badly calibrated the public record shows it — the sizing advice inherits the
accountability of the forecast underneath it rather than floating free of any record.
"""
from __future__ import annotations

import math

import numpy as np

from core.markets import HORIZONS, SCORED_TIMEFRAME

from .registry import Param, ServiceError, service

SYMBOL = Param("symbol", "string", "Unified ccxt symbol, e.g. 'BTC/USDT:USDT'.", required=True)
HORIZON = Param("horizon", "string", "Planning horizon: 1h, 4h, 12h or 24h.", default="24h")


def _distribution(inp: dict, ctx, horizon: str):
    """The committed forecast where one exists, otherwise a live one. Same rule as group C."""
    from services.forecast import _resolve
    row, from_ledger = _resolve({**inp, "horizon": horizon}, ctx)
    return row["body"]["distribution"], row, from_ledger


# ── E. risk and sizing ───────────────────────────────────────────────────────────────────────────

@service(
    endpoint="risk.size", group="E. Risk and sizing", price=0.05,
    title="The position consistent with your stated loss limit",
    short="Position size for your limit",
    summary="Takes the loss you are willing to accept and the confidence you want behind it, and "
            "returns the position size that satisfies both under the model's forecast distribution "
            "rather than under an assumed normal curve.",
    returns="Returns the position size in base units and notional, the loss at your confidence "
            "level, the expected shortfall beyond it, and how the answer changes across "
            "neighbouring confidence levels.",
    depth="Sized off the actual generated path distribution, so fat tails are priced in rather than "
          "assumed away — a normal-curve sizer systematically oversizes in crypto. This answers "
          "what is consistent with the limit you set. It has no opinion on whether to hold the "
          "position, and Horos does not sell one.",
    params=(SYMBOL, HORIZON,
            Param("account_equity", "number", "Total capital the limit applies to.", required=True),
            Param("max_loss_pct", "number", "Most you accept losing, as a percent of equity.",
                  default=1.0),
            Param("confidence", "number", "Confidence the limit holds, 0.5–0.999.", default=0.95),
            Param("direction", "string", "'long' or 'short'.", default="long")),
    example={"symbol": "BTC/USDT:USDT", "horizon": "24h", "account_equity": 100000,
             "max_loss_pct": 1.0, "confidence": 0.95},
)
def risk_size(inp: dict, ctx) -> dict:
    equity = float(inp.get("account_equity", 0))
    if equity <= 0:
        raise ServiceError("account_equity must be a positive number.", code="bad_equity")
    max_loss_pct = float(inp.get("max_loss_pct", 1.0))
    if not 0 < max_loss_pct <= 100:
        raise ServiceError("max_loss_pct must be between 0 and 100.", code="bad_max_loss")
    conf = float(inp.get("confidence", 0.95))
    if not 0.5 <= conf <= 0.999:
        raise ServiceError("confidence must be between 0.5 and 0.999.", code="bad_confidence")
    direction = str(inp.get("direction", "long")).lower()
    if direction not in ("long", "short"):
        raise ServiceError("direction must be 'long' or 'short'.", code="bad_direction")

    horizon = str(inp.get("horizon", "24h"))
    d, row, from_ledger = _distribution(inp, ctx, horizon)
    anchor = float(d["anchor_price"])
    terminal = np.asarray(d["samples"], dtype=float)
    # Per-unit profit and loss at the horizon, one figure per generated path.
    pnl = (terminal - anchor) if direction == "long" else (anchor - terminal)

    budget = equity * max_loss_pct / 100
    adverse = float(np.quantile(pnl, 1 - conf))     # the loss at the confidence level, per unit
    if adverse >= 0:
        return {
            "sizeable": False,
            "reason": (f"at {conf:.0%} confidence the {direction} position does not lose money over "
                       f"this horizon in the model's distribution, so a loss limit does not bind "
                       f"and no size follows from it. Size on your own conviction and exposure "
                       f"limits instead."),
            "loss_at_confidence_per_unit": round(adverse, 8),
            "provenance": _prov(row, from_ledger),
        }

    units = budget / abs(adverse)
    notional = units * anchor
    tail = pnl[pnl <= adverse]
    shortfall = float(tail.mean()) if tail.size else adverse

    sensitivity = {}
    for c in (0.90, 0.95, 0.99):
        q = float(np.quantile(pnl, 1 - c))
        sensitivity[f"{c:g}"] = {
            "loss_per_unit": round(q, 8),
            "units": round(budget / abs(q), 8) if q < 0 else None,
            "notional": round(budget / abs(q) * anchor, 2) if q < 0 else None,
        }

    return {
        "sizeable": True,
        "direction": direction,
        "anchor_price": anchor,
        "position": {
            "units": round(units, 8),
            "notional": round(notional, 2),
            "leverage_vs_equity": round(notional / equity, 3),
        },
        "loss_budget": {
            "max_loss_pct": max_loss_pct, "max_loss_absolute": round(budget, 2),
            "confidence": conf,
        },
        "at_this_size": {
            "loss_at_confidence": round(units * abs(adverse), 2),
            "expected_shortfall_beyond_it": round(units * abs(shortfall), 2),
            "shortfall_note": (f"if the {(1 - conf):.0%} case does occur, the average loss across "
                               f"those paths is this figure, not the limit — the limit is where the "
                               f"tail begins, not where it ends"),
            "worst_path_loss": round(units * abs(float(pnl.min())), 2) if pnl.min() < 0 else 0.0,
        },
        "sensitivity_to_confidence": sensitivity,
        "method": ("sized from the empirical distribution of generated paths, so tail behaviour is "
                   "taken from the model rather than assumed normal"),
        "not_advice": ("this is the size consistent with the limit you stated. It is not a "
                       "recommendation to take the position, and Horos publishes no directional "
                       "view."),
        "provenance": _prov(row, from_ledger),
    }


@service(
    endpoint="risk.stop", group="E. Risk and sizing", price=0.03,
    title="A stop placed where the distribution says, not on a round number",
    short="Distribution-based stop",
    summary="Places a stop at the level the forecast distribution implies for the stop-out "
            "probability you are willing to accept, and reports what that costs you when it is "
            "wrong.",
    returns="Returns the stop level for your chosen tolerance, the probability of being stopped, the "
            "same for several alternatives, and how often the model says price stops you out and "
            "then finishes in your favour anyway.",
    depth="The last figure is the one nobody quotes: the fraction of paths that touch your stop and "
          "then close where you wanted. That is the real cost of a tight stop and it is measurable "
          "on path data. Stop probabilities are read from path extremes, because a stop is hit "
          "intraday, not at the close.",
    params=(SYMBOL, HORIZON,
            Param("direction", "string", "'long' or 'short'.", default="long"),
            Param("stopout_tolerance", "number", "Acceptable probability of being stopped, 0–0.5.",
                  default=0.1)),
    example={"symbol": "BTC/USDT:USDT", "horizon": "24h", "direction": "long",
             "stopout_tolerance": 0.1},
)
def risk_stop(inp: dict, ctx) -> dict:
    direction = str(inp.get("direction", "long")).lower()
    if direction not in ("long", "short"):
        raise ServiceError("direction must be 'long' or 'short'.", code="bad_direction")
    tol = float(inp.get("stopout_tolerance", 0.1))
    if not 0 < tol < 0.5:
        raise ServiceError("stopout_tolerance must be between 0 and 0.5.", code="bad_tolerance")

    horizon = str(inp.get("horizon", "24h"))
    d, row, from_ledger = _distribution(inp, ctx, horizon)
    anchor = float(d["anchor_price"])
    terminal = np.asarray(d["samples"], dtype=float)
    lows = np.asarray(d.get("path_low") or terminal, dtype=float)
    highs = np.asarray(d.get("path_high") or terminal, dtype=float)
    extreme = lows if direction == "long" else highs

    # The level only `tol` of paths breach — read off the adverse extreme, not the close.
    level = float(np.quantile(extreme, tol)) if direction == "long" \
        else float(np.quantile(extreme, 1 - tol))

    def profile(stop: float) -> dict:
        stopped = (lows <= stop) if direction == "long" else (highs >= stop)
        favourable = (terminal > anchor) if direction == "long" else (terminal < anchor)
        whipsaw = int(np.sum(stopped & favourable))
        return {
            "level": round(stop, 8),
            "distance_pct": round(100 * (stop / anchor - 1), 4),
            "stopout_probability": round(float(np.mean(stopped)), 4),
            "whipsaw_paths": whipsaw,
            "whipsaw_probability": round(whipsaw / len(terminal), 4),
        }

    alternatives = {f"{t:g}": profile(
        float(np.quantile(extreme, t)) if direction == "long"
        else float(np.quantile(extreme, 1 - t))) for t in (0.05, 0.10, 0.20, 0.30)}

    chosen = profile(level)
    return {
        "direction": direction, "anchor_price": anchor, "horizon": horizon,
        "stop": chosen,
        "alternatives_by_tolerance": alternatives,
        "whipsaw_explained": (
            f"{chosen['whipsaw_probability']:.1%} of the model's paths touch this stop and then "
            f"finish in your favour anyway. That is the price of the tolerance you chose: a tighter "
            f"stop cuts losses faster and increases this number."),
        "measured_on": "path extremes, because a stop is hit intraday rather than at the close",
        "not_advice": ("this is where a stop sits for the stop-out rate you asked for. It is not a "
                       "recommendation to take or hold the position."),
        "provenance": _prov(row, from_ledger),
    }


@service(
    endpoint="risk.portfolio", group="E. Risk and sizing", price=0.05,
    title="Portfolio VaR with each position's real contribution",
    short="Portfolio VaR by position",
    summary="Computes value at risk and expected shortfall across a book of positions using the "
            "actual joint history of the assets, and decomposes the risk into what each position "
            "genuinely contributes rather than what it holds.",
    returns="Returns portfolio VaR and expected shortfall, each position's marginal and component "
            "contribution, the diversification benefit against the sum of standalone risks, and the "
            "correlation matrix behind it.",
    depth="Component contributions sum exactly to portfolio risk, which is what makes them "
          "actionable — a position holding 20% of notional can carry 50% of the risk, and the "
          "difference is invisible from position sizes alone. Every series is joined on timestamp "
          "so correlations are computed on genuinely simultaneous observations.",
    params=(Param("positions", "array", "Each: {symbol, quantity} — negative quantity is short.",
                  required=True),
            Param("lookback", "integer", "Candles of joint history.", default=720),
            Param("confidence", "number", "Confidence for VaR and shortfall.", default=0.95),
            Param("timeframe", "string", "Candle size.", default="1h")),
    example={"positions": [{"symbol": "BTC/USDT:USDT", "quantity": 1.5},
                           {"symbol": "ETH/USDT:USDT", "quantity": -10}],
             "confidence": 0.95},
)
def risk_portfolio(inp: dict, ctx) -> dict:
    import pandas as pd

    positions = inp.get("positions")
    if not isinstance(positions, list) or not positions:
        raise ServiceError("'positions' must be a non-empty list of {symbol, quantity}.",
                           code="bad_positions")
    if len(positions) > 20:
        raise ServiceError("at most 20 positions per call.", code="too_many_positions")
    conf = float(inp.get("confidence", 0.95))
    if not 0.5 <= conf <= 0.999:
        raise ServiceError("confidence must be between 0.5 and 0.999.", code="bad_confidence")
    tf = inp.get("timeframe", "1h")
    lookback = max(60, min(int(inp.get("lookback", 720)), 5000))

    series, values, unavailable = {}, {}, []
    for p in positions:
        if not isinstance(p, dict) or "symbol" not in p or "quantity" not in p:
            raise ServiceError("each position needs a 'symbol' and a 'quantity'.",
                               code="bad_position")
        sym = p["symbol"]
        try:
            qty = float(p["quantity"])
        except (TypeError, ValueError):
            raise ServiceError(f"quantity for {sym} is not a number.", code="bad_quantity") from None
        try:
            c = ctx.data.candles(sym, tf, limit=lookback)
        except Exception as e:                                     # noqa: BLE001
            unavailable.append({"symbol": sym, "reason": str(e)})
            continue
        idx = pd.to_datetime([r[0] for r in c.rows], unit="ms", utc=True)
        series[sym] = pd.Series(c.closes, index=idx).pct_change().dropna()
        values[sym] = qty * c.last_close

    if not series:
        raise ServiceError("none of the positions could be priced.", code="no_positions_priced",
                           unavailable=unavailable)

    frame = pd.DataFrame(series).dropna()
    if len(frame) < 30:
        raise ServiceError(
            f"only {len(frame)} timestamps are shared across these symbols — too few for a joint "
            f"risk estimate.", code="insufficient_overlap")

    v = np.array([values[s] for s in frame.columns])
    gross = float(np.abs(v).sum())
    net = float(v.sum())
    pnl = frame.to_numpy() @ v                       # portfolio P&L per historical candle
    var = float(np.quantile(pnl, 1 - conf))
    tail = pnl[pnl <= var]
    es = float(tail.mean()) if tail.size else var

    cov = frame.cov().to_numpy()
    port_var = float(v @ cov @ v)
    port_sd = math.sqrt(port_var) if port_var > 0 else 0.0
    marginal = (cov @ v) / port_sd if port_sd else np.zeros_like(v)
    component = v * marginal

    standalone = sum(abs(values[s]) * float(frame[s].std()) for s in frame.columns)
    diversification = 1 - (port_sd / standalone) if standalone else 0.0

    return {
        "positions_priced": len(frame.columns),
        "observations": int(len(frame)),
        "exposure": {"gross_notional": round(gross, 2), "net_notional": round(net, 2),
                     "per_position": {s: round(values[s], 2) for s in frame.columns}},
        "risk": {
            "confidence": conf,
            "value_at_risk": round(var, 2),
            "expected_shortfall": round(es, 2),
            "portfolio_sd_per_candle": round(port_sd, 2),
            "horizon": f"one {tf} candle",
        },
        "contribution": {
            s: {"notional": round(values[s], 2),
                "share_of_notional": round(abs(values[s]) / gross, 4) if gross else 0,
                "marginal_risk": round(float(marginal[i]), 6),
                "component_risk": round(float(component[i]), 2),
                "share_of_risk": round(float(component[i]) / port_sd, 4) if port_sd else 0}
            for i, s in enumerate(frame.columns)},
        "contribution_note": ("component risks sum to portfolio risk. Compare each position's share "
                              "of risk against its share of notional — where they diverge is where "
                              "the book is concentrated in a way position sizes do not show."),
        "diversification_benefit": round(diversification, 4),
        "correlation_matrix": {a: {b: round(float(frame[a].corr(frame[b])), 4)
                                   for b in frame.columns} for a in frame.columns},
        "positions_unavailable": unavailable,
        "method": "historical simulation on timestamp-aligned joint returns",
    }


@service(
    endpoint="risk.stress", group="E. Risk and sizing", price=0.05,
    title="What a shock actually does to this book",
    short="Scenario stress with betas",
    summary="Applies scenario shocks across a portfolio using each asset's measured beta to the "
            "asset you shock, so a correlated book moves the way it would really move rather than "
            "one position at a time.",
    returns="Returns the profit and loss under each scenario with the per-position breakdown, the "
            "worst case, the shock that would exhaust a loss limit you name, and the betas used.",
    depth="Naive stress testing shocks one asset and holds the rest still, which understates a "
            "crypto book badly because everything moves together. Betas are measured from the same "
            "joint history the correlation is computed on, and the reverse-stress figure answers the "
            "question people actually have: how big a move breaks me.",
    params=(Param("positions", "array", "Each: {symbol, quantity}.", required=True),
            Param("shocks_pct", "array", "Shocks to apply to the base asset.",
                  default=[-30, -20, -10, -5, 5, 10, 20]),
            Param("base", "string", "Asset the shock is applied to.", default="BTC/USDT:USDT"),
            Param("loss_limit", "number", "Loss to solve the reverse-stress shock for.",
                  default=None),
            Param("lookback", "integer", "Candles used to measure beta.", default=720)),
    example={"positions": [{"symbol": "BTC/USDT:USDT", "quantity": 1},
                           {"symbol": "SOL/USDT:USDT", "quantity": 100}],
             "shocks_pct": [-30, -20, -10, 10], "base": "BTC/USDT:USDT"},
)
def risk_stress(inp: dict, ctx) -> dict:
    import pandas as pd

    positions = inp.get("positions")
    if not isinstance(positions, list) or not positions:
        raise ServiceError("'positions' must be a non-empty list of {symbol, quantity}.",
                           code="bad_positions")
    base = inp.get("base", "BTC/USDT:USDT")
    shocks = inp.get("shocks_pct") or [-30, -20, -10, -5, 5, 10, 20]
    if not isinstance(shocks, list) or not shocks:
        raise ServiceError("'shocks_pct' must be a non-empty list of numbers.", code="bad_shocks")
    lookback = max(60, min(int(inp.get("lookback", 720)), 5000))
    tf = SCORED_TIMEFRAME

    series, values, unavailable = {}, {}, []
    for p in positions:
        sym = p.get("symbol")
        try:
            qty = float(p.get("quantity"))
        except (TypeError, ValueError):
            raise ServiceError(f"quantity for {sym} is not a number.", code="bad_quantity") from None
        try:
            c = ctx.data.candles(sym, tf, limit=lookback)
        except Exception as e:                                     # noqa: BLE001
            unavailable.append({"symbol": sym, "reason": str(e)})
            continue
        idx = pd.to_datetime([r[0] for r in c.rows], unit="ms", utc=True)
        series[sym] = pd.Series(c.closes, index=idx).pct_change().dropna()
        values[sym] = qty * c.last_close

    if base not in series:
        try:
            c = ctx.data.candles(base, tf, limit=lookback)
            idx = pd.to_datetime([r[0] for r in c.rows], unit="ms", utc=True)
            series[base] = pd.Series(c.closes, index=idx).pct_change().dropna()
        except Exception as e:                                     # noqa: BLE001
            raise ServiceError(f"the base asset {base} could not be fetched: {e}",
                               code="base_unavailable") from e

    frame = pd.DataFrame(series).dropna()
    if len(frame) < 30 or base not in frame.columns:
        raise ServiceError("not enough shared history to measure betas against the base asset.",
                           code="insufficient_overlap")

    bvar = float(frame[base].var())
    betas = {s: (float(frame[s].cov(frame[base]) / bvar) if bvar else 0.0)
             for s in values if s in frame.columns}

    scenarios = []
    for shock in shocks:
        try:
            shock = float(shock)
        except (TypeError, ValueError):
            continue
        rows, total = {}, 0.0
        for s, notional in values.items():
            move = betas.get(s, 0.0) * shock / 100
            pnl = notional * move
            rows[s] = {"beta": round(betas.get(s, 0.0), 4),
                       "implied_move_pct": round(100 * move, 4),
                       "pnl": round(pnl, 2)}
            total += pnl
        scenarios.append({"shock_pct": shock, "total_pnl": round(total, 2), "per_position": rows})

    gross = sum(abs(v) for v in values.values())
    worst = min(scenarios, key=lambda s: s["total_pnl"]) if scenarios else None

    reverse = None
    limit = inp.get("loss_limit")
    if limit is not None:
        try:
            limit = abs(float(limit))
        except (TypeError, ValueError):
            raise ServiceError("loss_limit must be a number.", code="bad_limit") from None
        # P&L is linear in the shock, so one unit of shock gives the slope directly.
        per_pct = sum(values[s] * betas.get(s, 0.0) / 100 for s in values)
        if per_pct:
            reverse = {"loss_limit": limit,
                       "shock_pct_that_reaches_it": round(-limit / per_pct, 4),
                       "note": "the move in the base asset that would produce exactly this loss, "
                               "holding the measured betas"}
        else:
            reverse = {"loss_limit": limit,
                       "note": "this book has no net exposure to the base asset, so no shock in it "
                               "reaches the limit"}

    return {
        "base_asset": base,
        "observations": int(len(frame)),
        "gross_notional": round(gross, 2),
        "betas": {s: round(b, 4) for s, b in betas.items()},
        "scenarios": scenarios,
        "worst_case": worst,
        "reverse_stress": reverse,
        "positions_unavailable": unavailable,
        "method": ("shocks propagate through measured betas rather than being applied to one asset "
                   "in isolation, because a crypto book moves together"),
        "caveat": ("betas are measured on the recent window and are not stable through a crisis — "
                   "correlations tend toward one exactly when a stress scenario matters most, so "
                   "treat these as a floor on the loss, not a ceiling."),
    }


@service(
    endpoint="portfolio.optimise", group="E. Risk and sizing", price=0.08,
    title="Efficient frontier, Black-Litterman and hierarchical risk parity",
    short="Efficient frontier and HRP",
    summary="Runs three genuinely different allocation methods over the same assets — mean-variance "
            "on the efficient frontier, hierarchical risk parity, and Black-Litterman if you supply "
            "views — and shows where they disagree.",
    returns="Returns the weights from each method, the expected return, volatility and Sharpe of "
            "each, the covariance treatment used, and the assets each method concentrates in.",
    depth="Mean-variance is famously unstable — small changes in expected returns produce wildly "
          "different weights — so Ledoit-Wolf shrinkage is applied to the covariance and "
          "hierarchical risk parity is run alongside as a method that needs no return forecast at "
          "all. Where the two agree the allocation is robust; where they diverge, mean-variance is "
          "probably fitting noise.",
    params=(Param("symbols", "array", "Two or more unified ccxt symbols.", required=True),
            Param("lookback", "integer", "Candles of joint history.", default=1000),
            Param("objective", "string", "'max_sharpe', 'min_volatility' or 'efficient_risk'.",
                  default="max_sharpe"),
            Param("target_volatility", "number", "Annual target for 'efficient_risk'.",
                  default=None),
            Param("views", "object", "Black-Litterman views: {symbol: expected annual return}.",
                  default=None)),
    example={"symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"],
             "objective": "max_sharpe", "lookback": 1000},
)
def portfolio_optimise(inp: dict, ctx) -> dict:
    import pandas as pd

    symbols = inp.get("symbols")
    if not isinstance(symbols, list) or len(symbols) < 2:
        raise ServiceError("'symbols' must be a list of at least two ccxt symbols.",
                           code="bad_symbols")
    if len(symbols) > 15:
        raise ServiceError("at most 15 symbols per optimisation.", code="too_many")
    objective = str(inp.get("objective", "max_sharpe"))
    if objective not in ("max_sharpe", "min_volatility", "efficient_risk"):
        raise ServiceError("objective must be max_sharpe, min_volatility or efficient_risk.",
                           code="bad_objective")
    lookback = max(120, min(int(inp.get("lookback", 1000)), 5000))
    ppy = 8760

    prices, unavailable = {}, []
    for s in symbols:
        try:
            c = ctx.data.candles(s, SCORED_TIMEFRAME, limit=lookback)
            idx = pd.to_datetime([r[0] for r in c.rows], unit="ms", utc=True)
            prices[s] = pd.Series(c.closes, index=idx)
        except Exception as e:                                     # noqa: BLE001
            unavailable.append({"symbol": s, "reason": str(e)})
    if len(prices) < 2:
        raise ServiceError(f"only {len(prices)} symbols could be priced; two are needed.",
                           code="insufficient_symbols", unavailable=unavailable)

    frame = pd.DataFrame(prices).dropna()
    if len(frame) < 60:
        raise ServiceError(f"only {len(frame)} shared timestamps — too few to optimise over.",
                           code="insufficient_overlap")

    from pypfopt import (BlackLittermanModel, EfficientFrontier, HRPOpt, expected_returns,
                         risk_models)

    mu = expected_returns.mean_historical_return(frame, frequency=ppy)
    # Ledoit-Wolf shrinkage: the sample covariance is badly conditioned at these sample sizes and
    # mean-variance amplifies exactly that error into extreme weights.
    S = risk_models.CovarianceShrinkage(frame, frequency=ppy).ledoit_wolf()

    results: dict = {}

    def clean(w):
        return {k: round(float(v), 6) for k, v in w.items() if abs(v) > 1e-6}

    try:
        ef = EfficientFrontier(mu, S)
        if objective == "max_sharpe":
            ef.max_sharpe()
        elif objective == "min_volatility":
            ef.min_volatility()
        else:
            target = inp.get("target_volatility")
            if target is None:
                raise ServiceError("efficient_risk needs a target_volatility.",
                                   code="missing_target")
            ef.efficient_risk(float(target))
        w = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance()
        results["mean_variance"] = {
            "objective": objective, "weights": clean(w),
            "expected_annual_return_pct": round(100 * ret, 3),
            "annual_volatility_pct": round(100 * vol, 3),
            "sharpe": round(sharpe, 4)}
    except Exception as e:                                         # noqa: BLE001
        results["mean_variance"] = {"failed": f"{type(e).__name__}: {e}"}

    try:
        hrp = HRPOpt(frame.pct_change().dropna())
        hrp.optimize()
        ret, vol, sharpe = hrp.portfolio_performance(frequency=ppy)
        results["hierarchical_risk_parity"] = {
            "weights": clean(hrp.clean_weights()),
            "expected_annual_return_pct": round(100 * ret, 3),
            "annual_volatility_pct": round(100 * vol, 3),
            "sharpe": round(sharpe, 4),
            "note": "needs no return forecast — allocates on the correlation structure alone"}
    except Exception as e:                                         # noqa: BLE001
        results["hierarchical_risk_parity"] = {"failed": f"{type(e).__name__}: {e}"}

    views = inp.get("views")
    if views:
        if not isinstance(views, dict):
            raise ServiceError("'views' must be an object of {symbol: expected annual return}.",
                               code="bad_views")
        try:
            unknown = [k for k in views if k not in frame.columns]
            if unknown:
                raise ServiceError(f"views reference symbols not in the portfolio: {unknown}",
                                   code="bad_views")
            bl = BlackLittermanModel(S, absolute_views={k: float(v) for k, v in views.items()})
            bl_mu, bl_S = bl.bl_returns(), bl.bl_cov()
            ef2 = EfficientFrontier(bl_mu, bl_S)
            ef2.max_sharpe()
            ret, vol, sharpe = ef2.portfolio_performance()
            results["black_litterman"] = {
                "views": views, "weights": clean(ef2.clean_weights()),
                "expected_annual_return_pct": round(100 * ret, 3),
                "annual_volatility_pct": round(100 * vol, 3),
                "sharpe": round(sharpe, 4)}
        except ServiceError:
            raise
        except Exception as e:                                     # noqa: BLE001
            results["black_litterman"] = {"failed": f"{type(e).__name__}: {e}"}
    else:
        results["black_litterman"] = {
            "skipped": "no views were supplied. Black-Litterman blends your own expected returns "
                       "into the market equilibrium; without views it reduces to the prior."}

    mv = results.get("mean_variance", {}).get("weights") or {}
    hr = results.get("hierarchical_risk_parity", {}).get("weights") or {}
    agreement = None
    if mv and hr:
        keys = set(mv) | set(hr)
        divergence = sum(abs(mv.get(k, 0) - hr.get(k, 0)) for k in keys) / 2
        agreement = {
            "total_weight_divergence": round(divergence, 4),
            "reading": ("the two methods broadly agree, which is a sign the allocation is driven by "
                        "the correlation structure rather than by noise in the return estimates"
                        if divergence < 0.25 else
                        "the two methods disagree substantially. Mean-variance is sensitive to "
                        "return estimates that are themselves uncertain, so the hierarchical result "
                        "is the more robust of the two here."),
        }

    return {
        "symbols": list(frame.columns),
        "observations": int(len(frame)),
        "annualisation_periods": ppy,
        "covariance": "Ledoit-Wolf shrinkage",
        "results": results,
        "method_agreement": agreement,
        "symbols_unavailable": unavailable,
        "caveat": ("expected returns estimated from historical means are noisy, and mean-variance "
                   "amplifies that noise. No forecast from the Horos model is used here — this is "
                   "allocation from realised statistics, and it is not a recommendation to hold "
                   "any of these assets."),
    }


# ── F. execution ─────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="exec.impact", group="F. Execution", price=0.05,
    title="What this order will cost before you send it",
    short="Order cost before sending",
    summary="Walks your order through the live order book level by level and reports the average "
            "fill price, the slippage against mid, and how far down the book the order reaches.",
    returns="Returns the expected average fill, slippage in basis points and absolute cost, the "
            "worst level touched, whether the book can absorb the order at all, and the same figures "
            "for several smaller sizes.",
    depth="A real walk through the resting book, not a square-root impact model fitted to someone "
          "else's market. If the order exhausts the visible book the answer says so rather than "
          "extrapolating — an extrapolated fill price beyond the book is a guess dressed as a "
          "measurement.",
    params=(SYMBOL,
            Param("side", "string", "'buy' or 'sell'.", required=True),
            Param("notional_usd", "number", "Order size in quote currency.", default=None),
            Param("quantity", "number", "Order size in base units, if you prefer.", default=None),
            Param("exchange", "string", "ccxt exchange id.", default="okx")),
    example={"symbol": "BTC/USDT:USDT", "side": "buy", "notional_usd": 250000},
)
def exec_impact(inp: dict, ctx) -> dict:
    side = str(inp.get("side", "")).lower()
    if side not in ("buy", "sell"):
        raise ServiceError("side must be 'buy' or 'sell'.", code="bad_side")
    ob = ctx.data.orderbook(inp["symbol"], depth=400, exchange=inp.get("exchange", "okx"))
    levels = ob["asks"] if side == "buy" else ob["bids"]
    mid = (ob["bids"][0][0] + ob["asks"][0][0]) / 2

    qty = inp.get("quantity")
    notional = inp.get("notional_usd")
    if qty is None and notional is None:
        raise ServiceError("supply either notional_usd or quantity.", code="missing_size")
    if qty is not None:
        qty = float(qty)
        if qty <= 0:
            raise ServiceError("quantity must be positive.", code="bad_size")
    else:
        notional = float(notional)
        if notional <= 0:
            raise ServiceError("notional_usd must be positive.", code="bad_size")
        qty = notional / mid

    def walk(target_qty: float) -> dict:
        remaining, cost, worst, touched = target_qty, 0.0, None, 0
        for price, size in levels:
            if remaining <= 0:
                break
            take = min(remaining, size)
            cost += take * price
            remaining -= take
            worst = price
            touched += 1
        filled = target_qty - remaining
        avg = cost / filled if filled else None
        return {
            "requested_quantity": round(target_qty, 10),
            "filled_quantity": round(filled, 10),
            "fully_filled": remaining <= 1e-12,
            "unfilled_quantity": round(max(remaining, 0), 10),
            "average_fill_price": round(avg, 8) if avg else None,
            "slippage_bps": round(10_000 * (avg - mid) / mid * (1 if side == "buy" else -1), 4)
            if avg else None,
            "slippage_cost_usd": round(abs(avg - mid) * filled, 2) if avg else None,
            "worst_price_touched": worst,
            "levels_consumed": touched,
            "notional_usd": round(cost, 2),
        }

    main = walk(qty)
    ladder = {f"{frac:g}x": walk(qty * frac) for frac in (0.1, 0.25, 0.5, 1.0)}

    if not main["fully_filled"]:
        main["warning"] = (
            f"the visible book holds only {main['filled_quantity']} of the {qty} requested. The "
            f"remainder cannot be priced from resting liquidity, and Horos does not extrapolate a "
            f"fill beyond the book — that would be a guess presented as a measurement. Work the "
            f"order, or check depth on other venues with market.venues.")

    return {
        "symbol": inp["symbol"], "exchange": ob["exchange"], "side": side,
        "mid_price": mid,
        "book_levels_available": ob["depth"],
        "impact": main,
        "by_size": ladder,
        "size_scaling_note": ("compare the slippage across the ladder — where it rises faster than "
                              "linearly, the order is exhausting resting liquidity and should be "
                              "worked rather than sent at once"),
        "not_included": ["exchange fees", "funding", "queue position and the chance the book moves "
                         "before the order lands", "hidden or iceberg liquidity"],
    }


@service(
    endpoint="exec.quality", group="F. Execution", price=0.03,
    title="Was that a fair fill",
    short="Was that a fair fill",
    summary="Grades a fill you have already made against the market at the time — the candle it "
            "landed in, the volume-weighted price over that window, and where your price sat in the "
            "actual traded range.",
    returns="Returns the benchmark prices, your slippage against each in basis points, your "
            "percentile within the candle's range, and a verdict on whether the fill was good, fair "
            "or poor.",
    depth="Nobody checks their fills, so nobody knows. Graded against three independent benchmarks "
          "— the candle VWAP proxy, the open and the close — because a fill can look good against "
          "one and poor against another, and which one is fair depends on when you decided to "
          "trade.",
    params=(SYMBOL,
            Param("fill_price", "number", "The price you were filled at.", required=True),
            Param("side", "string", "'buy' or 'sell'.", required=True),
            Param("timestamp_ms", "integer", "When the fill happened, epoch milliseconds.",
                  required=True),
            Param("quantity", "number", "Size filled, for the cost figures.", default=None),
            Param("timeframe", "string", "Candle size to grade against.", default="1m")),
    example={"symbol": "BTC/USDT:USDT", "fill_price": 65010.5, "side": "buy",
             "timestamp_ms": 1785150000000, "quantity": 0.5},
)
def exec_quality(inp: dict, ctx) -> dict:
    import time as _t

    side = str(inp.get("side", "")).lower()
    if side not in ("buy", "sell"):
        raise ServiceError("side must be 'buy' or 'sell'.", code="bad_side")
    try:
        fill = float(inp["fill_price"])
        ts = int(inp["timestamp_ms"])
    except (KeyError, TypeError, ValueError):
        raise ServiceError("fill_price and timestamp_ms are required and must be numbers.",
                           code="bad_fill") from None
    if fill <= 0:
        raise ServiceError("fill_price must be positive.", code="bad_fill")
    now_ms = int(_t.time() * 1000)
    if ts > now_ms:
        raise ServiceError("timestamp_ms is in the future; a fill cannot be graded before it "
                           "happens.", code="future_fill")

    tf = inp.get("timeframe", "1m")
    from core.markets import TIMEFRAME_SECONDS
    step_ms = TIMEFRAME_SECONDS.get(tf)
    if step_ms is None:
        raise ServiceError(f"unsupported timeframe {tf!r}.", code="bad_timeframe")
    step_ms *= 1000

    bar_open = (ts // step_ms) * step_ms
    candles = ctx.data.candles(inp["symbol"], tf, limit=5, until_ms=bar_open + step_ms * 2)
    bar = next((r for r in candles.rows if int(r[0]) == bar_open), None)
    if bar is None:
        raise ServiceError(
            f"no {tf} candle covering {ts} could be fetched. The fill may be older than the "
            f"exchange's history for this timeframe — try a larger timeframe.",
            code="no_candle_for_fill")

    o, h, l, c, vol = float(bar[1]), float(bar[2]), float(bar[3]), float(bar[4]), float(bar[5])
    typical = (h + l + c) / 3           # the standard VWAP proxy when trade data is not available
    rng = h - l
    position = (fill - l) / rng if rng else 0.5

    def bps(benchmark: float) -> float:
        raw = 10_000 * (fill - benchmark) / benchmark
        return round(raw if side == "buy" else -raw, 4)      # positive is always "worse for you"

    qty = inp.get("quantity")
    cost = None
    if qty is not None:
        try:
            qty = float(qty)
            cost = round((fill - typical) * qty * (1 if side == "buy" else -1), 2)
        except (TypeError, ValueError):
            raise ServiceError("quantity must be a number.", code="bad_quantity") from None

    vs_typical = bps(typical)
    verdict = ("a good fill — better than the average price traded in that window"
               if vs_typical < -1 else
               "a fair fill — within a basis point of the window's average price"
               if vs_typical <= 1 else
               "a poor fill — materially worse than the average price available in that window"
               if vs_typical > 5 else
               "a slightly unfavourable fill")

    return {
        "symbol": inp["symbol"], "side": side, "fill_price": fill, "timestamp_ms": ts,
        "candle": {"timeframe": tf, "open_ts": bar_open, "open": o, "high": h, "low": l,
                   "close": c, "volume": vol},
        "benchmarks": {
            "typical_price_vwap_proxy": round(typical, 8),
            "candle_open": o, "candle_close": c,
        },
        "slippage_bps": {
            "vs_typical_price": vs_typical,
            "vs_open": bps(o),
            "vs_close": bps(c),
            "convention": "positive is worse for you, on either side",
        },
        "position_in_range": {
            "percentile_rank_0_1": round(position, 4),
            "percentile_0_100": round(position * 100, 2),
            "reading": (f"your fill sat at the {position:.0%} point of the candle's traded range; "
                        f"for a {side} lower is better"),
        },
        "cost_vs_typical_usd": cost,
        "verdict": verdict,
        "benchmark_caveat": ("the typical price is the standard (high+low+close)/3 proxy for VWAP. "
                             "A true VWAP needs trade-by-trade data, which the exchange does not "
                             "publish for arbitrary historical windows, so this is stated as a "
                             "proxy rather than presented as VWAP."),
    }


@service(
    endpoint="exec.venue", group="F. Execution", price=0.03,
    title="Which venue is actually best for this size right now",
    short="Best venue for this size",
    summary="Walks the same order through every venue's live book at once and ranks them by the "
            "total cost you would actually pay, which is frequently not the venue showing the best "
            "top-of-book price.",
    returns="Returns each venue's average fill price, slippage and whether it can absorb the order, "
            "ranked by all-in cost, plus what routing to the best one saves against the worst and "
            "against the naive best-price choice.",
    depth="Best top-of-book and best execution are different things and diverge exactly when size "
          "matters. Each venue's book is walked for the real order, so a venue with a great quote "
          "and no depth behind it ranks where it belongs. Venues that cannot fill the order are "
          "reported as such rather than ranked on a partial fill.",
    params=(SYMBOL,
            Param("side", "string", "'buy' or 'sell'.", required=True),
            Param("notional_usd", "number", "Order size in quote currency.", required=True),
            Param("exchanges", "array", "Venues to compare. Reachable from the deployment region. Binance and Bybit geo-block US IP ranges, and Horos runs in us-east-1 — defaulting to venues that always fail would make every cross-venue answer a list of errors.",
                  default=["okx", "gate", "bitget", "kucoin"])),
    example={"symbol": "BTC/USDT:USDT", "side": "buy", "notional_usd": 500000},
)
def exec_venue(inp: dict, ctx) -> dict:
    side = str(inp.get("side", "")).lower()
    if side not in ("buy", "sell"):
        raise ServiceError("side must be 'buy' or 'sell'.", code="bad_side")
    try:
        notional = float(inp["notional_usd"])
    except (KeyError, TypeError, ValueError):
        raise ServiceError("notional_usd is required and must be a number.",
                           code="missing_size") from None
    if notional <= 0:
        raise ServiceError("notional_usd must be positive.", code="bad_size")
    names = inp.get("exchanges") or ["okx", "gate", "bitget", "kucoin"]

    ranked, unavailable, partial = [], [], []
    for name in names[:10]:
        try:
            r = ctx.registry_call("exec.impact", {"symbol": inp["symbol"], "side": side,
                                                  "notional_usd": notional, "exchange": name})
        except Exception as e:                                     # noqa: BLE001
            unavailable.append({"exchange": name, "reason": str(e)})
            continue
        row = {"exchange": name,
               "average_fill_price": r["impact"]["average_fill_price"],
               "slippage_bps": r["impact"]["slippage_bps"],
               "mid_price": r["mid_price"],
               "fully_filled": r["impact"]["fully_filled"],
               "filled_quantity": r["impact"]["filled_quantity"],
               "levels_consumed": r["impact"]["levels_consumed"]}
        (ranked if r["impact"]["fully_filled"] else partial).append(row)

    if not ranked:
        raise ServiceError(
            f"no venue can fill ${notional:,.0f} from its visible book. "
            f"{len(partial)} could fill part of it, {len(unavailable)} could not be reached.",
            code="no_venue_can_fill", partial=partial, unavailable=unavailable)

    ranked.sort(key=lambda r: r["average_fill_price"] if side == "buy"
                else -r["average_fill_price"])
    best, worst = ranked[0], ranked[-1]
    naive = min(ranked, key=lambda r: r["mid_price"]) if side == "buy" \
        else max(ranked, key=lambda r: r["mid_price"])
    qty = notional / best["mid_price"]

    return {
        "symbol": inp["symbol"], "side": side, "notional_usd": notional,
        "ranked": ranked,
        "best": best["exchange"],
        "saving_vs_worst_usd": round(abs(worst["average_fill_price"] -
                                         best["average_fill_price"]) * qty, 2),
        "saving_vs_best_top_of_book_usd": round(abs(naive["average_fill_price"] -
                                                    best["average_fill_price"]) * qty, 2),
        "best_top_of_book_venue": naive["exchange"],
        "routing_note": (
            f"the venue with the best mid price is {naive['exchange']}; the venue with the best "
            f"actual fill for this size is {best['exchange']}. They differ when the tighter quote "
            f"has less depth behind it — which is why this walks each book rather than comparing "
            f"top of book."
            if naive["exchange"] != best["exchange"] else
            f"{best['exchange']} has both the best quote and the best fill at this size"),
        "cannot_fill_this_size": partial,
        "unreachable": unavailable,
        "not_included": ["per-venue fee schedules and your fee tier", "transfer time and cost of "
                         "moving collateral between venues", "funding differentials"],
    }


def _prov(row: dict, from_ledger: bool) -> dict:
    from services.forecast import _provenance
    return _provenance(row, from_ledger, row.get("body", {}).get("symbol", ""))
