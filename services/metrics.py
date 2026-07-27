"""Group B — the deterministic tier. Eight services, $0.005–0.01.

Nothing here is a forecast. Every number is a function of candles that already exist, which makes
this the one tier a buyer can verify completely: the inputs are echoed back with their digest, the
method is named, and anyone with the same candles gets the same answer to the last decimal.

That is why it is priced low and why it matters. A caller who checks one of these and finds it exact
has a reason to believe the parts they cannot check.

Measured surface, not claimed: `ta.add_all_ta_features` produces **86 indicator columns** on a
5-column OHLCV frame (85 non-null on the final row; `trend_psar_up`/`down` are null by construction
until a reversal). FinanceToolkit's standalone risk models expose 37 functions across VaR, CVaR, EVaR
and the general risk module, and its performance module exposes 41.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from .registry import Param, ServiceError, service

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

SYMBOL = Param("symbol", "string", "Unified ccxt symbol, e.g. 'BTC/USDT:USDT'.", required=True)
TIMEFRAME = Param("timeframe", "string", "Candle size: 1m 5m 15m 30m 1h 4h 12h 1d 1w.", default="1h")
LOOKBACK = Param("lookback", "integer", "Candles to compute over, 30–5000.", default=500)
EXCHANGE = Param("exchange", "string", "ccxt exchange id.", default="okx")

# Hourly candles, so the annualisation factor is hours in a year. Crypto trades continuously — there
# is no 252-day convention to apply, and using one would overstate every annualised figure by ~40%.
PERIODS_PER_YEAR = {"1m": 525_600, "5m": 105_120, "15m": 35_040, "30m": 17_520,
                    "1h": 8_760, "2h": 4_380, "4h": 2_190, "6h": 1_460, "12h": 730,
                    "1d": 365, "1w": 52}


def _frame(inp: dict, ctx) -> tuple[pd.DataFrame, pd.Series, str, int]:
    """Candles as a frame, plus the simple-return series everything else is computed from."""
    tf = inp.get("timeframe", "1h")
    lookback = int(inp.get("lookback", 500))
    if not 30 <= lookback <= 5000:
        raise ServiceError("lookback must be between 30 and 5000 candles.", code="bad_lookback")
    c = ctx.data.candles(inp["symbol"], tf, limit=lookback, exchange=inp.get("exchange", "okx"))
    df = pd.DataFrame(c.rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    returns = df["close"].pct_change().dropna()
    return df, returns, tf, len(c)


def _clean(value) -> float | None:
    """NaN and inf are not JSON. A metric that could not be computed is reported as null."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(v) or math.isinf(v)) else round(v, 10)


def _provenance(df: pd.DataFrame, inp: dict, tf: str, n: int, method: str) -> dict:
    from core.crypto import digest
    return {
        "symbol": inp["symbol"], "exchange": inp.get("exchange", "okx"), "timeframe": tf,
        "candles_used": n,
        "first_candle_ts": int(df["ts"].iloc[0]), "last_candle_ts": int(df["ts"].iloc[-1]),
        "candles_sha256": digest([[int(r[0])] + [float(x) for x in r[1:]]
                                  for r in df[["ts", "open", "high", "low", "close",
                                               "volume"]].to_numpy().tolist()]),
        "annualisation_periods_per_year": PERIODS_PER_YEAR.get(tf),
        "method": method,
        "reproducible": ("fetch the same candles, hash them to the digest above, and every figure "
                         "here recomputes exactly"),
    }


# ── indicators ───────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="metrics.indicators", group="B. Deterministic metrics", price=0.005,
    title="Every technical indicator, computed at once",
    short="All technical indicators",
    summary="Computes the full technical indicator set over a candle window in one call — momentum, "
            "trend, volatility and volume families — instead of making you assemble and align them "
            "yourself.",
    returns="Returns the latest value of every indicator grouped by family, plus any that could not "
            "be computed on this window and the reason, and the candle digest to recompute from.",
    depth="86 indicator columns across five families, computed with the ta library on the exact "
          "candles echoed back. Indicators that legitimately have no value yet — a parabolic SAR "
          "before its first reversal, a long moving average on a short window — are returned as "
          "null with the reason, never as zero.",
    params=(SYMBOL, TIMEFRAME, LOOKBACK, EXCHANGE,
            Param("families", "array", "Restrict to some of: momentum, trend, volatility, volume, "
                                       "others.", default=None)),
    example={"symbol": "BTC/USDT:USDT", "timeframe": "1h", "lookback": 500},
)
def metrics_indicators(inp: dict, ctx) -> dict:
    import ta

    df, _, tf, n = _frame(inp, ctx)
    base = set(df.columns)
    enriched = ta.add_all_ta_features(df.copy(), open="open", high="high", low="low",
                                      close="close", volume="volume", fillna=False)
    cols = [c for c in enriched.columns if c not in base]
    last = enriched[cols].iloc[-1]

    wanted = inp.get("families")
    if wanted is not None:
        if not isinstance(wanted, list) or not wanted:
            raise ServiceError("'families' must be a non-empty list.", code="bad_families")
        allowed = {"momentum", "trend", "volatility", "volume", "others"}
        bad = [f for f in wanted if f not in allowed]
        if bad:
            raise ServiceError(f"unknown families {bad}; choose from {sorted(allowed)}.",
                               code="bad_families")
        cols = [c for c in cols if c.split("_")[0] in set(wanted)]

    grouped: dict[str, dict] = {}
    unavailable: list[dict] = []
    for c in cols:
        family = c.split("_")[0]
        value = _clean(last[c])
        if value is None:
            unavailable.append({"indicator": c, "reason":
                                "no value on the final candle — the indicator needs more history "
                                "than this window provides, or its condition has not yet occurred"})
        grouped.setdefault(family, {})[c] = value

    return {
        "indicators": grouped,
        "counts": {family: len(v) for family, v in grouped.items()},
        "total": sum(len(v) for v in grouped.values()),
        "unavailable": unavailable,
        "last_close": float(df["close"].iloc[-1]),
        "provenance": _provenance(df, inp, tf, n, "ta.add_all_ta_features"),
    }


# ── risk ─────────────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="metrics.risk", group="B. Deterministic metrics", price=0.01,
    title="Value at risk, tail risk and the drawdown family",
    short="VaR, CVaR and drawdown",
    summary="Computes value at risk four ways and conditional value at risk five ways — historical, "
            "Gaussian, Student-t, Laplace, logistic and extreme-value — alongside the full drawdown "
            "and tail-risk family.",
    returns="Returns every VaR and CVaR estimate side by side at the confidence you set, plus "
            "maximum drawdown and its duration, Ulcer index, downside deviation, tail ratio and "
            "conditional drawdown at risk.",
    depth="Presenting several estimators together is the point. Historical VaR says what actually "
          "happened; Gaussian VaR assumes normality that crypto returns violate; extreme-value VaR "
          "fits the tail specifically. Where they disagree sharply, that disagreement is the signal "
          "— and a single number would have hidden it.",
    params=(SYMBOL, TIMEFRAME, LOOKBACK, EXCHANGE,
            Param("confidence", "number", "Confidence level for VaR and CVaR, 0.5–0.999.",
                  default=0.95)),
    example={"symbol": "BTC/USDT:USDT", "lookback": 720, "confidence": 0.95},
)
def metrics_risk(inp: dict, ctx) -> dict:
    from financetoolkit.risk import cvar_model, evar_model, risk_model, var_model

    df, returns, tf, n = _frame(inp, ctx)
    conf = float(inp.get("confidence", 0.95))
    if not 0.5 <= conf <= 0.999:
        raise ServiceError("confidence must be between 0.5 and 0.999.", code="bad_confidence")
    alpha = 1 - conf

    # A metric that could not be computed records *why*. Returning a bare null for a failure is the
    # same defect as reporting an unreachable source as an absence: the caller cannot tell "this
    # market has no tail ratio" from "we called the function wrong". Three of these were silently
    # null until this helper started keeping the reason.
    failures: dict[str, str] = {}

    def call(name: str, fn, *a, **kw):
        try:
            return _clean(fn(*a, **kw))
        except Exception as e:                                     # noqa: BLE001
            failures[name] = f"{type(e).__name__}: {e}"
            return None

    var = {
        "historic": call("var.historic", var_model.get_var_historic, returns, alpha),
        "gaussian": call("var.gaussian", var_model.get_var_gaussian, returns, alpha),
        "student_t": call("var.student_t", var_model.get_var_studentt, returns, alpha),
        "extreme_value": call("var.evt", var_model.get_var_evt, returns, alpha),
    }
    cvar = {
        "historic": call("cvar.historic", cvar_model.get_cvar_historic, returns, alpha),
        "gaussian": call("cvar.gaussian", cvar_model.get_cvar_gaussian, returns, alpha),
        "student_t": call("cvar.student_t", cvar_model.get_cvar_studentt, returns, alpha),
        "laplace": call("cvar.laplace", cvar_model.get_cvar_laplace, returns, alpha),
        "logistic": call("cvar.logistic", cvar_model.get_cvar_logistic, returns, alpha),
    }
    entropic = {"gaussian": call("evar.gaussian", evar_model.get_evar_gaussian, returns, alpha)}

    drawdown = {
        "max_drawdown": call("drawdown.max", risk_model.get_max_drawdown, returns),
        "max_drawdown_duration_candles": call("drawdown.duration",
                                              risk_model.get_max_drawdown_duration, returns),
        "conditional_drawdown_at_risk": call("drawdown.cdar",
                                             risk_model.get_conditional_drawdown_at_risk,
                                             returns, alpha),
        "ulcer_index": call("drawdown.ulcer", risk_model.get_ui, returns),
    }
    r = returns.to_numpy()
    tail = {
        "downside_deviation": call("tail.downside_deviation",
                                   risk_model.get_downside_deviation, returns),
        "tail_ratio": call("tail.ratio", risk_model.get_tail_ratio, returns, alpha),
        "skew": call("tail.skew", risk_model.get_skewness, returns),
        "excess_kurtosis": call("tail.kurtosis", risk_model.get_kurtosis, returns),
        # Computed directly rather than through FinanceToolkit, whose versions take a calendar
        # `period` string ('weekly'|'monthly'|'quarterly'|'yearly') and resample against a
        # DatetimeIndex. That is an equity convention; on hourly crypto candles it would silently
        # answer a different question than the one asked.
        "mean_absolute_deviation": _clean(np.mean(np.abs(r - r.mean()))),
        "coefficient_of_variation": _clean(r.std() / r.mean()) if r.mean() else None,
    }
    if not r.mean():
        failures["tail.coefficient_of_variation"] = ("mean return is exactly zero, so the "
                                                     "coefficient of variation is undefined")

    spread = [v for v in var.values() if v is not None]
    disagreement = (max(spread) - min(spread)) if len(spread) > 1 else None
    computed = sum(1 for d in (var, cvar, entropic, drawdown, tail) for v in d.values()
                   if v is not None)

    return {
        "confidence": conf,
        "horizon": f"one {tf} candle",
        "value_at_risk": var,
        "conditional_value_at_risk": cvar,
        "entropic_value_at_risk": entropic,
        "drawdown": drawdown,
        "tail": tail,
        "estimator_disagreement": {
            "var_range": disagreement,
            "reading": ("the estimators broadly agree; the return distribution is close to the "
                        "assumptions they make" if disagreement is not None and spread
                        and abs(disagreement) < abs(np.mean(spread)) * 0.25 else
                        "the estimators disagree materially — the return distribution has fatter "
                        "tails than the parametric models assume, so the historical and "
                        "extreme-value figures are the ones to weight"
                        if disagreement is not None else
                        "not enough estimators computed to compare"),
        },
        "metrics_computed": computed,
        # Never omitted. An empty object is the claim that every metric computed; a populated one
        # names each that did not and why, so a null is never mistaken for a measured zero.
        "not_computed": failures,
        "how_to_read": (f"A VaR of -0.02 at {conf:.0%} confidence means: on {(1 - conf):.0%} of "
                        f"{tf} candles the return was worse than -2%. CVaR is the average loss on "
                        f"those candles, so it is always the more pessimistic figure."),
        "provenance": _provenance(df, inp, tf, n, "FinanceToolkit risk models on simple returns"),
    }


# ── performance ──────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="metrics.performance", group="B. Deterministic metrics", price=0.01,
    title="Risk-adjusted performance against a benchmark",
    short="Risk-adjusted performance",
    summary="Computes the risk-adjusted return family — Sharpe, Sortino, Calmar, Treynor, Jensen's "
            "alpha, information ratio and the capture ratios — measuring an asset against a "
            "benchmark you choose over the same window.",
    returns="Returns each ratio with the inputs behind it, the beta and alpha against the benchmark, "
            "upside and downside capture, and the annualisation factor used.",
    depth="Annualised with the hours in a year, not the 252 trading days equity convention. Crypto "
          "trades continuously; applying the equity factor overstates every annualised figure by "
          "about 40%, which is the sort of error that makes a whole page of numbers quietly wrong.",
    params=(SYMBOL, TIMEFRAME, LOOKBACK, EXCHANGE,
            Param("benchmark", "string", "Symbol to measure against.", default="BTC/USDT:USDT"),
            Param("risk_free_rate", "number", "Annual risk-free rate as a decimal.", default=0.0)),
    example={"symbol": "ETH/USDT:USDT", "benchmark": "BTC/USDT:USDT", "lookback": 720},
)
def metrics_performance(inp: dict, ctx) -> dict:
    from financetoolkit.performance import performance_model
    from financetoolkit.risk import risk_model

    df, returns, tf, n = _frame(inp, ctx)
    ppy = PERIODS_PER_YEAR.get(tf, 8760)
    rf_annual = float(inp.get("risk_free_rate", 0.0))
    rf = rf_annual / ppy
    bench_symbol = inp.get("benchmark", "BTC/USDT:USDT")

    bench_returns = None
    bench_note = None
    if bench_symbol and bench_symbol != inp["symbol"]:
        try:
            bc = ctx.data.candles(bench_symbol, tf, limit=n, exchange=inp.get("exchange", "okx"))
            bdf = pd.DataFrame(bc.rows, columns=["ts", "open", "high", "low", "close", "volume"])
            bdf.index = pd.to_datetime(bdf["ts"], unit="ms", utc=True)
            bench_returns = bdf["close"].pct_change().dropna()
            # Align on timestamp — a length-matched but time-misaligned pair produces a beta that is
            # arithmetically fine and completely meaningless.
            joined = pd.concat([returns.rename("a"), bench_returns.rename("b")], axis=1).dropna()
            returns_aligned, bench_returns = joined["a"], joined["b"]
        except Exception as e:                                     # noqa: BLE001
            bench_note = f"the benchmark {bench_symbol} could not be fetched: {e}"
            returns_aligned = returns
    else:
        returns_aligned = returns
        bench_note = "no benchmark was used; the symbol is its own benchmark"

    def call(fn, *a, **kw):
        try:
            return _clean(fn(*a, **kw))
        except Exception:                                          # noqa: BLE001
            return None

    excess = returns_aligned - rf
    sd = float(returns_aligned.std())
    downside = call(risk_model.get_downside_deviation, returns_aligned)
    max_dd = call(risk_model.get_max_drawdown, returns_aligned)
    mean = float(excess.mean())
    root = math.sqrt(ppy)

    ratios: dict = {
        "sharpe": _clean(mean / sd * root) if sd else None,
        "sortino": _clean(mean / downside * root) if downside else None,
        "calmar": _clean((returns_aligned.mean() * ppy) / abs(max_dd)) if max_dd else None,
        "gain_to_pain": call(performance_model.get_gain_to_pain_ratio, returns_aligned),
        "burke": call(performance_model.get_burke_ratio, returns_aligned, rf),
    }

    against: dict = {}
    if bench_returns is not None and len(bench_returns) > 2:
        beta = call(performance_model.get_beta, returns_aligned, bench_returns)
        against = {
            "benchmark": bench_symbol,
            "observations": int(len(bench_returns)),
            "beta": beta,
            "alpha": call(performance_model.get_alpha, returns_aligned, bench_returns),
            "jensens_alpha": call(performance_model.get_jensens_alpha, returns_aligned,
                                  bench_returns, rf, beta) if beta is not None else None,
            "correlation": _clean(returns_aligned.corr(bench_returns)),
            "treynor": _clean(mean / beta * ppy) if beta else None,
            "information_ratio": _clean(
                (returns_aligned - bench_returns).mean() /
                (returns_aligned - bench_returns).std() * root)
            if (returns_aligned - bench_returns).std() else None,
            "upside_capture": call(performance_model.get_upside_capture_ratio, returns_aligned,
                                   bench_returns),
            "downside_capture": call(performance_model.get_downside_capture_ratio, returns_aligned,
                                     bench_returns),
        }

    total_return = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)

    # An annualised ratio from a short window is not an estimate of anything durable, and a Sharpe
    # of 6 on 720 hourly candles is a trend that happened, not skill that will continue. The
    # standard error of an annualised Sharpe is roughly sqrt(1/N) in annual units, so it is reported
    # next to the ratio and the window is called out when it is too short to support the figure.
    years = len(returns_aligned) / ppy
    sharpe_se = _clean(math.sqrt((1 + 0.5 * (ratios["sharpe"] or 0) ** 2) / len(returns_aligned))
                       * root) if len(returns_aligned) > 1 else None
    reliability = {
        "observations": int(len(returns_aligned)),
        "window_in_years": round(years, 4),
        "sharpe_standard_error": sharpe_se,
        "warning": None,
    }
    if years < 0.25:
        reliability["warning"] = (
            f"this window is {years * 365:.0f} days. Annualising it multiplies both the return and "
            f"the ratio by {root:.0f}, so a strong recent trend produces a spectacular Sharpe that "
            f"says nothing about what comes next. Treat these ratios as a description of this "
            f"window, not as an estimate of future risk-adjusted return.")
    elif sharpe_se and ratios["sharpe"] and abs(ratios["sharpe"]) < 2 * sharpe_se:
        reliability["warning"] = ("the Sharpe ratio is smaller than twice its own standard error, "
                                  "so it is not distinguishable from zero on this sample.")

    return {
        "window_return_pct": round(100 * total_return, 4),
        "annualised_return_pct": round(100 * (returns_aligned.mean() * ppy), 4),
        "annualised_volatility_pct": round(100 * sd * root, 4),
        "risk_free_rate_annual": rf_annual,
        "ratios": ratios,
        "reliability": reliability,
        "against_benchmark": against or None,
        "benchmark_note": bench_note,
        "annualisation": {"periods_per_year": ppy,
                          "note": "hours in a year, not 252 trading days — crypto trades "
                                  "continuously"},
        "provenance": _provenance(df, inp, tf, n, "FinanceToolkit performance models"),
    }


# ── volatility ───────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="metrics.volatility", group="B. Deterministic metrics", price=0.01,
    title="Volatility five ways, including a fitted GARCH",
    short="Volatility, five methods",
    summary="Estimates volatility by five different methods — close-to-close, EWMA, Parkinson, "
            "Garman-Klass and a fitted GARCH(1,1) — because each uses different information and they "
            "disagree in informative ways.",
    returns="Returns every estimate annualised, the GARCH parameters and its next-period forecast, "
            "the volatility-of-volatility, and a reading of what the disagreement between the "
            "methods implies.",
    depth="Close-to-close ignores the intraday range entirely; Parkinson and Garman-Klass use the "
          "high and low and are several times more efficient per observation; GARCH captures the "
          "clustering the others assume away. When the range-based estimates greatly exceed "
          "close-to-close, the market is whipsawing within candles and closing flat.",
    params=(SYMBOL, TIMEFRAME, LOOKBACK, EXCHANGE,
            Param("ewma_lambda", "number", "Decay for the EWMA estimate.", default=0.94)),
    example={"symbol": "BTC/USDT:USDT", "timeframe": "1h", "lookback": 720},
)
def metrics_volatility(inp: dict, ctx) -> dict:
    df, returns, tf, n = _frame(inp, ctx)
    ppy = PERIODS_PER_YEAR.get(tf, 8760)
    root = math.sqrt(ppy)
    lam = float(inp.get("ewma_lambda", 0.94))
    if not 0.5 <= lam < 1.0:
        raise ServiceError("ewma_lambda must be between 0.5 and 1.", code="bad_lambda")

    log_ret = np.log(df["close"] / df["close"].shift(1)).dropna()
    close_to_close = float(log_ret.std())

    weights = np.array([(1 - lam) * lam ** i for i in range(len(log_ret))])[::-1]
    weights /= weights.sum()
    ewma = float(np.sqrt(np.sum(weights * (log_ret.to_numpy() - log_ret.mean()) ** 2)))

    hl = np.log(df["high"] / df["low"]).dropna()
    parkinson = float(np.sqrt((hl ** 2).mean() / (4 * math.log(2))))

    co = np.log(df["close"] / df["open"])
    gk = float(np.sqrt(np.mean(0.5 * hl ** 2 - (2 * math.log(2) - 1) * co ** 2)))

    garch: dict = {"fitted": False}
    try:
        from arch import arch_model
        # Percentage returns: arch warns and rescales otherwise, and the rescaling changes the
        # reported parameters in a way that is easy to misread.
        res = arch_model(log_ret * 100, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        forecast = res.forecast(horizon=1, reindex=False)
        next_var = float(forecast.variance.iloc[-1, 0])
        garch = {
            "fitted": True, "model": "GARCH(1,1), Student-t errors",
            "omega": _clean(res.params.get("omega")),
            "alpha": _clean(res.params.get("alpha[1]")),
            "beta": _clean(res.params.get("beta[1]")),
            "persistence": _clean((res.params.get("alpha[1]", 0) + res.params.get("beta[1]", 0))),
            "next_period_sd": _clean(math.sqrt(next_var) / 100),
            "next_period_annualised_pct": _clean(100 * math.sqrt(next_var) / 100 * root),
            "log_likelihood": _clean(res.loglikelihood),
        }
        if garch["persistence"] and garch["persistence"] > 0.99:
            garch["note"] = ("persistence is very close to 1: shocks decay extremely slowly and the "
                             "unconditional variance is barely identified on this window")
    except Exception as e:                                         # noqa: BLE001
        garch = {"fitted": False, "reason": f"the GARCH fit did not converge: {e}"}

    rolling = log_ret.rolling(24).std().dropna()
    estimates = {"close_to_close": close_to_close, "ewma": ewma,
                 "parkinson": parkinson, "garman_klass": gk}
    ratio = parkinson / close_to_close if close_to_close else None

    return {
        "estimates_per_candle": {k: _clean(v) for k, v in estimates.items()},
        "estimates_annualised_pct": {k: _clean(100 * v * root) for k, v in estimates.items()},
        "garch": garch,
        "volatility_of_volatility": _clean(rolling.std()) if len(rolling) > 2 else None,
        "range_to_close_ratio": _clean(ratio),
        "reading": (
            "the range-based estimates greatly exceed close-to-close: price is moving a long way "
            "inside candles and closing near where it opened — whipsaw"
            if ratio and ratio > 1.5 else
            "close-to-close exceeds the range-based estimates, which is unusual and usually means "
            "gaps between candles rather than intra-candle movement"
            if ratio and ratio < 0.7 else
            "the estimators broadly agree" if ratio else "no comparison was possible"),
        "provenance": _provenance(df, inp, tf, n,
                                  "close-to-close, EWMA, Parkinson, Garman-Klass, arch GARCH(1,1)"),
    }


# ── distribution ─────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="metrics.distribution", group="B. Deterministic metrics", price=0.01,
    title="What shape the return distribution actually has",
    short="Return distribution shape",
    summary="Tests whether returns are anywhere near normal — skew, kurtosis, Jarque-Bera, "
            "autocorrelation at several lags and the Hurst exponent — which decides whether any "
            "model assuming normality can be trusted on this market.",
    returns="Returns each statistic with its interpretation, the normality test result with its "
            "p-value, the autocorrelation profile, and an explicit verdict on whether normal-based "
            "risk models apply here.",
    depth="This is the service that tells you whether the other services' parametric estimates are "
          "safe to use. Crypto returns almost always fail normality with heavy positive excess "
          "kurtosis, which is exactly why metrics.risk shows historical and extreme-value VaR "
          "alongside the Gaussian one rather than instead of it.",
    params=(SYMBOL, TIMEFRAME, LOOKBACK, EXCHANGE,
            Param("lags", "array", "Autocorrelation lags to report.", default=[1, 2, 4, 12, 24])),
    example={"symbol": "BTC/USDT:USDT", "lookback": 1000},
)
def metrics_distribution(inp: dict, ctx) -> dict:
    from scipy import stats

    df, returns, tf, n = _frame(inp, ctx)
    r = returns.to_numpy()
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r))                # excess kurtosis: 0 is normal
    jb_stat, jb_p = stats.jarque_bera(r)
    lags = inp.get("lags") or [1, 2, 4, 12, 24]
    if not isinstance(lags, list):
        raise ServiceError("'lags' must be a list of integers.", code="bad_lags")

    autocorr = {}
    for lag in lags[:20]:
        try:
            lag = int(lag)
        except (TypeError, ValueError):
            continue
        if 0 < lag < len(r) // 2:
            autocorr[str(lag)] = _clean(pd.Series(r).autocorr(lag=lag))

    from services.forecast import _hurst
    hurst = _hurst(np.log(df["close"] / df["close"].shift(1)).dropna().to_numpy())
    normal = bool(jb_p > 0.05)

    return {
        "observations": int(len(r)),
        "moments": {
            "mean": _clean(r.mean()), "std": _clean(r.std()),
            "skew": _clean(skew), "excess_kurtosis": _clean(kurt),
            "min": _clean(r.min()), "max": _clean(r.max()),
        },
        "normality": {
            "test": "Jarque-Bera",
            "statistic": _clean(jb_stat), "p_value": _clean(jb_p),
            "normal_at_5pct": normal,
            "verdict": ("returns are not distinguishable from normal on this window, so "
                        "Gaussian risk estimates are reasonable here"
                        if normal else
                        "returns are decisively non-normal, so Gaussian VaR will understate tail "
                        "risk — weight the historical and extreme-value estimates instead"),
        },
        "tails": {
            "excess_kurtosis": _clean(kurt),
            "reading": ("far heavier tails than a normal distribution — large moves are much more "
                        "common than a Gaussian model expects" if kurt > 1 else
                        "tails close to normal" if abs(kurt) <= 1 else
                        "lighter tails than normal"),
            "skew_reading": ("a long left tail — large falls outsize large rises" if skew < -0.2
                             else "a long right tail — large rises outsize large falls" if skew > 0.2
                             else "broadly symmetric"),
        },
        "autocorrelation": autocorr,
        "autocorrelation_reading": (
            "there is measurable serial dependence at short lags"
            if any(abs(v or 0) > 2 / math.sqrt(len(r)) for v in autocorr.values())
            else "no serial dependence beyond what noise would produce at this sample size"),
        "hurst_exponent": _clean(hurst),
        "provenance": _provenance(df, inp, tf, n,
                                  "scipy moments and Jarque-Bera; rescaled-range Hurst"),
    }


# ── correlation ──────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="metrics.correlation", group="B. Deterministic metrics", price=0.01,
    title="Cross-asset correlation and beta, time-aligned",
    short="Correlation and beta",
    summary="Builds the correlation and beta matrix across several assets over the same window, "
            "aligning every series on its actual timestamps rather than assuming the candle counts "
            "line up.",
    returns="Returns the full correlation matrix, each asset's beta to the one you nominate, how "
            "correlation in the recent sub-window differs from the whole, and the number of aligned "
            "observations behind every figure.",
    depth="Timestamp alignment is where this normally goes wrong. Two 500-candle series from "
          "different venues can cover different periods, and correlating them by position produces "
          "a number that is arithmetically valid and completely meaningless. Every pair here is "
          "joined on timestamp and the surviving observation count is reported.",
    params=(Param("symbols", "array", "Two or more unified ccxt symbols.", required=True),
            TIMEFRAME, LOOKBACK, EXCHANGE,
            Param("base", "string", "Symbol to compute betas against. Defaults to the first.",
                  default=None)),
    example={"symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"], "lookback": 500},
)
def metrics_correlation(inp: dict, ctx) -> dict:
    symbols = inp.get("symbols")
    if not isinstance(symbols, list) or len(symbols) < 2:
        raise ServiceError("'symbols' must be a list of at least two ccxt symbols.",
                           code="bad_symbols")
    if len(symbols) > 12:
        raise ServiceError("at most 12 symbols can be compared in one call.", code="too_many")
    tf = inp.get("timeframe", "1h")
    lookback = int(inp.get("lookback", 500))
    base = inp.get("base") or symbols[0]

    series, unavailable = {}, []
    for s in symbols:
        try:
            c = ctx.data.candles(s, tf, limit=lookback, exchange=inp.get("exchange", "okx"))
            idx = pd.to_datetime([r[0] for r in c.rows], unit="ms", utc=True)
            series[s] = pd.Series(c.closes, index=idx).pct_change().dropna()
        except Exception as e:                                     # noqa: BLE001
            unavailable.append({"symbol": s, "reason": str(e)})
    if len(series) < 2:
        raise ServiceError(
            f"only {len(series)} of {len(symbols)} symbols could be fetched; a correlation needs "
            f"two.", code="insufficient_symbols", unavailable=unavailable)

    frame = pd.DataFrame(series).dropna()
    if len(frame) < 10:
        raise ServiceError(
            f"only {len(frame)} timestamps are shared across all symbols — too few to correlate. "
            f"The symbols may trade on different schedules or venues.",
            code="insufficient_overlap")

    corr = frame.corr()
    betas = {}
    if base in frame.columns:
        bvar = frame[base].var()
        for s in frame.columns:
            betas[s] = _clean(frame[s].cov(frame[base]) / bvar) if bvar else None

    recent = frame.tail(max(24, len(frame) // 4))
    recent_corr = recent.corr()
    shifts = []
    cols = list(frame.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            delta = float(recent_corr.loc[a, b] - corr.loc[a, b])
            if abs(delta) > 0.2:
                shifts.append({"pair": [a, b], "full_window": _clean(corr.loc[a, b]),
                               "recent": _clean(recent_corr.loc[a, b]), "change": _clean(delta)})

    return {
        "symbols": list(frame.columns),
        "aligned_observations": int(len(frame)),
        "timeframe": tf,
        "correlation_matrix": {a: {b: _clean(corr.loc[a, b]) for b in frame.columns}
                               for a in frame.columns},
        "beta_to": {"base": base, "betas": betas} if betas else None,
        "recent_window_observations": int(len(recent)),
        "correlation_shifts": shifts,
        "shift_reading": (f"{len(shifts)} pair(s) show a correlation shift of more than 0.2 between "
                          f"the full window and the recent one" if shifts else
                          "correlations in the recent window are consistent with the full window"),
        "symbols_unavailable": unavailable,
        "alignment_note": ("every series is joined on its actual candle timestamps, so the "
                           "observation count above is what all symbols genuinely share"),
    }


# ── liquidity ────────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="metrics.liquidity", group="B. Deterministic metrics", price=0.01,
    title="How liquid this market actually is",
    short="Liquidity, book and traded",
    summary="Measures liquidity from both sides — the resting book and realised trading — combining "
            "spread, depth, Amihud illiquidity, Roll's effective spread and turnover into one "
            "picture.",
    returns="Returns each liquidity measure with what it means, the notional you could move within "
            "several price impacts, and a verdict on the size this market supports.",
    depth="Book depth alone overstates liquidity because resting orders vanish under pressure; "
          "realised measures alone miss the current state. Amihud illiquidity is price impact per "
          "unit of volume from actual candles, Roll's measure infers the effective spread from "
          "return autocovariance, and both are computed alongside the live book.",
    params=(SYMBOL, TIMEFRAME, LOOKBACK, EXCHANGE),
    example={"symbol": "BTC/USDT:USDT", "timeframe": "1h", "lookback": 500},
)
def metrics_liquidity(inp: dict, ctx) -> dict:
    df, returns, tf, n = _frame(inp, ctx)
    notional = df["close"] * df["volume"]
    amihud = float((returns.abs() / notional.iloc[1:].replace(0, np.nan)).mean()) * 1e9

    # Roll's effective spread: 2*sqrt(-cov(r_t, r_{t-1})), defined only for negative autocovariance.
    cov = float(pd.Series(returns.to_numpy()).autocorr(lag=1) * returns.var())
    roll = 2 * math.sqrt(-cov) if cov < 0 else None

    book: dict
    try:
        ob = ctx.registry_call("market.orderbook", {"symbol": inp["symbol"], "depth": 400,
                                                    "exchange": inp.get("exchange", "okx")})
        book = {"spread_bps": ob["spread_bps"], "imbalance": ob["imbalance"],
                "depth_notional_usd": ob["depth_notional_usd"],
                "book_reach_bps": ob["book_reach_bps"]}
    except Exception as e:                                         # noqa: BLE001
        book = {"unavailable": str(e)}

    mean_notional = float(notional.mean())
    return {
        "realised": {
            "mean_candle_notional_usd": round(mean_notional, 2),
            "median_candle_notional_usd": round(float(notional.median()), 2),
            "amihud_illiquidity_x1e9": _clean(amihud),
            "amihud_reading": ("price moves a lot per unit of volume — thin"
                               if amihud > 1 else "price absorbs volume well — deep"),
            "roll_effective_spread": _clean(roll),
            "roll_note": (None if roll is not None else
                          "Roll's measure is undefined here: return autocovariance is positive, "
                          "which the model does not admit. That usually means trending rather than "
                          "bid-ask bounce."),
            "turnover_per_candle": _clean(df["volume"].mean()),
        },
        "book": book,
        "verdict": _liquidity_verdict(mean_notional, book),
        "provenance": _provenance(df, inp, tf, n,
                                  "Amihud illiquidity, Roll effective spread, live order book"),
    }


def _liquidity_verdict(mean_notional: float, book: dict) -> str:
    if "unavailable" in book:
        return (f"average candle turnover is ${mean_notional:,.0f}; the live book could not be read, "
                f"so this is based on realised volume alone")
    spread = book.get("spread_bps", 0)
    if mean_notional > 5e7 and spread < 2:
        return "deep and tight — supports institutional size without material impact"
    if mean_notional > 5e6 and spread < 10:
        return "liquid — supports meaningful size with modest impact"
    if mean_notional > 5e5:
        return "moderately liquid — size should be worked rather than sent at once"
    return "thin — expect material impact on anything but small size"


# ── microstructure ───────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="metrics.microstructure", group="B. Deterministic metrics", price=0.01,
    title="What the candle shapes reveal",
    short="Candle microstructure",
    summary="Reads structure out of the candles themselves — body versus wick proportions, gap "
            "behaviour, run lengths, the intraday range profile and where closes sit inside their "
            "own range.",
    returns="Returns the body-to-range profile, upper and lower wick asymmetry, the distribution of "
            "consecutive up and down runs, the close-location value, and what each implies about "
            "who is in control.",
    depth="Candle geometry carries information that closing prices discard. Persistent long upper "
          "wicks mean rallies are being sold into; a close-location value consistently near the "
          "high means buyers control the settle. Run-length distributions are compared against what "
          "a fair coin would produce, so persistence claims are measured rather than eyeballed.",
    params=(SYMBOL, TIMEFRAME, LOOKBACK, EXCHANGE),
    example={"symbol": "BTC/USDT:USDT", "timeframe": "1h", "lookback": 500},
)
def metrics_microstructure(inp: dict, ctx) -> dict:
    df, returns, tf, n = _frame(inp, ctx)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    upper = df["high"] - df[["open", "close"]].max(axis=1)
    lower = df[["open", "close"]].min(axis=1) - df["low"]
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng

    up = (df["close"] > df["open"]).to_numpy()
    runs, current, direction = [], 0, None
    for u in up:
        if u == direction:
            current += 1
        else:
            if direction is not None:
                runs.append((direction, current))
            direction, current = bool(u), 1
    runs.append((direction, current))
    up_runs = [ln for d, ln in runs if d]
    down_runs = [ln for d, ln in runs if not d]
    expected_longest = math.log2(max(len(up), 2))

    return {
        "candles": int(len(df)),
        "body": {
            "mean_body_to_range": _clean((body / rng).mean()),
            "reading": ("candles close far from where they opened — directional conviction"
                        if (body / rng).mean() > 0.6 else
                        "candles close near where they opened — indecision, or two-sided flow"),
        },
        "wicks": {
            "mean_upper_to_range": _clean((upper / rng).mean()),
            "mean_lower_to_range": _clean((lower / rng).mean()),
            "asymmetry": _clean((upper / rng).mean() - (lower / rng).mean()),
            "reading": ("upper wicks dominate — rallies are being sold into"
                        if (upper / rng).mean() > (lower / rng).mean() * 1.2 else
                        "lower wicks dominate — dips are being bought"
                        if (lower / rng).mean() > (upper / rng).mean() * 1.2 else
                        "wicks are broadly symmetric"),
        },
        "close_location_value": {
            "mean": _clean(clv.mean()),
            "reading": ("closes cluster near candle highs — buyers control the settle"
                        if clv.mean() > 0.2 else
                        "closes cluster near candle lows — sellers control the settle"
                        if clv.mean() < -0.2 else
                        "closes sit mid-range on average"),
        },
        "runs": {
            "up_runs": len(up_runs), "down_runs": len(down_runs),
            "longest_up": max(up_runs) if up_runs else 0,
            "longest_down": max(down_runs) if down_runs else 0,
            "mean_up_length": _clean(np.mean(up_runs)) if up_runs else None,
            "mean_down_length": _clean(np.mean(down_runs)) if down_runs else None,
            "expected_longest_under_coin_flip": round(expected_longest, 2),
            "reading": ("run lengths exceed what a fair coin would produce — moves persist"
                        if max(max(up_runs or [0]), max(down_runs or [0])) > expected_longest * 1.6
                        else "run lengths are consistent with a fair coin — no measurable "
                             "persistence in direction"),
        },
        "range": {
            "mean_range_pct": _clean(100 * (rng / df["close"]).mean()),
            "max_range_pct": _clean(100 * (rng / df["close"]).max()),
        },
        "provenance": _provenance(df, inp, tf, n, "candle geometry and run-length analysis"),
    }


# ── drawdown ─────────────────────────────────────────────────────────────────────────────────────

@service(
    endpoint="metrics.drawdown", group="B. Deterministic metrics", price=0.01,
    title="Every drawdown, not just the worst one",
    short="Full drawdown history",
    summary="Reconstructs the full drawdown history over a window — every peak-to-trough episode, "
            "how deep it went, how long it lasted and whether it ever recovered — rather than "
            "reporting only the single deepest figure.",
    returns="Returns the ranked drawdown episodes with their dates and durations, the current "
            "drawdown and its age, time spent underwater, and the recovery statistics.",
    depth="A single maximum-drawdown number hides whether the market takes one big hit and recovers "
          "or bleeds repeatedly. The episode list, the underwater fraction and the recovery-time "
          "distribution are what actually inform position sizing, and they are computed from the "
          "same candles echoed back.",
    params=(SYMBOL, TIMEFRAME, LOOKBACK, EXCHANGE,
            Param("threshold_pct", "number", "Ignore drawdowns shallower than this.", default=1.0),
            Param("top", "integer", "How many episodes to return.", default=10)),
    example={"symbol": "BTC/USDT:USDT", "lookback": 2000, "threshold_pct": 2.0},
)
def metrics_drawdown(inp: dict, ctx) -> dict:
    df, returns, tf, n = _frame(inp, ctx)
    threshold = float(inp.get("threshold_pct", 1.0)) / 100
    top = max(1, min(int(inp.get("top", 10)), 100))

    close = df["close"].to_numpy()
    ts = df["ts"].to_numpy()
    peak, peak_i = close[0], 0
    episodes, in_dd, trough, trough_i, start_i = [], False, close[0], 0, 0

    for i, price in enumerate(close):
        if price >= peak:
            if in_dd and (peak - trough) / peak >= threshold:
                episodes.append({
                    "depth_pct": round(-100 * (peak - trough) / peak, 4),
                    "peak_ts": int(ts[start_i]), "trough_ts": int(ts[trough_i]),
                    "recovered_ts": int(ts[i]),
                    "candles_to_trough": trough_i - start_i,
                    "candles_to_recover": i - trough_i,
                    "recovered": True})
            peak, peak_i, in_dd = price, i, False
        else:
            if not in_dd:
                in_dd, trough, trough_i, start_i = True, price, i, peak_i
            if price < trough:
                trough, trough_i = price, i

    current = None
    if in_dd and (peak - trough) / peak >= threshold:
        current = {
            "depth_pct": round(-100 * (peak - close[-1]) / peak, 4),
            "max_depth_pct": round(-100 * (peak - trough) / peak, 4),
            "peak_ts": int(ts[start_i]), "trough_ts": int(ts[trough_i]),
            "candles_since_peak": len(close) - 1 - start_i,
            "recovered": False}

    ranked = sorted(episodes, key=lambda e: e["depth_pct"])[:top]
    running_peak = np.maximum.accumulate(close)
    underwater = float(np.mean(close < running_peak))
    recoveries = [e["candles_to_recover"] for e in episodes]

    # The worst drawdown must include the one still open. A market that fell 29% and never recovered
    # produces *zero* completed episodes, and reporting "episodes_found: 0, worst: null" next to a
    # market sitting 21% below its high is true, useless, and read as the opposite of the truth.
    # `worst` is therefore the deepest of everything seen, completed or not.
    max_dd_pct = round(100 * float((close / running_peak - 1).min()), 4)
    candidates = list(ranked) + ([current] if current else [])
    worst = min(candidates, key=lambda e: e.get("max_depth_pct", e["depth_pct"])) \
        if candidates else None

    if current and not episodes:
        summary = (f"one drawdown, still open: {current['max_depth_pct']:.2f}% at its worst and "
                   f"{current['depth_pct']:.2f}% now, running for "
                   f"{current['candles_since_peak']} candles without recovering. There are no "
                   f"completed episodes because price never regained its prior peak in this window.")
    elif current:
        summary = (f"{len(episodes)} completed drawdown(s) past {threshold:.1%}, worst "
                   f"{ranked[0]['depth_pct']:.2f}%, plus one still open at "
                   f"{current['depth_pct']:.2f}%.")
    elif episodes:
        summary = (f"{len(episodes)} completed drawdown(s) past {threshold:.1%}, worst "
                   f"{ranked[0]['depth_pct']:.2f}%. Price is at or near its window high.")
    else:
        summary = (f"no drawdown deeper than {threshold:.1%} occurred in this window; the maximum "
                   f"was {max_dd_pct:.2f}%.")

    return {
        "summary": summary,
        "max_drawdown_pct": max_dd_pct,
        "worst": worst,
        "in_drawdown_now": current is not None,
        "current_drawdown": current,
        "completed_episodes_found": len(episodes),
        "completed_episodes": ranked,
        "threshold_pct": float(inp.get("threshold_pct", 1.0)),
        "time_underwater_fraction": round(underwater, 4),
        "time_underwater_reading": f"price was below a prior peak {underwater:.0%} of this window",
        "recovery": {
            "count": len(recoveries),
            "median_candles": _clean(np.median(recoveries)) if recoveries else None,
            "longest_candles": max(recoveries) if recoveries else None,
        } if recoveries else {
            "count": 0,
            "note": ("no drawdown past the threshold recovered within this window. That is not the "
                     "same as the market being calm — check `current_drawdown` and "
                     "`max_drawdown_pct`.")},
        "provenance": _provenance(df, inp, tf, n,
                                  "peak-to-trough episode reconstruction on closing prices"),
    }
