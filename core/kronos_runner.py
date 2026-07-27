"""Kronos, serving the distribution instead of the mean.

Kronos is the only open foundation model for K-lines. It generates `sample_count` independent future
paths by autoregressive sampling — and then, in `auto_regressive_inference`, throws them away:

    model/kronos.py:465-467
        z = z.reshape(-1, sample_count, z.size(1), z.size(2))
        preds = z.cpu().numpy()
        preds = np.mean(preds, axis=1)      # <- the distribution, averaged into a line

Every caller of the library out of the box receives a single path. The bands, the touch
probabilities, the tail percentile that makes surprise detection possible and the volatility
forecast all live in the fan that line discards, so recovering it is the precondition for the entire
product.

**How this recovers it without modifying the library.** With `sample_count=1` that mean is a
reduction over an axis of length one — the identity. And `predict_batch` runs a whole batch of series
through a single forward pass, sampling independently per batch element via `torch.multinomial`. So
feeding N *identical copies* of one series with `sample_count=1` yields N independent draws from the
same predictive distribution, in one batched pass, using the published library exactly as written.

No vendored fork, no monkeypatch, nothing to keep in sync with upstream. The library was always able
to do this; its default just happens to hide it.

**What the model cannot see.** OHLCV and calendar features, nothing else. No funding, no open
interest, no basis, no order book, no news. Every response says so, because a forecast that does not
disclose its blind spots is being sold as more than it is.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("horos.kronos")

REPO_ROOT = Path(__file__).resolve().parents[2]

# Both are configurable. Deriving them from the directory layout works in the development tree,
# where the service sits next to `repos/`, and silently breaks on a deployment where it does not —
# the installed path resolved to /opt/repos/Kronos and the model refused to load. An explicit
# environment variable is the difference between a deployment that works and one that looks fine
# until the first forecast round.
KRONOS_SRC = Path(os.environ.get("HOROS_KRONOS_SRC") or (REPO_ROOT / "repos" / "Kronos"))
MODEL_CACHE = Path(os.environ.get("HOROS_MODEL_CACHE") or (REPO_ROOT / ".models"))

# Pinned. A checkpoint change silently changes every forecast, so the scorecard would be comparing
# two different models under one name. Changing these means starting a new record, deliberately.
#
# **Why small and not base.** Kronos-base is 102.3M parameters against small's 24.7M and is the
# better model, but the production VM is 2 vCPU with no GPU, and inference cost is strictly linear
# in paths x horizon (batching gives no speedup — measured at 1068ms/path for 16 paths and
# 1100ms/path for 64). Measured per-symbol cost of the hourly 24-candle, 128-path generation at two
# threads with int8:
#
#     Kronos-small   ~16 min      3 symbols = ~47 min of each hour     fits
#     Kronos-base    ~45 min      3 symbols = ~134 min of each hour    does not fit
#
# A model that cannot finish before the next candle opens cannot keep a forward record at all, and
# the record is the product. So: small, and the choice is stated on the scorecard rather than
# implied.
TOKENIZER_REPO = os.environ.get("HOROS_TOKENIZER_REPO", "NeoQuasar/Kronos-Tokenizer-base")
MODEL_REPO = os.environ.get("HOROS_MODEL_REPO", "NeoQuasar/Kronos-small")
MODEL_NAME = MODEL_REPO.rsplit("/", 1)[-1].lower()
MODEL_PARAMS = {"kronos-mini": "4.1M", "kronos-small": "24.7M", "kronos-base": "102.3M"}.get(
    MODEL_NAME, "unknown")
MAX_CONTEXT = 512          # for Kronos-small and Kronos-base; mini takes 2048

# Sampling defaults. `top_p=0.9` and `T=1.0` are the library's own published defaults; deviating
# would need a reason and a note on the scorecard, since it changes the shape of every distribution.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 0

# Paths per scheduled forecast, and per live one. The scheduled record can afford more because it
# runs on a clock; a live call for an off-list symbol is someone waiting on an HTTP response, and
# x402 declares a 300-second ceiling. Both counts are disclosed in the response rather than implied.
# 96, not 128. Measured on the production host (m7i-flex.large, 2 vCPU, int8): 0.194 s per
# path-candle, so the hourly round over 3 symbols x 24 candles costs 22.3 min at 96 paths and
# 29.7 min at 128. The instance is a Flex type with a 40% sustained-CPU baseline, and 128 paths
# would sit at 50% duty — over the line, so it would eventually be throttled and the record
# would start missing rounds. 96 paths is 37% duty, which fits with margin for serving traffic.
SCHEDULED_PATHS = 96
LIVE_PATHS = 64
DEFAULT_PATHS = SCHEDULED_PATHS

# How many identical series go through the model at once. Measured: there is no batching win on CPU
# at all (1068ms/path at 16, 1100ms/path at 64 — the cores are already saturated), so this exists to
# bound peak memory on a 3 GB VM rather than to go faster.
BATCH_CHUNK = 32

# The features Kronos consumes, in the order its predictor expects.
PRICE_COLS = ("open", "high", "low", "close")


class ForecastError(RuntimeError):
    """A named model failure. Never a bare 500 — OKX rejected an agent for exactly that."""

    def __init__(self, message: str, code: str = "model_error"):
        super().__init__(message)
        self.code = code


@dataclass
class Fan:
    """The full predictive distribution: every generated path, undiscarded."""
    symbol: str
    timeframe: str
    horizon_candles: int
    anchor_price: float
    anchor_ts: int
    paths: np.ndarray            # (n_paths, horizon_candles, 4) — open, high, low, close
    model: str
    model_version: str
    seconds: float

    @property
    def n_paths(self) -> int:
        return int(self.paths.shape[0])

    @property
    def terminal(self) -> np.ndarray:
        """Closing price at the end of the horizon, one per path. What CRPS is computed on."""
        return self.paths[:, -1, 3]

    @property
    def path_high(self) -> np.ndarray:
        """Highest price reached anywhere in each path — the input to an upward touch probability."""
        return self.paths[:, :, 1].max(axis=1)

    @property
    def path_low(self) -> np.ndarray:
        return self.paths[:, :, 2].min(axis=1)

    def slice_to(self, candles: int) -> "Fan":
        """The same fan, truncated to a shorter horizon.

        One generation of 24 candles answers 1h, 4h, 12h and 24h at once: a sampled 24-step path is a
        draw from the *joint* future, so its value at step 4 is a valid draw from the 4-hour marginal.

        Slicing is not merely a saving of four-fifths of the compute. Four independent generations
        would produce horizons that contradict each other — a 24h band that does not contain its own
        4h band — because nothing would tie the draws together. Slicing one path set makes the
        horizons mutually consistent by construction, which is the correct answer as well as the
        cheap one.

        The extremes must be re-derived from the truncated path, not carried over, or a 1h touch
        probability would be answered with the high of a 24-hour path.
        """
        if not 1 <= candles <= self.horizon_candles:
            raise ForecastError(
                f"cannot slice a {self.horizon_candles}-candle fan to {candles} candles",
                "bad_horizon")
        return Fan(symbol=self.symbol, timeframe=self.timeframe, horizon_candles=candles,
                   anchor_price=self.anchor_price, anchor_ts=self.anchor_ts,
                   paths=self.paths[:, :candles, :], model=self.model,
                   model_version=self.model_version, seconds=self.seconds)

    def quantiles(self, levels) -> list[float]:
        return [float(np.quantile(self.terminal, q)) for q in levels]

    def band(self, coverage: float) -> list[float]:
        """A central band at the requested coverage, straight from the empirical fan.

        Uncalibrated — this is what the model claims, before the conformal layer corrects it against
        what the model has actually delivered. Nothing sells a band at this stage.
        """
        tail = (1.0 - coverage) / 2.0
        return [float(np.quantile(self.terminal, tail)),
                float(np.quantile(self.terminal, 1.0 - tail))]

    def touch_probability(self, level: float, direction: str) -> float:
        """Fraction of paths that reach `level` at any point before the horizon expires.

        Read off the path extremes rather than the terminal price, because "will it touch" is a
        question about the whole path. A path that spiked through and came back did touch it.
        """
        if direction == "up":
            return float(np.mean(self.path_high >= level))
        if direction == "down":
            return float(np.mean(self.path_low <= level))
        raise ForecastError(f"direction must be 'up' or 'down', got {direction!r}", "bad_direction")

    def realized_volatility(self) -> float:
        """Annualised realised volatility implied by the fan, from per-path log returns."""
        closes = self.paths[:, :, 3]
        start = np.full((closes.shape[0], 1), self.anchor_price)
        series = np.concatenate([start, closes], axis=1)
        rets = np.diff(np.log(np.maximum(series, 1e-12)), axis=1)
        per_candle = float(np.std(rets))
        return per_candle

    def to_distribution(self, levels, bands, include_paths: bool = False) -> dict:
        """The block written into the ledger. Carries the fan, not a summary.

        The commitment has to be to exactly what was predicted, or the reveal proves nothing — so
        the terminal prices of every path are recorded, always. Full paths are optional because they
        are ~50x larger and the terminal price is what every score is computed against.
        """
        d: dict = {
            "anchor_price": self.anchor_price,
            "anchor_timestamp_ms": self.anchor_ts,
            "n_paths": self.n_paths,
            "horizon_candles": self.horizon_candles,
            "quantile_levels": list(levels),
            "quantiles": self.quantiles(levels),
            "bands": {f"{c:g}": self.band(c) for c in bands},
            "samples": [float(v) for v in self.terminal],
            "path_high": [float(v) for v in self.path_high],
            "path_low": [float(v) for v in self.path_low],
            "per_candle_log_return_sd": self.realized_volatility(),
        }
        if include_paths:
            d["paths"] = self.paths.round(6).tolist()
        return d


class KronosRunner:
    """Loads the model once and serves fans from it.

    Loading is lazy and guarded: a 409 MB checkpoint should not be read at import time, and two
    concurrent first requests must not both try to load it.
    """

    def __init__(self, device: str = "cpu", model_repo: str = MODEL_REPO,
                 tokenizer_repo: str = TOKENIZER_REPO, quantize: bool | None = None):
        self.device = device
        self.model_repo = model_repo
        self.tokenizer_repo = tokenizer_repo
        # int8 dynamic quantization, on CPU only — it is a CPU kernel and does nothing on CUDA.
        # Measured on this codebase at 2 threads: 94.6s -> 39.4s for 32 paths over 4 candles, a
        # 2.40x speedup, with the terminal standard deviation moving from 163.31 to 163.28 and the
        # median by 0.03% of the anchor price. The distribution survives; only the latency changes.
        # That measurement is what makes the production VM (2 vCPU, no GPU) viable at all.
        self.quantize = (device == "cpu") if quantize is None else quantize
        self._predictor = None
        self._lock = threading.Lock()
        self._checkpoint_digest: str | None = None

    # ── loading ──────────────────────────────────────────────────────────────────────────────

    def _ensure_path(self) -> None:
        if str(KRONOS_SRC) not in sys.path:
            if not (KRONOS_SRC / "model" / "kronos.py").exists():
                raise ForecastError(
                    f"the Kronos source is not present at {KRONOS_SRC}. Clone it into repos/.",
                    "model_source_missing")
            sys.path.insert(0, str(KRONOS_SRC))

    @property
    def predictor(self):
        if self._predictor is not None:
            return self._predictor
        with self._lock:
            if self._predictor is not None:
                return self._predictor
            self._ensure_path()
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            try:
                from model import Kronos, KronosPredictor, KronosTokenizer
            except Exception as e:                                        # noqa: BLE001
                raise ForecastError(f"the Kronos package could not be imported: "
                                    f"{type(e).__name__}: {e}", "model_import_failed") from e
            t0 = time.time()
            try:
                tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_repo,
                                                            cache_dir=str(MODEL_CACHE))
                model = Kronos.from_pretrained(self.model_repo, cache_dir=str(MODEL_CACHE))
            except Exception as e:                                        # noqa: BLE001
                raise ForecastError(f"the pinned checkpoint {self.model_repo} could not be loaded: "
                                    f"{type(e).__name__}: {e}", "checkpoint_unavailable") from e
            self._predictor = KronosPredictor(model, tokenizer, device=self.device,
                                              max_context=MAX_CONTEXT)
            if self.quantize:
                import torch
                self._predictor.model = torch.quantization.quantize_dynamic(
                    self._predictor.model, {torch.nn.Linear}, dtype=torch.qint8)
                self._predictor.tokenizer = torch.quantization.quantize_dynamic(
                    self._predictor.tokenizer, {torch.nn.Linear}, dtype=torch.qint8)
            log.info("loaded %s on %s in %.1fs (quantized=%s)", self.model_repo, self.device,
                     time.time() - t0, self.quantize)
            return self._predictor

    def checkpoint_digest(self) -> str:
        """Hash of the weights actually loaded.

        Recorded with every forecast. "Kronos-base" is a name; this is the thing that produced the
        number, and it is what makes a forecast reproducible after we upgrade.
        """
        if self._checkpoint_digest:
            return self._checkpoint_digest
        matches = list(MODEL_CACHE.glob(
            f"models--{self.model_repo.replace('/', '--')}/snapshots/*/model.safetensors"))
        if not matches:
            return "unknown"
        h = hashlib.sha256()
        with open(matches[0], "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        self._checkpoint_digest = h.hexdigest()[:16]
        return self._checkpoint_digest

    def warm(self) -> None:
        """Force the load. Called at startup so no buyer pays the 409 MB read."""
        _ = self.predictor

    # ── inference ────────────────────────────────────────────────────────────────────────────

    def forecast(self, candles, horizon_candles: int, n_paths: int = DEFAULT_PATHS,
                 temperature: float = DEFAULT_TEMPERATURE, top_p: float = DEFAULT_TOP_P,
                 top_k: int = DEFAULT_TOP_K, seed: int | None = None) -> Fan:
        """Generate `n_paths` independent futures for the next `horizon_candles`.

        `candles` is a `market_data.Candles` — closed bars only, contiguous, most recent last.
        """
        import pandas as pd
        import torch

        if horizon_candles < 1:
            raise ForecastError("the horizon must be at least one candle", "bad_horizon")
        if n_paths < 1:
            raise ForecastError("at least one path is required", "bad_path_count")
        if len(candles) < 32:
            raise ForecastError(
                f"only {len(candles)} closed candles are available for {candles.symbol}; "
                f"the model needs a meaningful context and 32 is the floor.", "insufficient_history")

        context = candles.rows[-MAX_CONTEXT:]
        df = pd.DataFrame(context, columns=["ts", "open", "high", "low", "close", "volume"])
        x_timestamp = pd.to_datetime(df["ts"], unit="ms", utc=True).dt.tz_localize(None)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        if not np.isfinite(df.to_numpy()).all():
            raise ForecastError(f"the {candles.symbol} history contains non-finite values; "
                                f"refusing to forecast from it", "bad_history")

        step_ms = int(context[1][0] - context[0][0]) if len(context) > 1 else 3_600_000
        last_ts = int(context[-1][0])
        y_timestamp = pd.to_datetime(
            [last_ts + step_ms * (i + 1) for i in range(horizon_candles)], unit="ms", utc=True
        ).tz_localize(None).to_series().reset_index(drop=True)

        predictor = self.predictor
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        t0 = time.time()
        collected: list[np.ndarray] = []
        remaining = n_paths
        while remaining > 0:
            size = min(BATCH_CHUNK, remaining)
            try:
                # sample_count=1 makes the library's mean a no-op; the batch supplies the ensemble.
                out = predictor.predict_batch(
                    df_list=[df] * size,
                    x_timestamp_list=[x_timestamp] * size,
                    y_timestamp_list=[y_timestamp] * size,
                    pred_len=horizon_candles,
                    T=temperature, top_k=top_k, top_p=top_p,
                    sample_count=1, verbose=False)
            except Exception as e:                                        # noqa: BLE001
                raise ForecastError(f"inference failed for {candles.symbol}: "
                                    f"{type(e).__name__}: {e}", "inference_failed") from e
            for frame in out:
                collected.append(frame[list(PRICE_COLS)].to_numpy(dtype=float))
            remaining -= size

        paths = np.stack(collected, axis=0)
        if not np.isfinite(paths).all():
            raise ForecastError(f"the model produced non-finite prices for {candles.symbol}",
                                "bad_output")

        return Fan(symbol=candles.symbol, timeframe=candles.timeframe,
                   horizon_candles=horizon_candles, anchor_price=float(candles.last_close),
                   anchor_ts=last_ts, paths=paths, model=MODEL_NAME,
                   model_version=f"{self.model_repo}@{self.checkpoint_digest()}",
                   seconds=time.time() - t0)

    # ── provenance ───────────────────────────────────────────────────────────────────────────

    def describe(self) -> dict:
        """What the model is and, just as importantly, what it cannot see."""
        return {
            "model": MODEL_NAME,
            "checkpoint": self.model_repo,
            "tokenizer": self.tokenizer_repo,
            "parameters": MODEL_PARAMS,
            "max_context_candles": MAX_CONTEXT,
            "device": self.device,
            "sampling": {"temperature": DEFAULT_TEMPERATURE, "top_p": DEFAULT_TOP_P,
                         "top_k": DEFAULT_TOP_K, "default_paths": DEFAULT_PATHS},
            "sees": ["open", "high", "low", "close", "volume", "calendar features"],
            "cannot_see": [
                "funding rates", "open interest", "basis", "order book depth",
                "liquidations", "news, filings or social sentiment", "any other market",
            ],
            "no_backtest": (
                "Kronos was pre-trained across 45 global exchanges to an unpublished cutoff, so any "
                "historical evaluation risks scoring it on its own training data. Horos publishes "
                "no backtest. The record starts when the ledger starts and accumulates forward."),
        }


_RUNNER: KronosRunner | None = None
_RUNNER_LOCK = threading.Lock()


def shared() -> KronosRunner:
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            _RUNNER = KronosRunner()
        return _RUNNER
