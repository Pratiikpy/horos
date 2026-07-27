# Horos

**Marked before the outcome.**

Market forecasts with a public, on-chain accuracy record. Every prediction is hash-committed to
X Layer *before its window closes*, scored automatically against what actually happened, and
published — losses included.

[**The record**](https://horos.ivaronix.xyz/scorecard) ·
[**Proof: all 40 services bought with real money**](https://horos.ivaronix.xyz/proof) ·
[**Download the ledger**](https://horos.ivaronix.xyz/ledger) ·
[**Verify the chain**](https://horos.ivaronix.xyz/ledger/verify) ·
[**Services**](https://horos.ivaronix.xyz/services)

---

## The problem

Every forecasting product shares one defect: **it is unfalsifiable at the moment of sale.** You are
told what will happen, you pay, and nobody ever tells you whether it was right. The misses are not
published because nothing forces them to be.

That is the whole opportunity.

> A forecast is unverifiable when you buy it and perfectly verifiable a day later.
> So hash-commit every prediction on-chain before its window closes, score it automatically against
> what happened, and publish the record. Sell the forecast; be judged on the ledger.

## The name

In classical Athens a *horos* (ὅρος) was a marble slab planted **on the pledged property itself**,
inscribed with what was owed and to whom, so that anyone could read the claim before acting on it.
The same word means a **boundary** — a limit — and in Aristotle the **term** of a proposition.

Three meanings, all exact: the on-chain commitment, the forecast band, and the precise claim being
judged. ([Fine, *Horoi*, ASCSA](https://www.ascsa.edu.gr/publications/book/?i=9780876615096) ·
[Finley, *Studies in Land and Credit*](https://books.google.com/books/about/Studies_in_Land_and_Credit_in_Ancient_At.html?id=WbYNeX5TWecC))

---

## How it works

```
   x402 paywall
        |
        v
   +----------------+     +------------------+     +--------------------+
   |  40 services   | --> |  Kronos + fan    | --> |  Conformal layer   |
   |  39 A2MCP      |     |  512 candles     |     |  calibrated bands  |
   |   1 A2A        |     |  distribution    |     |  coverage proof    |
   +----------------+     +------------------+     +--------------------+
        |                                                   |
        v                                                   v
   +--------------------------------------------------------------+
   |  Hash-chained ledger  ·  Ed25519 receipts  ·  proper scoring  |
   +--------------------------------------------------------------+
                                |
                                v
   +--------------------------------------------------------------+
   | Commit-reveal on X Layer  ·  public scorecard  ·  leaderboard |
   +--------------------------------------------------------------+
```

1. **Forecast on a clock.** Hourly, for BTC/ETH/SOL perpetuals, a foundation model generates
   hundreds of independent future paths from 512 candles of context.
2. **Write it down before the outcome exists.** The full distribution goes into a hash-chained
   ledger — each entry carries the previous entry's hash.
3. **Commit the head on X Layer** before the window closes. A rebuilt history cannot reproduce a
   hash that is already in a block.
4. **Score automatically** when the window closes, with proper scoring rules against a free baseline.
5. **Publish, including the losses.** The scorecard is generated from the ledger, never hand-written.
6. **Judge everyone else on the same terms.** Anyone can commit their own forecast and be scored by
   the same code.

---

## The one-line change the whole product rests on

Kronos is the only open foundation model for K-lines. It generates `sample_count` independent future
paths — and then throws them away:

```python
# model/kronos.py, auto_regressive_inference, line 467
z = z.reshape(-1, sample_count, z.size(1), z.size(2))
preds = z.cpu().numpy()
preds = np.mean(preds, axis=1)      # <- the distribution, averaged into a line
```

Every user of the library out of the box receives a single line. The bands, the touch probabilities,
the tail percentile that makes surprise detection possible and the volatility forecast **all live in
the fan that line discards.**

Horos recovers it **without modifying the library**: with `sample_count=1` that mean is a reduction
over an axis of length one — the identity — and `predict_batch` samples independently per batch
element. Feeding N identical copies of one series yields N independent draws in one batched pass.

No vendored fork, no monkeypatch. See [`core/kronos_runner.py`](core/kronos_runner.py).

---

## What it will not claim

- **No buy, sell or hold.** Every service is a measurement or an inference with its uncertainty
  attached. `risk.size` answers "what is consistent with *your* stated limit", never "should you
  trade".
- **No direction calls.** Short-horizon direction is close to random; selling it is selling noise.
- **No backtested performance.** Kronos was pre-trained across 45 global exchanges to an unpublished
  cutoff, so any historical evaluation risks scoring it on its own training data. The record starts
  empty and accumulates forward.
- **No liquidation feed** — ccxt exposes none for OKX, so cascade detection would be inferred rather
  than measured.
- **Not financial advice.** Measurement, assembled and signed.

---

## The 40 services

| Group | Count | Price | What it covers |
|---|---|---|---|
| Market data | 6 | $0.001–0.01 | Candles, book depth and imbalance, funding in its own percentile, open-interest divergence, basis, cross-venue spread |
| Deterministic metrics | 9 | $0.005–0.01 | 86 indicators, VaR/CVaR nine ways, risk-adjusted performance, volatility five ways incl. GARCH, distribution shape, correlation, liquidity, microstructure, full drawdown history |
| Forecast | 5 | $0.02–0.08 | Full predictive distribution, conformally calibrated envelope, touch probability, expected volatility, and where the model is actually reliable |
| Surprise & regime | 4 | $0.02–0.05 | Was that candle unusual, volatility regime + Hurst, price/funding/OI divergence, real-or-artefact cross-venue dislocation |
| Risk & sizing | 5 | $0.03–0.08 | Position size for your loss limit, distribution-based stops, portfolio VaR with contributions, scenario stress with betas, efficient frontier + HRP |
| Execution | 3 | $0.03–0.05 | Order cost before sending, post-trade fill quality, best venue for this size |
| Accountability | 7 | $0.001–0.02 | The record, per-regime and per-model breakdowns, third-party commit/judge, leaderboard, receipt verification |
| Agent to agent | 1 | negotiated | Hire Horos in conversation for work that does not fit a fixed call |

Full descriptions, parameters and worked examples: [`/services`](https://horos.ivaronix.xyz/services).

---

## Verify any answer without trusting us

Every response carries an Ed25519 receipt over a manifest of six fields, all echoed in the response:

```python
import hashlib, json
from nacl.signing import VerifyKey

m = {'endpoint': env['endpoint'], 'input_sha256': env['input_sha256'],
     'output_sha256': env['output_sha256'], 'tool': 'horos',
     'price_usdt': env['price_usdt'], 'job_id': env['job_id']}
blob = json.dumps(m, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
d = 'sha256:' + hashlib.sha256(blob.encode()).hexdigest()
assert d == env['receipt']['manifest_sha256']
VerifyKey(bytes.fromhex(env['receipt']['public_key'])).verify(
    d.encode(), bytes.fromhex(env['receipt']['signature']))
```

And any on-chain anchor decodes to a ledger head with no ABI:

```python
raw = bytes.fromhex(tx_input.removeprefix('0x'))   # 46 bytes
assert raw[:5] == b'HOROS'
head    = 'sha256:' + raw[6:38].hex()
entries = int.from_bytes(raw[38:46], 'big')
```

---

## Layout

```
core/         the engine
  ledger.py         hash-chained append-only log; corrections are appends, never edits
  scoring.py        CRPS, pinball, interval score, Brier, PIT, skill vs baseline
  conformal.py      split conformal calibration with the finite-sample correction
  commit.py         46-byte on-chain anchor codec
  kronos_runner.py  the model, serving the distribution instead of the mean
  market_data.py    ccxt layer; drops the in-progress candle, refuses gapped series
  recorder.py       issue on the candle clock, score on close, anchor after every round
services/     the 40 paid services, grouped
tests/        174 tests — every tamper attack, every scoring property
scripts/      recorder daemon, paid sweep, deployment
```

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env          # then fill in your own keys
uvicorn server:app --port 8080
python -m scripts.recorder_daemon      # the forecast/score/anchor loop
pytest -m "not live"                   # 174 offline tests
python -m scripts.paid_sweep --real    # buy all 40 with real USD₮0 and judge the output
```

## Built on

[Kronos](https://github.com/shiyu-coder/Kronos) ·
[ccxt](https://github.com/ccxt/ccxt) ·
[FinanceToolkit](https://github.com/JerBouma/FinanceToolkit) ·
[ta](https://github.com/bukosabino/ta) ·
[PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) ·
[conformal-prediction](https://github.com/aangelopoulos/conformal-prediction) ·
[arch](https://github.com/bashtage/arch) — all MIT or equivalent.

## Licence

MIT.
