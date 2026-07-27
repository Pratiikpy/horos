# Horos

**Marked before the outcome.**

Market forecasts with a public, on-chain accuracy record. Every prediction is hash-committed to
X Layer *before its window closes*, scored automatically against what actually happened, and
published — losses included.

[**The live record**](https://horos.ivaronix.xyz/scorecard) ·
[**Proof: all 40 services bought with real money**](https://horos.ivaronix.xyz/proof) ·
[**Full product story**](https://comfortable-goal-205.notion.site/Horos-Marked-before-the-outcome-3aa9c0ce7876819eb301d408ce3e4c3b) ·
[**Download the ledger**](https://horos.ivaronix.xyz/ledger) ·
[**Verify the chain**](https://horos.ivaronix.xyz/ledger/verify)

> At the time of writing the record says the model is **losing to a random walk by 26%**, on a
> sample far too small to mean anything. It says that itself, on its own front page. That is the
> product working — see [Honest status](#honest-status).

---

## The problem

Ask any forecasting service for its track record and you get a cherry-picked backtest, a
screenshot, or silence. This is not a few bad actors — it is structural. **Nothing forces a
forecaster to publish its misses**, and the buyer has no mechanism to check. So the market prices
confidence instead of accuracy.

> A forecast is unverifiable when you buy it and **perfectly verifiable a day later.**

Close that gap and forecasting becomes a market where being right is worth more than sounding right.

## The name

In classical Athens, if you pledged land against a loan, a marble slab called a *horos* (ὅρος) was
planted **in that soil**, carved with what was owed and to whom — so anyone could read the claim
*before* dealing with you. The claim went into the ground first; the outcome came later.

The same word means a **boundary**, and in Aristotle the **term** of a proposition. Three meanings,
all exact: the on-chain commitment, the forecast band, and the precise claim being judged.

Athens' answer to *"how do you make a financial claim publicly verifiable?"* was to carve it in
stone, in public, before the fact. ([Fine, *Horoi*, ASCSA](https://www.ascsa.edu.gr/publications/book/?i=9780876615096) ·
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

1. **Forecast on a clock.** Hourly, for BTC/ETH/SOL perpetuals, from 512 candles of context.
2. **Write it down before the outcome exists.** Every path's terminal price, high and low goes into
   a hash-chained ledger — each entry carries the previous entry's hash.
3. **Commit the head on X Layer** before the window closes. 46 bytes of calldata; a rebuilt history
   cannot reproduce a hash already in a block.
4. **Score automatically** with proper rules — CRPS, pinball, Winkler interval score, Brier, PIT —
   against a free baseline anyone could have used instead.
5. **Publish, including the losses.** The scorecard is generated *from the ledger*.
6. **Judge everyone else on the same terms** — `commit` and `judge` are open to third parties.

**One writer, enforced by a pid lock.** Two writers appending to one hash chain would not lose data;
they would produce a chain that fails verification permanently, which is worse.

---

## The one-line change the product rests on

Kronos is the only open foundation model for K-lines. It generates hundreds of independent futures
— and throws them away:

```python
# model/kronos.py, auto_regressive_inference, line 467
z = z.reshape(-1, sample_count, z.size(1), z.size(2))
preds = z.cpu().numpy()
preds = np.mean(preds, axis=1)      # <- the distribution, averaged into a line
```

Every user of the library out of the box receives **a single line**. The bands, the touch
probabilities, the tail percentile that makes surprise detection possible and the volatility
forecast **all live in the fan that line discards.**

Horos recovers it **without modifying the library**: with `sample_count=1` that mean is a reduction
over an axis of length one — the identity — and `predict_batch` samples independently per batch
element via `torch.multinomial`. Feeding N identical copies of one series yields N independent draws
in a single batched pass.

No vendored fork, no monkeypatch, nothing to keep in sync with upstream.
See [`core/kronos_runner.py`](core/kronos_runner.py).

### Two more decisions made by measurement

| Decision | Measurement |
|---|---|
| int8 quantization | **2.40× faster**, terminal sd 163.31 → 163.28 — the distribution survives |
| One 24-candle generation, sliced to 4 horizons | 4× cheaper, *and* the horizons become mutually consistent. Four independent generations would produce a 24h band not containing its own 4h band |
| Kronos-small over base | base needs ~134 min of each hour on the production host; small needs ~47. A model that cannot finish before the next candle opens cannot keep a forward record |

---

## Design principles

These are enforced in code, not aspirational.

**Never a silent null.** A source that is down is reported as *down*, never as an absence.
`metrics.risk` returns a `not_computed` map naming every metric that failed and why. A band that has
not been conformally calibrated says so rather than being passed off as guaranteed.

**A failure is never rendered as a pass.** `market.dislocation` reports "this spread does not survive
the checks" rather than an opportunity. `forecast.reliability` will tell a caller *not to pay* for a
forecast at a horizon where the model does not beat a free baseline.

**Corrections are appends.** `ledger.record_void()` is the only correction mechanism — nothing is
ever edited or deleted, and a voided forecast stays in the chain with its reason attached.

**The description is the floor, not the pitch.** `services/registry.py` fails at import if any
service lacks a two-part description, parameter details, a worked example, or a statement of how far
it actually goes.

---

## The 40 services

**39 A2MCP + 1 A2A.** Raw exchange data is free; the measurement on top is the product.

| Group | # | Price | The extra step that makes it worth buying |
|---|---|---|---|
| **Forecast** | 5 | $0.02–0.08 | The full distribution, not the averaged line. Conformally calibrated bands with real coverage guarantees. Touch probabilities resolved on path *extremes*, so a wick through your stop counts |
| **Surprise & regime** | 4 | $0.02–0.05 | Measured against a distribution fixed and hashed *before* the candle existed — fitting one afterwards makes every outcome look ordinary |
| **Risk & sizing** | 5 | $0.03–0.08 | Sized off the real path distribution so fat tails are priced in. Reports what fraction of paths hit your stop then finish in your favour anyway |
| **Execution** | 3 | $0.03–0.05 | A real walk through the resting book. If the order exhausts visible depth it says so rather than extrapolating a fill |
| **Deterministic metrics** | 9 | $0.005–0.01 | 86 indicators, VaR nine ways, GARCH, timestamp-aligned correlation — all recomputable from the candles echoed back with their SHA-256 |
| **Market data** | 6 | $0.001–0.01 | In-progress bar always dropped, gapped series refused, stale venues excluded from spreads |
| **Accountability** | 7 | $0.001–0.02 | The record, per-regime and per-checkpoint breakdowns, third-party commit/judge, leaderboard |
| **Agent to agent** | 1 | negotiated | A language model plans and narrates; **it never produces a number** |

Full descriptions, parameters and worked examples: [`/services`](https://horos.ivaronix.xyz/services).

---

## Verify anything without trusting us

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

Any on-chain anchor decodes to a ledger head with no ABI — 46 bytes of calldata:

```python
raw = bytes.fromhex(tx_input.removeprefix('0x'))
assert raw[:5] == b'HOROS'
head    = 'sha256:' + raw[6:38].hex()
entries = int.from_bytes(raw[38:46], 'big')
```

Pull the input field of any Horos anchor transaction from an X Layer explorer and decode it — it
must match a head in the ledger you can download.

---

## What testing actually caught

**160 test functions across 8 files.** They exist because seven real bugs were found by them — every
one of the silent-failure class, returning 200 with a plausible wrong answer:

1. **Signature forgery hole** — `verify()` checked the signature against the digest *stored in the
   entry* instead of rebuilding it from the body, so any field could be edited if the old hash was
   left in place. Fixed with one shared manifest builder and a pinned key.
2. **The forecast window was off by one candle** — the model predicts the bar *after* the last
   closed one; the window pointed one further. Every score would have graded a candle the model
   never predicted, consistently and invisibly.
3. **Voided forecasts blocked re-issue**, freezing the recorder while it logged "already_issued".
4. **OKX returns 3-element orderbook rows** — unpacking as pairs was a hard 500 on a paid endpoint.
5. **`percentile: 1.0` read as "1st percentile"** when it meant 100th, producing a materially wrong
   sentence in a generated brief.
6. **Drawdown reported "0 episodes"** during a live 29% drawdown, because none had *recovered*.
7. **Binance and Bybit geo-block US IPs** — only findable by testing from the real deployment region.

When #2 was found, twelve already-issued forecasts carried bad windows. They were **voided on the
record** — signed, permanent, reason attached — not deleted. The product's own correction rule,
applied to itself on day one.

The tamper tests are the interesting ones: each performs the specific edit a dishonest operator
would actually attempt — drop the losing forecast, soften it, reorder two, re-sign with your own key
— and asserts `verify()` names it.

---

## What it will never do

- **No buy, sell or hold.** `risk.size` answers "what is consistent with *your* stated limit", never
  "should you trade".
- **No direction calls.** Short-horizon direction is close to random; selling it is selling noise.
- **No backtested performance, ever.** Kronos was pre-trained across 45 global exchanges to an
  unpublished cutoff, so any historical evaluation risks scoring it on its own training data. The
  record starts empty and accumulates forward.
- **No liquidation feed** — ccxt exposes none for OKX, so cascade detection would be inferred rather
  than measured.
- **Not financial advice.** Measurement, assembled and signed.

---

## Honest status

The record is **thin**, and the scorecard says so itself.

At the last check: **10 scored forecasts.** The model is **losing to an empirical random walk by
26%** and beating a naive "price stays where it is" baseline by 2%. Those figures *flipped* from the
previous reading (+11% and −80%) — which is exactly what a ten-observation sample does, and the page
states plainly that it is far too small to distinguish skill from luck.

A scorecard that only ever showed wins would be the thing this product exists to replace.

The record fills at ~2,000 scored forecasts per week. Conformal calibration enables itself once
enough are scored; until then every band is labelled *uncalibrated*.

---

## Layout

```
core/         the engine
  ledger.py         hash-chained append-only log; corrections are appends, never edits
  scoring.py        CRPS (fair estimator), pinball, interval score, Brier, PIT, skill vs baseline
  conformal.py      split conformal with the finite-sample correction most implementations drop
  commit.py         46-byte on-chain anchor codec
  kronos_runner.py  the model, serving the distribution instead of the mean
  market_data.py    ccxt layer; drops the in-progress candle, refuses gapped series
  recorder.py       issue on the candle clock, score on close, anchor after every round
services/     the 40 paid services, grouped
tests/        160 tests — every tamper attack, every scoring property
scripts/      recorder daemon, real-money paid sweep, deployment
```

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                   # then fill in your own keys
uvicorn server:app --port 8080
python -m scripts.recorder_daemon      # the forecast / score / anchor loop
pytest -m "not live"                   # the offline suite
python -m scripts.paid_sweep --real    # buy all 40 with real USD₮0 and judge the output
```

## Built on

[Kronos](https://github.com/shiyu-coder/Kronos) ·
[ccxt](https://github.com/ccxt/ccxt) ·
[FinanceToolkit](https://github.com/JerBouma/FinanceToolkit) ·
[ta](https://github.com/bukosabino/ta) ·
[PyPortfolioOpt](https://github.com/robertmartin8/PyPortfolioOpt) ·
[conformal-prediction](https://github.com/aangelopoulos/conformal-prediction) ·
[arch](https://github.com/bashtage/arch)

All MIT or equivalent. Twelve repositories were studied; two of the most-starred were deliberately
**not** built on — an RL policy cannot produce an answer a buyer can check, and this is deliberately
not a trading bot.

## Licence

MIT.
