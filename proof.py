"""The proof deck — every service, bought with real money, with the answer it returned.

Rendered from `.data/proof-data.json`, which is written by `scripts/paid_sweep.py --real`. Nothing
on this page is typed by hand, so it cannot claim a number that did not happen: each row's status
code, settlement transaction, timing and output are read out of a recorded exchange.

Two things a reader should be able to do here that most "proof" pages do not allow:

  * **See the actual answer**, not a tick. A green check proves a request returned 200; it proves
    nothing about whether the output was worth paying for. Every row expands to the real payload.
  * **Check the money themselves.** Every settlement hash links to an X Layer explorer. A reader who
    doubts any of it can open the transaction and see USD₮0 move.

The outcome checks shown per row are the ones in `paid_sweep.CHECKS` — assertions about what the
service promised to return, run against what it actually returned. They are the difference between
"the endpoint works" and "the endpoint is worth money".
"""
from __future__ import annotations

import html
import io
import json
from pathlib import Path

DATA = Path(__file__).parent / ".data" / "proof-data.json"
EXPLORER = "https://www.oklink.com/xlayer/tx/"

GROUPS = [
    ("Forecast", "The distribution the model actually produced, not the averaged line its library "
                 "returns by default.", ("forecast.", "market.surprise", "market.regime",
                                         "market.anomaly", "market.dislocation")),
    ("Accountability", "The record, and the machinery that lets anyone else be judged by it too.",
     ("scorecard", "leaderboard", "commit", "judge", "receipt.verify")),
    ("Market data", "Not a proxied feed — each one does the measurement on top that makes it worth "
                    "buying.", ("market.",)),
    ("Deterministic metrics", "Recomputable from the candles echoed back. Verify one and you have "
                              "reason to trust the rest.", ("metrics.",)),
    ("Risk and execution", "What is consistent with the limit you stated. Never whether to trade.",
     ("risk.", "portfolio.", "exec.")),
    ("Agent to agent", "Hire Horos in conversation for work that does not fit a fixed call.",
     ("analysis.",)),
]


def _e(x) -> str:
    return html.escape(str(x if x is not None else ""), quote=True)


def _pretty(obj, limit: int = 2600) -> str:
    s = json.dumps(obj, indent=1, ensure_ascii=False, default=str)
    if len(s) > limit:
        s = s[:limit].rsplit("\n", 1)[0] + "\n  … truncated for the page; the full payload is what "
        s += "the call returned"
    return _e(s)


CSS = """
*{box-sizing:border-box}
:root{--bg:#0f1012;--panel:#17181c;--line:#26282e;--ink:#e9e7e3;--dim:#9a968f;--faint:#6c6a64;
 --good:#5fb37a;--bad:#d4685f;--warn:#d0a24c;--accent:#c9b896;--claim:#e8a13c;
 --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace}
@media(prefers-color-scheme:light){:root{--bg:#faf9f7;--panel:#fff;--line:#e4e1db;--ink:#1a1917;
 --dim:#6b6862;--faint:#93908a;--good:#2f7d4f;--bad:#b03a30;--warn:#8a6412;--accent:#6b5a3a;
 --claim:#b0561a}}
:root[data-theme=dark]{--bg:#0f1012;--panel:#17181c;--line:#26282e;--ink:#e9e7e3;--dim:#9a968f;
 --faint:#6c6a64;--accent:#c9b896;--claim:#e8a13c;--good:#5fb37a;--bad:#d4685f}
:root[data-theme=light]{--bg:#faf9f7;--panel:#fff;--line:#e4e1db;--ink:#1a1917;--dim:#6b6862;
 --faint:#93908a;--accent:#6b5a3a;--claim:#b0561a;--good:#2f7d4f;--bad:#b03a30}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 ui-sans-serif,-apple-system,
 "Segoe UI",Inter,system-ui,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:60px 22px 110px}
header{border-bottom:1px solid var(--line);padding-bottom:38px;margin-bottom:42px}
.brand{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
h1{font-size:38px;margin:0;letter-spacing:-.022em;font-weight:600}
.tag{color:var(--accent);font-size:15px}
.lede{color:var(--dim);margin-top:20px;max-width:74ch}
.lede strong{color:var(--ink)}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.14em;color:var(--faint);
 margin:56px 0 6px;font-weight:600}
.gsub{color:var(--dim);font-size:14px;margin:0 0 18px;max-width:76ch}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;
 min-width:0}
.stat .k{font-size:11.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--faint)}
.stat .v{font-size:27px;margin-top:7px;font-variant-numeric:tabular-nums;letter-spacing:-.015em}
.stat .s{font-size:12.5px;color:var(--dim);margin-top:5px;overflow-wrap:anywhere}
details{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin-bottom:9px;
 overflow:hidden}
details[open]{border-color:var(--accent)}
summary{cursor:pointer;padding:13px 17px;display:flex;gap:14px;align-items:center;list-style:none;
 flex-wrap:wrap}
summary::-webkit-details-marker{display:none}
summary:hover{background:rgba(201,184,150,.05)}
.nm{font-weight:600;min-width:210px;flex:1}
.nm small{display:block;font-weight:400;color:var(--dim);font-size:12.5px;margin-top:2px}
.px{font-family:var(--mono);font-size:12.5px;color:var(--claim);min-width:66px;text-align:right}
.ck{font-family:var(--mono);font-size:12px;color:var(--good);min-width:52px;text-align:right}
.tx{font-family:var(--mono);font-size:11.5px}
.body{padding:4px 17px 20px;border-top:1px solid var(--line)}
.h3{font-size:11.5px;text-transform:uppercase;letter-spacing:.1em;color:var(--faint);
 margin:16px 0 7px;font-weight:600}
pre{background:var(--bg);border:1px solid var(--line);border-radius:7px;padding:12px 14px;
 overflow-x:auto;font-family:var(--mono);font-size:12px;line-height:1.55;margin:0;
 white-space:pre;color:var(--ink)}
ul.checks{margin:0;padding-left:18px;color:var(--dim);font-size:13.5px}
ul.checks li{margin:3px 0}
ul.checks li b{color:var(--good);font-weight:600}
a{color:var(--accent);overflow-wrap:anywhere}a:hover{opacity:.78}
.note{color:var(--dim);font-size:13.5px;max-width:78ch;margin-top:16px}
.flow{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:6px 0;
 margin-bottom:14px}
.step{display:flex;gap:15px;padding:13px 20px;border-bottom:1px solid var(--line);align-items:start}
.step:last-child{border-bottom:none}
.step .n{width:25px;height:25px;border-radius:50%;background:var(--accent);color:var(--bg);
 display:flex;align-items:center;justify-content:center;font-size:12.5px;font-weight:700;
 flex-shrink:0;margin-top:1px}
.step .t{flex:1;min-width:0}
.step .t b{display:block;margin-bottom:2px}
.step .t span{color:var(--dim);font-size:13.5px}
.arch{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:20px;
 overflow-x:auto}
.arch pre{border:none;background:none;padding:0;font-size:12px;color:var(--dim)}
footer{margin-top:80px;padding-top:28px;border-top:1px solid var(--line);color:var(--faint);
 font-size:13px}
code{font-family:var(--mono);font-size:12.5px;background:var(--panel);border:1px solid var(--line);
 padding:1.5px 5px;border-radius:4px}
"""


def page() -> str:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    rows = d["results"]
    a2a = d.get("a2a") or {}

    delivered = sum(1 for r in rows if r.get("delivered"))
    settled = [r for r in rows if r.get("settlement_tx")]
    checks_ok = sum(r.get("outcome_passed", 0) for r in rows)
    checks_all = sum(r.get("outcome_total", 0) for r in rows)
    receipts = sum(1 for r in rows if r.get("receipt_verifies"))
    spent = sum(float(r["price"]) for r in rows)

    # A complete document. Without the viewport meta a phone lays this out at ~980px and zooms out,
    # which turns the per-service evidence table into something unreadable on the device a reviewer
    # is most likely holding. Without a doctype the browser also drops into quirks mode.
    out = ['<!doctype html>', '<html lang="en">', '<head>', '<meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width,initial-scale=1">',
           "<title>Horos — proof</title>",
           '<meta name="description" content="Every Horos service bought with real USD₮0 on '
           'X Layer, and the answer it returned.">',
           f"<style>{CSS}</style>", '</head>', '<body>', '<div class="wrap">']
    out.append(f"""
<header>
  <div class="brand"><h1>Horos</h1><span class="tag">Marked before the outcome.</span></div>
  <p class="lede">Every service below was <strong>bought with real USD₮0 on X Layer</strong>, by a
  registered OKX user agent, exactly as a customer would. This page is generated from the recording
  of those purchases — the status codes, the settlement hashes, the timings and the payloads are all
  read back from what actually happened, not written by hand.</p>
  <p class="lede">A green tick would only prove a request returned 200. So each row also carries
  <strong>outcome checks</strong>: assertions about what the service promised, run against what it
  really returned. And every payment links to the transaction, so you can verify the money moved
  without taking our word for it.</p>
  <div class="grid" style="margin-top:28px">
    <div class="stat"><div class="k">Services bought</div><div class="v">{delivered}/{len(rows)}</div>
      <div class="s">every one delivered</div></div>
    <div class="stat"><div class="k">Settled on X Layer</div><div class="v">{len(settled)}</div>
      <div class="s">distinct transactions</div></div>
    <div class="stat"><div class="k">Outcome checks</div><div class="v">{checks_ok}/{checks_all}</div>
      <div class="s">quality, not status codes</div></div>
    <div class="stat"><div class="k">Receipts verified</div><div class="v">{receipts}/{len(rows)}</div>
      <div class="s">Ed25519, offline-checkable</div></div>
    <div class="stat"><div class="k">Total spent</div><div class="v">${spent:.3f}</div>
      <div class="s">real money, not a testnet</div></div>
  </div>
</header>""")

    out.append(_architecture())
    if a2a:
        out.append(_a2a_section(a2a))

    seen = set()
    for title, sub, prefixes in GROUPS:
        group = [r for r in rows
                 if r["endpoint"] not in seen and any(r["endpoint"].startswith(p) for p in prefixes)]
        if not group:
            continue
        seen.update(r["endpoint"] for r in group)
        out.append(f'<h2>{_e(title)}</h2><p class="gsub">{_e(sub)}</p>')
        for r in group:
            out.append(_row(r))

    leftover = [r for r in rows if r["endpoint"] not in seen]
    if leftover:
        out.append('<h2>Also purchased</h2><p class="gsub">&nbsp;</p>')
        for r in leftover:
            out.append(_row(r))

    out.append(_footer())
    return "\n".join(out)


def _row(r: dict) -> str:
    ep = r["endpoint"]
    checks = r.get("outcome", [])
    passed = sum(1 for c in checks if c[1])
    tx = r.get("settlement_tx")
    resp = r.get("response") or {}
    output = resp.get("output", {})
    title = resp.get("title") or ep

    check_html = "".join(
        f'<li><b>{"PASS" if c[1] else "FAIL"}</b> — {_e(c[0])}</li>' for c in checks)
    tx_html = (f'<a class="tx" href="{EXPLORER}{tx}" rel="noopener" target="_blank">'
               f'{_e(tx[:18])}…</a>' if tx else '<span class="tx">—</span>')

    receipt = resp.get("receipt") or {}
    return f"""
<details>
  <summary>
    <span class="nm">{_e(ep)}<small>{_e(title)}</small></span>
    <span class="px">${_e(r['price'])}</span>
    <span class="ck">{passed}/{len(checks)}</span>
    <span class="ck" style="color:var(--dim)">{_e(r.get('seconds'))}s</span>
    {tx_html}
  </summary>
  <div class="body">
    <div class="h3">What was sent</div>
    <pre>{_pretty(r.get('request'), 700)}</pre>
    <div class="h3">What came back</div>
    <pre>{_pretty(output)}</pre>
    <div class="h3">Outcome checks — is the answer actually good</div>
    <ul class="checks">{check_html}</ul>
    <div class="h3">Signed receipt</div>
    <pre>{_pretty({k: receipt.get(k) for k in ('algorithm','manifest_sha256','public_key')}, 500)}</pre>
    <div class="note">Rebuild the manifest from this response and verify the signature yourself —
    the exact steps are published at <code>/verify</code>. Nothing is needed from Horos but the
    public key.</div>
  </div>
</details>"""


def _a2a_section(a: dict) -> str:
    steps = "".join(
        f"""<div class="step"><div class="n">{i}</div><div class="t">
        <b>{_e(s['step'])}</b><span>{_e(s.get('detail',''))}</span>
        {f'<span class="tx"><a href="{EXPLORER}{s["tx"]}" target="_blank" rel="noopener">{_e(s["tx"][:24])}…</a></span>' if s.get('tx') else ''}
        </div></div>"""
        for i, s in enumerate(a["steps"], 1))
    return f"""
<h2>Agent to agent — hired in conversation, delivered on chain</h2>
<p class="gsub">The second listing gate is a live A2A probe: a reviewer registers a user agent, hires
the ASP, and waits. Agents that never apply on-chain are the most common rejection in the whole
review record. This is that flow, run end to end against Horos.</p>
<div class="flow">{steps}</div>
<details>
  <summary><span class="nm">The request and what was delivered<small>job {_e(a['job_id'][:20])}…
   · {_e(a['budget_usdt'])} USDT in escrow</small></span></summary>
  <div class="body">
    <div class="h3">What the buyer asked</div>
    <pre>{_e(a['request'])}</pre>
    <div class="h3">What Horos delivered</div>
    <pre>{_e(a['deliverable'])}</pre>
    <div class="note">Every figure names the service that produced it, and each of those services can
    be bought separately and compared against this answer. The language model chose which to run and
    wrote the prose — it produced no numbers.</div>
  </div>
</details>"""


def _architecture() -> str:
    return """
<h2>How it works</h2>
<p class="gsub">A forecast is unverifiable when you buy it and perfectly verifiable a day later.
That gap is the entire product.</p>
<div class="flow">
  <div class="step"><div class="n">1</div><div class="t"><b>Forecast on a clock</b>
    <span>Every hour, for BTC, ETH and SOL perpetuals, a foundation model generates hundreds of
    independent future paths from 512 candles of context. Its library averages those paths into one
    line before returning them; Horos keeps the whole distribution, which is where the bands, touch
    probabilities and tail percentiles live.</span></div></div>
  <div class="step"><div class="n">2</div><div class="t"><b>Write it down before the outcome exists</b>
    <span>The full distribution goes into a hash-chained ledger — each entry carries the previous
    entry's hash, so nothing can be removed, reordered or softened without every subsequent hash
    changing.</span></div></div>
  <div class="step"><div class="n">3</div><div class="t"><b>Commit the head on X Layer</b>
    <span>The chain head is written into transaction calldata <em>before the forecast's window
    closes</em>. A rebuilt history could not reproduce a hash that is already in a block.</span></div></div>
  <div class="step"><div class="n">4</div><div class="t"><b>Score it automatically when the window closes</b>
    <span>A job fetches what the market actually did and grades the forecast with proper scoring
    rules — CRPS, pinball loss, interval score, Brier — against a free baseline anyone could have
    used instead.</span></div></div>
  <div class="step"><div class="n">5</div><div class="t"><b>Publish, including the losses</b>
    <span>The scorecard is generated from the ledger, never written by hand. Scoring is automatic
    and the commitments are on chain, so a miss could not be hidden even if we wanted to.</span></div></div>
  <div class="step"><div class="n">6</div><div class="t"><b>Judge everyone else on the same terms</b>
    <span>Anyone can commit their own forecast to the same ledger and have it scored by the same
    code, in the same leaderboard — including when they beat us.</span></div></div>
</div>
<div class="arch"><pre>   x402 paywall
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
   |  Commit-reveal on X Layer  ·  public scorecard  ·  leaderboard |
   +--------------------------------------------------------------+</pre></div>
<div class="note">No backtest is published anywhere. The model was pre-trained across 45 global
exchanges to an unpublished cutoff, so any historical evaluation risks scoring it on its own
training data. The record starts empty and accumulates forward — which is slower, and the only
honest option.</div>"""


def _footer() -> str:
    return """
<footer>
  <p><strong>Horos</strong> — Marked before the outcome. ·
  <a href="/scorecard">the public record</a> ·
  <a href="/services">all 40 services</a> ·
  <a href="/ledger">download the ledger</a> ·
  <a href="/ledger/verify">verify the chain</a> ·
  <a href="/verify">check a receipt</a></p>
  <p>No buy, sell or hold. No direction calls. No backtested performance. Measurement with its
  uncertainty attached, and a public record of how often that uncertainty was right.</p>
</footer></div>
</body>
</html>"""


if __name__ == "__main__":
    print(page())
