"""The public record, rendered.

Generated entirely from the ledger. Nothing on this page is written by hand, so it cannot claim a
number that did not happen — the same construction as Doxa's proof deck, for the same reason.

It will show losses. That is the product working.
"""
from __future__ import annotations

import html
import json

from core.config import PROVENANCE, SERVICE_NAME, TAGLINE, get_settings
from core.markets import HORIZONS, SCORED_MARKETS, coverage_note
from core.scoring import BAND_LEVELS, aggregate

S = get_settings()


def _e(x) -> str:
    return html.escape(str(x), quote=True)


def _num(v, digits: int = 4, dash: str = "—") -> str:
    if v is None:
        return dash
    try:
        f = float(v)
    except (TypeError, ValueError):
        return _e(v)
    if abs(f) >= 1000:
        return f"{f:,.0f}"
    return f"{f:.{digits}f}".rstrip("0").rstrip(".") or "0"


def _pct(v, dash: str = "—") -> str:
    return dash if v is None else f"{float(v) * 100:.1f}%"


def _signed_pct(v) -> str:
    if v is None:
        return "—"
    f = float(v) * 100
    return f"{f:+.1f}%"


CSS = """
*{box-sizing:border-box}
:root{
  --bg:#0f1012;--panel:#17181c;--line:#26282e;--ink:#e8e6e3;--dim:#95928d;--faint:#6a6862;
  --good:#5fb37a;--bad:#d4685f;--warn:#d0a24c;--accent:#c9b896;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
}
@media(prefers-color-scheme:light){:root{
  --bg:#faf9f7;--panel:#fff;--line:#e3e0da;--ink:#1a1917;--dim:#6b6862;--faint:#95928d;
  --good:#2f7d4f;--bad:#b03a30;--warn:#966d18;--accent:#6b5a3a;}}
:root[data-theme=dark]{--bg:#0f1012;--panel:#17181c;--line:#26282e;--ink:#e8e6e3;--dim:#95928d;
  --faint:#6a6862;--good:#5fb37a;--bad:#d4685f;--warn:#d0a24c;--accent:#c9b896}
:root[data-theme=light]{--bg:#faf9f7;--panel:#fff;--line:#e3e0da;--ink:#1a1917;--dim:#6b6862;
  --faint:#95928d;--good:#2f7d4f;--bad:#b03a30;--warn:#966d18;--accent:#6b5a3a}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.65 ui-sans-serif,-apple-system,"Segoe UI",Inter,system-ui,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:56px 22px 96px}
header{border-bottom:1px solid var(--line);padding-bottom:34px;margin-bottom:40px}
.brand{display:flex;align-items:center;gap:16px;flex-wrap:wrap}
h1{font-size:34px;margin:0;letter-spacing:-.02em;font-weight:600}
.tag{color:var(--accent);font-size:15px;letter-spacing:.01em}
.lede{color:var(--dim);margin-top:18px;max-width:70ch;font-size:15px}
.prov{color:var(--faint);margin-top:14px;max-width:76ch;font-size:13.5px;font-style:italic}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.13em;color:var(--faint);
  margin:52px 0 16px;font-weight:600}
h3{font-size:16px;margin:26px 0 10px;font-weight:600}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:22px 24px}
.grid{display:grid;gap:14px}
.g4{grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}
.g2{grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
.stat{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;
  min-width:0}
.stat .k{font-size:11.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--faint)}
.stat .v{font-size:25px;margin-top:7px;font-variant-numeric:tabular-nums;letter-spacing:-.01em;
  overflow-wrap:anywhere}
.stat .s{font-size:12.5px;color:var(--dim);margin-top:5px;overflow-wrap:anywhere}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:520px}
th,td{padding:11px 15px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-size:11.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--faint);
  font-weight:600}
tr:last-child td{border-bottom:none}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;font-family:var(--mono);
  font-size:13px}
.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.mono{font-family:var(--mono);font-size:12.5px;overflow-wrap:anywhere}
a{color:var(--accent)}a:hover{opacity:.8}
.empty{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--warn);
  border-radius:10px;padding:24px 26px;color:var(--dim);max-width:78ch}
.empty strong{color:var(--ink);display:block;margin-bottom:9px;font-size:16px}
.note{color:var(--dim);font-size:13.5px;margin-top:14px;max-width:76ch}
.bars{display:flex;gap:3px;align-items:flex-end;height:56px;margin-top:12px}
.bars div{flex:1;background:var(--accent);opacity:.75;border-radius:2px 2px 0 0;min-height:2px}
.cov{position:relative;height:9px;background:var(--line);border-radius:5px;margin-top:7px;
  overflow:hidden}
.cov i{position:absolute;left:0;top:0;bottom:0;background:var(--accent);border-radius:5px}
.cov b{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink);opacity:.8}
footer{margin-top:76px;padding-top:26px;border-top:1px solid var(--line);color:var(--faint);
  font-size:13px}
code{font-family:var(--mono);font-size:12.5px;background:var(--bg);border:1px solid var(--line);
  padding:1.5px 5px;border-radius:4px}
"""


def render(ctx) -> str:
    counts = ctx.counts()
    anchors = ctx.anchors()
    rows = _scored(ctx)
    ok, problems = ctx.ledger.verify()

    parts = [_head(), _header(counts, anchors, ok, problems)]
    if not rows:
        parts.append(_empty(counts, anchors))
    else:
        parts.append(_overall(rows))
        parts.append(_coverage(rows))
        parts.append(_by_dimension(rows))
        parts.append(_recent(rows))
    parts.append(_anchors_section(anchors))
    parts.append(_coverage_note())
    parts.append(_footer())
    return "\n".join(parts)


def _scored(ctx) -> list[dict]:
    issued, voided = {}, set()
    for e in ctx.ledger:
        if e.kind == "issued":
            issued[e.forecast_id] = e.body
        elif e.kind == "void":
            voided.add(e.body["forecast_id"])
    out = []
    for e in ctx.ledger:
        if e.kind == "scored" and e.forecast_id not in voided and e.forecast_id in issued:
            out.append({"id": e.forecast_id, "issued": issued[e.forecast_id],
                        "scores": e.body.get("scores", {}), "actual": e.body.get("actual", {}),
                        "at": e.at})
    return out


def _head() -> str:
    return (f"<title>{SERVICE_NAME} — the record</title>\n<style>{CSS}</style>\n"
            f'<div class="wrap">')


def _header(counts: dict, anchors: list, ok: bool, problems: list) -> str:
    chain = "verifies" if ok else "FAILS VERIFICATION"
    cls = "good" if ok else "bad"
    prob = ""
    if not ok:
        prob = ('<div class="note bad">The hash chain does not verify. This is a fault in Horos '
                'and it is shown rather than hidden: ' + _e("; ".join(problems[:3])) + "</div>")
    return f"""
<header>
  <div class="brand"><h1>{SERVICE_NAME}</h1><span class="tag">{_e(TAGLINE)}</span></div>
  <p class="lede">Every forecast Horos issues is written to a hash-chained ledger and its head is
  committed to X Layer <strong>before the window closes</strong>. When the window closes the
  forecast is scored automatically against what the market actually did. This page is generated from
  that ledger — nothing on it is written by hand, and it shows losses as readily as wins.</p>
  <p class="prov">{_e(PROVENANCE)}</p>
  <div class="grid g4" style="margin-top:26px">
    <div class="stat"><div class="k">Ledger entries</div><div class="v">{counts['entries']}</div>
      <div class="s"><span class="{cls}">chain {chain}</span></div></div>
    <div class="stat"><div class="k">Forecasts issued</div><div class="v">{counts['issued']}</div>
      <div class="s">{counts['voided']} withdrawn, on the record</div></div>
    <div class="stat"><div class="k">Scored</div><div class="v">{counts['scored']}</div>
      <div class="s">graded against realised candles</div></div>
    <div class="stat"><div class="k">On-chain anchors</div><div class="v">{len(anchors)}</div>
      <div class="s">X Layer mainnet</div></div>
  </div>
  {prob}
</header>"""


def _empty(counts: dict, anchors: list) -> str:
    anchored = (f"The ledger head has been committed on chain {len(anchors)} time(s) already, so "
                f"the record provably starts before any of these forecasts can be graded."
                if anchors else "")
    return f"""
<h2>The record</h2>
<div class="empty">
  <strong>No forecast has been scored yet.</strong>
  {counts['issued']} forecast(s) are committed and waiting for their windows to close. This page
  publishes empty rather than showing a backtest: Kronos was pre-trained across 45 global exchanges
  to an unpublished cutoff, so any historical evaluation risks scoring the model on its own training
  data. The record starts here and accumulates forward. {anchored}
  <div class="note">Every figure that appears here will be reproducible — download
  <code>/ledger</code>, run <code>core/scoring.py</code>, and you get these numbers without
  trusting us.</div>
</div>"""


def _overall(rows: list[dict]) -> str:
    agg = aggregate([r["scores"] for r in rows])
    skill = agg.get("crps_skill_vs_empirical")
    pskill = agg.get("crps_skill_vs_persistence")
    n = agg["count"]
    scls = "good" if (skill or 0) > 0 else "bad"
    # Coloured by sign, like the other one. Leaving a negative figure in neutral white next to a
    # green positive one reads as "one good result and one neutral" when it is one good and one bad.
    pcls = "good" if (pskill or 0) > 0 else "bad"

    # The verdict has to speak to *both* baselines. An earlier version reported only the random-walk
    # comparison, so a page showing +11.3% against the random walk and -79.8% against persistence
    # summarised itself as "the model is beating a free empirical random walk" — true, and a
    # materially misleading thing to lead with.
    parts = []
    parts.append("The model is beating an empirical random walk." if (skill or 0) > 0
                 else "The model is <strong>not</strong> beating a free empirical random walk.")
    if pskill is not None:
        parts.append("It is beating a naive “price stays where it is” forecast."
                     if pskill > 0 else
                     "It is <strong>losing</strong> to a naive “price stays where it is” "
                     "forecast, which costs nothing.")
    if n < 30:
        parts.append(f"All of this rests on {n} scored forecast(s) — far too few to distinguish "
                     f"skill from luck. Treat none of it as evidence yet.")
    elif (skill or 0) <= 0:
        parts.append("On this evidence you should not pay for a forecast here — resample recent "
                     "returns yourself and you will do at least as well.")
    verdict = " ".join(parts)

    pit = agg.get("pit") or {}
    bars = ""
    # A ten-bin histogram drawn from a handful of observations looks broken rather than sparse, and
    # invites reading structure into noise. Below the threshold the counts are stated instead.
    if pit.get("histogram") and pit.get("count", 0) >= 20:
        top = max(pit["histogram"]) or 1
        bars = ('<div class="bars">'
                + "".join(f'<div style="height:{max(2, int(100 * c / top))}%"></div>'
                          for c in pit["histogram"]) + "</div>")
    elif pit.get("count"):
        seen = ", ".join(f"{p:.2f}" for p in sorted(pit.get("values") or []))
        bars = (f'<div class="note">Not drawn yet: a ten-bin calibration histogram needs at least '
                f'20 scored forecasts to mean anything and there are {pit["count"]}. '
                f'The values so far are {seen or "recorded in the ledger"}.</div>')

    # The shape note only makes sense beside an actual histogram. Printed under the "not drawn yet"
    # message it described a picture that was not on the page.
    if not bars:
        calibration_block = ""
    elif pit.get("count", 0) >= 20:
        calibration_block = ('<h3>Calibration (PIT)</h3>' + bars +
                             '<div class="note">Flat is calibrated. A U shape means the bands are '
                             'too narrow; a hump in the middle means they are too wide.</div>')
    else:
        calibration_block = ('<h3>Calibration (PIT)</h3>' + bars +
                             '<div class="note">Each value is where the outcome fell inside its own '
                             'predicted distribution. Over many forecasts these should spread evenly '
                             'between 0 and 1; clustering at the ends would mean the bands are too '
                             'narrow.</div>')
    return f"""
<h2>Overall</h2>
<div class="grid g4">
  <div class="stat"><div class="k">Scored forecasts</div><div class="v">{agg['count']}</div></div>
  <div class="stat"><div class="k">Mean CRPS</div>
    <div class="v">{_num(agg.get('mean_crps'), 2)}</div>
    <div class="s">price units, lower is better</div></div>
  <div class="stat"><div class="k">Skill vs random walk</div>
    <div class="v {scls}">{_signed_pct(skill)}</div>
    <div class="s">above zero beats the free baseline</div></div>
  <div class="stat"><div class="k">Skill vs persistence</div>
    <div class="v {pcls}">{_signed_pct(pskill)}</div>
    <div class="s">against "price stays where it is"</div></div>
</div>
<div class="note">{verdict}</div>
{calibration_block}"""


def _coverage(rows: list[dict]) -> str:
    agg = aggregate([r["scores"] for r in rows])
    cov = agg.get("coverage") or {}
    if not cov:
        return ""
    body = []
    for key in sorted(cov, key=float):
        c = cov[key]
        nominal, emp = float(c["nominal"]), float(c["empirical"])
        delta = emp - nominal
        cls = "good" if abs(delta) <= 0.05 else ("warn" if abs(delta) <= 0.12 else "bad")
        body.append(f"""<tr>
  <td>{_pct(nominal)} band</td>
  <td class="n">{c['n']}</td>
  <td class="n {cls}">{_pct(emp)}</td>
  <td class="n {cls}">{_signed_pct(delta)}</td>
  <td style="min-width:170px"><div class="cov"><i style="width:{min(100, emp * 100):.0f}%"></i>
    <b style="left:{min(100, nominal * 100):.0f}%"></b></div></td>
  <td class="n">{_num(c['mean_interval_score'], 1)}</td>
  <td class="n">{_num(c['mean_width'], 1)}</td>
</tr>""")
    return f"""
<h2>Coverage — what the bands claimed against what they delivered</h2>
<div class="tablewrap"><table>
<thead><tr><th>Band</th><th class="n">n</th><th class="n">Empirical</th><th class="n">Error</th>
<th>Nominal ▏ actual</th><th class="n">Interval score</th><th class="n">Mean width</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></div>
<div class="note">A calibrated 90% band contains the outcome about 90% of the time. Below that the
bands are too narrow and the model is over-confident; above it they are too wide and the forecast is
less useful than it looks. The interval score charges for width and for misses together, so neither
can be gamed.</div>"""


def _by_dimension(rows: list[dict]) -> str:
    def table(title: str, key: str, order: list[str]) -> str:
        body = []
        for name in order:
            subset = [r["scores"] for r in rows if r["issued"].get(key) == name]
            if not subset:
                continue
            a = aggregate(subset)
            skill = a.get("crps_skill_vs_empirical")
            cls = "good" if (skill or 0) > 0 else "bad"
            verdict = ("beats the baseline" if (skill or 0) > 0.05 else
                       "marginal" if (skill or 0) > 0 else "does not beat the baseline")
            if a["count"] < 30:
                verdict, cls = f"provisional ({a['count']} scored)", "warn"
            body.append(f"""<tr><td>{_e(name)}</td><td class="n">{a['count']}</td>
              <td class="n">{_num(a.get('mean_crps'), 2)}</td>
              <td class="n {cls}">{_signed_pct(skill)}</td>
              <td class="{cls}">{verdict}</td></tr>""")
        if not body:
            return ""
        return f"""<h3>{title}</h3><div class="tablewrap"><table>
        <thead><tr><th>{title}</th><th class="n">Scored</th><th class="n">Mean CRPS</th>
        <th class="n">Skill</th><th>Reading</th></tr></thead>
        <tbody>{''.join(body)}</tbody></table></div>"""

    return ("<h2>Where the model is and is not useful</h2>"
            + table("Symbol", "symbol", [m.symbol for m in SCORED_MARKETS])
            + table("Horizon", "horizon", sorted(HORIZONS, key=lambda h: HORIZONS[h]))
            + '<div class="note">Fewer than 30 scored forecasts cannot distinguish skill from luck, '
              'so those rows are marked provisional rather than given a verdict.</div>')


def _recent(rows: list[dict], limit: int = 25) -> str:
    body = []
    for r in rows[-limit:][::-1]:
        s, i = r["scores"], r["issued"]
        band = (s.get("bands") or {}).get("0.9") or {}
        hit = band.get("covered")
        cls = "good" if hit else ("bad" if hit is False else "")
        mark = "inside" if hit else ("outside" if hit is False else "—")
        body.append(f"""<tr>
          <td class="mono">{_e(r['id'][:12])}</td>
          <td>{_e(i.get('symbol', ''))}</td>
          <td>{_e(i.get('horizon', ''))}</td>
          <td class="mono">{_e((i.get('window') or {}).get('close', ''))}</td>
          <td class="n">{_num((i.get('distribution') or {}).get('anchor_price'), 2)}</td>
          <td class="n">{_num(s.get('actual_close'), 2)}</td>
          <td class="n">{_num(s.get('crps'), 2)}</td>
          <td class="n {cls}">{mark}</td>
          <td class="n">{_num(s.get('pit'), 3)}</td></tr>""")
    return f"""
<h2>Most recent scored forecasts</h2>
<div class="tablewrap"><table>
<thead><tr><th>Forecast</th><th>Symbol</th><th>Horizon</th><th>Window closed</th>
<th class="n">Anchor</th><th class="n">Actual</th><th class="n">CRPS</th><th class="n">90% band</th>
<th class="n">PIT</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></div>
<div class="note">Every row here can be checked: the forecast was committed before the window
opened, and its full distribution is in the ledger.</div>"""


def _anchors_section(anchors: list[dict]) -> str:
    if not anchors:
        return """
<h2>On-chain anchoring</h2>
<div class="empty"><strong>Not yet anchored.</strong>
The ledger hash-chains, so it is tamper-evident to anyone holding a copy — but nothing yet stops the
whole chain being rebuilt from scratch. Until a head reaches an X Layer block, this record is
<em>not</em> provably prior to its own outcomes, and it says so here rather than implying otherwise.
</div>"""
    body = []
    for a in anchors[-10:][::-1]:
        tx = a.get("tx", "")
        body.append(f"""<tr><td class="mono">{_e(a.get('at', ''))}</td>
          <td class="mono">{_e((a.get('head') or '')[7:23])}…</td>
          <td class="mono"><a href="{_e(S.xlayer_explorer + tx)}" rel="noopener">{_e(tx[:22])}…</a>
          </td></tr>""")
    return f"""
<h2>On-chain anchoring — {len(anchors)} commitments on X Layer</h2>
<div class="tablewrap"><table>
<thead><tr><th>Committed</th><th>Ledger head</th><th>Transaction</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></div>
<div class="note">The calldata of each transaction is 46 bytes: <code>HOROS</code>, a version byte,
the 32-byte ledger head, and the entry count. Pull the input field from any X Layer explorer and
decode it — it must match a head in the ledger you can download at <code>/ledger</code>. Full
instructions at <code>/verify</code>.</div>"""


def _coverage_note() -> str:
    n = coverage_note()
    sel = n["selection"]
    rows = "".join(
        f"""<tr><td>{_e(s['symbol'])}</td><td>{_e(s['label'])}</td>
        <td class="n">{_pct(s['share_of_okx_usdt_perp_notional'])}</td>
        <td class="n">${s['notional_24h_usd'] / 1e9:.2f}B</td></tr>"""
        for s in n["symbols"])
    return f"""
<h2>What is scored, and why exactly this</h2>
<div class="tablewrap"><table>
<thead><tr><th>Symbol</th><th>Market</th><th class="n">Share of book</th>
<th class="n">24h notional</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="note">Selected by {_e(sel['basis'])}, measured on {_e(sel['measured_on'])} across
{sel['instruments_measured']} instruments totalling ${sel['book_notional_24h_usd'] / 1e9:.2f}B.
{_e(sel['note'])}</div>
<div class="note">Timeframe {_e(n['timeframe'])}; horizons {_e(', '.join(n['horizons']))}; the model
sees {n['context_candles']} candles of context, which is {_e(n['context_span'])}. At this cadence the
record takes {n['scored_forecasts_per_week']:,} scored forecasts a week.</div>"""


def _footer() -> str:
    return f"""
<footer>
  <p><strong>{SERVICE_NAME}</strong> — {_e(TAGLINE)} ·
  <a href="/services">services</a> · <a href="/ledger">download the ledger</a> ·
  <a href="/ledger/verify">verify the chain</a> · <a href="/verify">check a receipt</a></p>
  <p>No backtest is published anywhere on this site. No buy, sell or hold. No directional calls.
  Measurement with its uncertainty attached, and a record of how often that uncertainty was right.</p>
</footer></div>
<script>
// Respect an explicit theme choice if the host page sets one; otherwise follow the system.
(function(){{var t=document.documentElement.getAttribute('data-theme');if(!t){{}}}})();
</script>"""
