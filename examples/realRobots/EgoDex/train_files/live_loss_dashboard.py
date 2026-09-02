#!/usr/bin/env python3
"""Live loss dashboard for starVLA training logs (no wandb needed).

Parses the rich/tqdm-wrapped training log for "Step N, Loss {...}" records
and serves a self-refreshing chart page on localhost.

Usage:
    python live_loss_dashboard.py [logfile] [port]
    # default: newest regressor*_train_*.log in the repo root, port 8899
"""

import glob
import json
import os
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = "/mnt/DATA2/yyg/starVLA"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# rich wraps dict lines with a deep indent; rejoin continuations without a space
CONT_RE = re.compile(r"\n {10,}")
STEP_RE = re.compile(r"Step (\d+), Loss[^{]*(\{[^}]*\})")
KV_RE = re.compile(r"'([\w/]+)':\s*([0-9.e+-]+)")


def resolve_log(argv):
    if len(argv) > 1:
        return argv[1]
    cands = sorted(glob.glob(os.path.join(ROOT, "regressor*_train_*.log")), key=os.path.getmtime)
    if not cands:
        sys.exit("no regressor*_train_*.log found; pass a path")
    return cands[-1]


LOG_PATH = resolve_log(sys.argv)
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8899

_cache = {"offset": 0, "tail": "", "records": []}


def parse_new():
    """Incrementally parse appended log content into loss records."""
    try:
        size = os.path.getsize(LOG_PATH)
    except OSError:
        return
    if size < _cache["offset"]:  # rotated
        _cache.update(offset=0, tail="", records=[])
    with open(LOG_PATH, "r", errors="replace") as f:
        f.seek(_cache["offset"])
        chunk = f.read()
        _cache["offset"] = f.tell()
    if not chunk:
        return
    text = _cache["tail"] + chunk
    _cache["tail"] = text[-4000:]  # keep overlap for records split across reads
    text = ANSI_RE.sub("", text)
    text = CONT_RE.sub("", text)
    seen = {r["step"] for r in _cache["records"]}
    for m in STEP_RE.finditer(text):
        step = int(m.group(1))
        if step in seen:
            continue
        kv = dict(KV_RE.findall(m.group(2)))
        try:
            rec = {"step": step, "t": time.time()}
            for k, v in kv.items():
                if k == "action_dit_loss":
                    rec["loss"] = float(v)
                elif k.startswith("learning_rate/"):
                    rec[k.split("/", 1)[1]] = float(v)
            if "loss" in rec:
                _cache["records"].append(rec)
                seen.add(step)
        except ValueError:
            continue  # number split by a wrap boundary; skip this sample
    _cache["records"].sort(key=lambda r: r["step"])


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>starVLA live loss</title>
<style>
:root { --bg:#fcfcfb; --ink:#0b0b0b; --muted:#52514e; --grid:#e4e3dc; --series:#2a78d6; --card:#ffffff; --border:#dddbd2; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#1a1a19; --ink:#f2f1ec; --muted:#a5a49c; --grid:#33332f; --series:#3987e5; --card:#222220; --border:#3a3a36; }
}
* { box-sizing:border-box; margin:0; }
body { background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,sans-serif; padding:24px; }
h1 { font-size:16px; font-weight:600; margin-bottom:2px; }
.sub { color:var(--muted); font-size:12px; margin-bottom:16px; }
.tiles { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }
.tile { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:10px 16px; min-width:120px; }
.tile .v { font-size:22px; font-weight:650; font-variant-numeric:tabular-nums; }
.tile .k { color:var(--muted); font-size:11px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:14px; margin-bottom:16px; }
.card h2 { font-size:13px; font-weight:600; margin-bottom:8px; }
svg { display:block; width:100%; }
.tip { position:fixed; pointer-events:none; background:var(--card); border:1px solid var(--border);
  border-radius:6px; padding:4px 8px; font-size:12px; display:none; box-shadow:0 2px 8px rgba(0,0,0,.15); }
</style></head><body>
<h1>starVLA live loss</h1>
<div class="sub" id="src"></div>
<div class="tiles">
  <div class="tile"><div class="v" id="tstep">–</div><div class="k">step</div></div>
  <div class="tile"><div class="v" id="tloss">–</div><div class="k">latest loss</div></div>
  <div class="tile"><div class="v" id="tema">–</div><div class="k">loss (EMA 50)</div></div>
  <div class="tile"><div class="v" id="tlr">–</div><div class="k">muon lr</div></div>
</div>
<div class="card"><h2>action loss</h2><div id="lossChart"></div></div>
<div class="card"><h2>learning rate (muon group)</h2><div id="lrChart"></div></div>
<div class="tip" id="tip"></div>
<script>
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v,d=4) => v==null ? "–" : (+v).toPrecision(d).replace(/\\.?0+$/,"");
function line(el, xs, ys, {ema=null, ylog=false}={}) {
  el.innerHTML = "";
  if (xs.length < 2) { el.textContent = "waiting for data…"; return; }
  const W = el.clientWidth || 800, H = 220, L = 52, R = 12, T = 10, B = 26;
  const yv = ylog ? ys.map(Math.log10) : ys;
  const ev = ema && (ylog ? ema.map(Math.log10) : ema);
  let ymin = Math.min(...yv, ...(ev||yv)), ymax = Math.max(...yv, ...(ev||yv));
  if (ymax === ymin) ymax = ymin + 1e-9;
  const pad = (ymax - ymin) * 0.06; ymin -= pad; ymax += pad;
  const X = x => L + (x - xs[0]) / (xs[xs.length-1] - xs[0]) * (W - L - R);
  const Y = y => T + (1 - (y - ymin) / (ymax - ymin)) * (H - T - B);
  const svg = document.createElementNS("http://www.w3.org/2000/svg","svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.style.height = H+"px";
  let g = "";
  for (let i = 0; i <= 4; i++) {
    const fy = ymin + (ymax - ymin) * i / 4, py = Y(fy);
    const lab = ylog ? fmt(Math.pow(10, fy), 3) : fmt(fy, 3);
    g += `<line x1="${L}" x2="${W-R}" y1="${py}" y2="${py}" stroke="${css('--grid')}" stroke-width="1"/>`;
    g += `<text x="${L-6}" y="${py+4}" text-anchor="end" font-size="11" fill="${css('--muted')}">${lab}</text>`;
  }
  for (let i = 0; i <= 4; i++) {
    const fx = xs[0] + (xs[xs.length-1] - xs[0]) * i / 4;
    g += `<text x="${X(fx)}" y="${H-8}" text-anchor="middle" font-size="11" fill="${css('--muted')}">${Math.round(fx)}</text>`;
  }
  const path = a => a.map((v,i) => (i?"L":"M") + X(xs[i]).toFixed(1) + " " + Y(v).toFixed(1)).join(" ");
  g += `<path d="${path(yv)}" fill="none" stroke="${css('--series')}" stroke-width="${ev?1:2}" opacity="${ev?0.35:1}"/>`;
  if (ev) g += `<path d="${path(ev)}" fill="none" stroke="${css('--series')}" stroke-width="2"/>`;
  g += `<line id="cross" y1="${T}" y2="${H-B}" stroke="${css('--muted')}" stroke-width="1" opacity="0"/>`;
  svg.innerHTML = g;
  el.appendChild(svg);
  const tip = document.getElementById("tip"), cross = svg.querySelector("#cross");
  svg.onmousemove = e => {
    const r = svg.getBoundingClientRect();
    const mx = (e.clientX - r.left) * W / r.width;
    let best = 0, bd = 1e18;
    for (let i = 0; i < xs.length; i++) { const d = Math.abs(X(xs[i]) - mx); if (d < bd) { bd = d; best = i; } }
    cross.setAttribute("x1", X(xs[best])); cross.setAttribute("x2", X(xs[best])); cross.setAttribute("opacity", 0.6);
    tip.style.display = "block"; tip.style.left = (e.clientX + 12) + "px"; tip.style.top = (e.clientY - 10) + "px";
    tip.textContent = `step ${xs[best]}: ${fmt(ys[best], 5)}` + (ema ? ` (ema ${fmt(ema[best],5)})` : "");
  };
  svg.onmouseleave = () => { tip.style.display = "none"; cross.setAttribute("opacity", 0); };
}
async function refresh() {
  try {
    const d = await (await fetch("data")).json();
    document.getElementById("src").textContent = d.log + "  ·  " + d.n + " samples  ·  updated " + new Date().toLocaleTimeString();
    if (!d.n) return;
    const xs = d.records.map(r => r.step), ys = d.records.map(r => r.loss);
    const ema = []; let m = ys[0];
    for (const y of ys) { m = m + (y - m) / 50; ema.push(m); }
    const last = d.records[d.records.length - 1];
    document.getElementById("tstep").textContent = last.step.toLocaleString();
    document.getElementById("tloss").textContent = fmt(last.loss, 4);
    document.getElementById("tema").textContent = fmt(ema[ema.length-1], 4);
    const lrKey = Object.keys(last).find(k => k.endsWith("_muon")) || Object.keys(last).find(k => k.startsWith("action_model") || k === "base");
    document.getElementById("tlr").textContent = lrKey ? fmt(last[lrKey], 3) : "–";
    line(document.getElementById("lossChart"), xs, ys, {ema});
    if (lrKey) line(document.getElementById("lrChart"), xs, d.records.map(r => r[lrKey] ?? null));
  } catch (e) { /* server restarting; retry next tick */ }
}
refresh(); setInterval(refresh, 10000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.rstrip("/") in ("", "/index.html"):
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        elif self.path.startswith("/data"):
            parse_new()
            body = json.dumps(
                {"log": os.path.basename(LOG_PATH), "n": len(_cache["records"]), "records": _cache["records"]}
            ).encode()
            ctype = "application/json"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"serving {LOG_PATH} on http://localhost:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
