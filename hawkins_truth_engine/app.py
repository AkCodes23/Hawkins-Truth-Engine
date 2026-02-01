from __future__ import annotations

import asyncio

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .analyzers.claims import analyze_claims
from .analyzers.linguistic import analyze_linguistic
from .analyzers.source_intel import analyze_source
from .analyzers.statistical import analyze_statistical
from .explain import generate_explanation
from .ingest import build_document
from .reasoning import aggregate
from .schemas import AnalyzeRequest, AnalysisResponse


app = FastAPI(title="Hawkins Truth Engine (POC)")


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return """
<!doctype html>
<html>
  <head>
    <meta charset='utf-8' />
    <meta name='viewport' content='width=device-width, initial-scale=1' />
    <title>Hawkins Truth Engine (POC)</title>
    <style>
      :root { --bg:#0f172a; --panel:#111c33; --text:#e5e7eb; --muted:#9ca3af; --accent:#f59e0b; --good:#22c55e; --bad:#ef4444; }
      body { margin:0; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; background: radial-gradient(1000px 800px at 20% 0%, #1f2a44, var(--bg)); color: var(--text); }
      .wrap { max-width: 1100px; margin: 0 auto; padding: 24px; }
      .hdr { display:flex; align-items:baseline; justify-content:space-between; gap:16px; }
      h1 { font-size: 22px; margin: 0; letter-spacing: 0.4px; }
      .tag { color: var(--muted); font-size: 12px; }
      .grid { display:grid; grid-template-columns: 1.2fr 1fr; gap: 16px; margin-top: 16px; }
      @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
      .card { background: color-mix(in oklab, var(--panel) 86%, black 14%); border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; padding: 14px; }
      textarea { width: 100%; min-height: 220px; background: rgba(0,0,0,0.25); color: var(--text); border: 1px solid rgba(255,255,255,0.15); border-radius: 10px; padding: 10px; }
      select, button, input { background: rgba(0,0,0,0.25); color: var(--text); border: 1px solid rgba(255,255,255,0.15); border-radius: 10px; padding: 10px; }
      button { cursor:pointer; }
      .row { display:flex; gap: 10px; align-items:center; flex-wrap: wrap; }
      .meter { height: 12px; background: rgba(255,255,255,0.1); border-radius: 999px; overflow:hidden; }
      .bar { height:100%; width: 0%; background: linear-gradient(90deg, var(--bad), var(--accent), var(--good)); }
      .kv { display:grid; grid-template-columns: 140px 1fr; gap: 8px; font-size: 13px; }
      .muted { color: var(--muted); }
      pre { white-space: pre-wrap; word-break: break-word; font-size: 12px; background: rgba(0,0,0,0.25); padding: 10px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.12); max-height: 420px; overflow:auto; }
      .pill { display:inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; border: 1px solid rgba(255,255,255,0.15); }
    </style>
  </head>
  <body>
    <div class='wrap'>
      <div class='hdr'>
        <h1>Hawkins Truth Engine</h1>
        <div class='tag'>Explainable credibility reasoning (POC)</div>
      </div>
      <div class='grid'>
        <div class='card'>
          <div class='row'>
            <label class='muted'>Input type</label>
            <select id='inputType'>
              <option value='raw_text'>Raw text</option>
              <option value='url'>URL</option>
              <option value='social_post'>Social post</option>
            </select>
            <button onclick='run()'>Analyze</button>
            <span id='status' class='muted'></span>
          </div>
          <textarea id='content' placeholder='Paste text or URL...'></textarea>
          <div class='muted' style='margin-top:8px'>Output is evidence-based and uncertainty-aware; it does not claim ground truth.</div>
        </div>
        <div class='card'>
          <div class='kv'>
            <div class='muted'>Credibility</div>
            <div><span id='score' class='pill'>--</span></div>
            <div class='muted'>World</div>
            <div><span id='world' class='pill'>--</span></div>
            <div class='muted'>Verdict</div>
            <div><span id='verdict' class='pill'>--</span></div>
            <div class='muted'>Confidence</div>
            <div><span id='conf' class='pill'>--</span></div>
          </div>
          <div style='margin-top:10px' class='meter'><div id='bar' class='bar'></div></div>
          <div style='margin-top:10px' class='muted'>Key evidence (≤10s view)</div>
          <pre id='bullets'></pre>
        </div>
      </div>
      <div class='card' style='margin-top:16px'>
        <div class='row'><span class='muted'>Full response</span><span class='pill'>JSON</span></div>
        <pre id='out'></pre>
      </div>
    </div>
    <script>
      async function run() {
        const input_type = document.getElementById('inputType').value;
        const content = document.getElementById('content').value;
        document.getElementById('status').innerText = 'running...';
        document.getElementById('out').innerText = '';
        const r = await fetch('/analyze', { method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify({input_type, content}) });
        const j = await r.json();
        document.getElementById('status').innerText = r.ok ? 'done' : 'error';
        if (!r.ok) { document.getElementById('out').innerText = JSON.stringify(j, null, 2); return; }
        const score = j.aggregation.credibility_score;
        document.getElementById('score').innerText = score;
        document.getElementById('world').innerText = j.aggregation.world_label;
        document.getElementById('verdict').innerText = j.aggregation.verdict;
        document.getElementById('conf').innerText = Math.round(j.aggregation.confidence*100) + '%';
        document.getElementById('bar').style.width = score + '%';
        document.getElementById('bullets').innerText = j.explanation.verdict_text + '\n\n' + j.explanation.evidence_bullets.map(x => '- ' + x).join('\n') + '\n\nUncertainty: ' + (j.aggregation.uncertainty_flags || []).join(', ');
        document.getElementById('out').innerText = JSON.stringify(j, null, 2);
      }
    </script>
  </body>
</html>
"""


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalyzeRequest) -> AnalysisResponse:
    doc = await build_document(req.input_type, req.content)
    linguistic = analyze_linguistic(doc)
    statistical = analyze_statistical(doc)
    source = await analyze_source(doc)
    claims = await analyze_claims(doc)
    aggregation = aggregate(linguistic, statistical, source, claims)
    explanation = generate_explanation(
        doc, linguistic, statistical, source, claims, aggregation
    )
    return AnalysisResponse(
        document=doc,
        linguistic=linguistic,
        statistical=statistical,
        source=source,
        claims=claims,
        aggregation=aggregation,
        explanation=explanation,
    )


def run() -> None:
    import uvicorn

    uvicorn.run(
        "hawkins_truth_engine.app:app", host="127.0.0.1", port=8000, reload=False
    )


if __name__ == "__main__":
    run()
