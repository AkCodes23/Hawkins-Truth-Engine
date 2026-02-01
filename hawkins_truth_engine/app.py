from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .analyzers.claims import analyze_claims
from .analyzers.linguistic import analyze_linguistic
from .analyzers.source_intel import analyze_source
from .analyzers.statistical import analyze_statistical
from .explain import generate_explanation
from .ingest import build_document
from .reasoning import aggregate
from .schemas import AnalyzeRequest, AnalysisResponse


app = FastAPI(title="Hawkins Truth Engine (POC)")

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", response_class=HTMLResponse)
async def home() -> FileResponse:
    """Serve the Stranger Things themed frontend."""
    return FileResponse(STATIC_DIR / "index.html")


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
