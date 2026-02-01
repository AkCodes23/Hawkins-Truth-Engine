import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Union
from uuid import uuid4

from fastapi import FastAPI, Query, HTTPException, Request, Body
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger(__name__)

from .analyzers.claims import analyze_claims
from .analyzers.linguistic import analyze_linguistic
from .analyzers.source_intel import analyze_source
from .analyzers.statistical import analyze_statistical
from .explain import generate_explanation
from .graph.claim_graph import build_claim_graph
from .graph.evidence_graph import build_evidence_graph
from .graph.query_interface import GraphQueryInterface
from .ingest import build_document
from .reasoning import aggregate
from .schemas import (
    AnalyzeRequest,
    AnalysisResponse,
    ClaimGraph,
    EvidenceGraph,
)
from .validators import validate_analyze_request, ValidationError


app = FastAPI(title="Hawkins Truth Engine (POC)", version="1.0")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda request, exc: JSONResponse(
    status_code=429,
    content={"detail": "Rate limit exceeded"},
))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared graph query interface
_graph_query_interface = GraphQueryInterface()

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"

# API version
API_VERSION = "1.0"


@app.get("/", response_class=HTMLResponse)
async def home() -> FileResponse:
    """Serve the Stranger Things themed frontend."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/presentation", response_class=HTMLResponse)
async def presentation() -> FileResponse:
    """Serve the project presentation/explainer page."""
    return FileResponse(STATIC_DIR / "presentation.html")


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Health status: 'healthy' or 'degraded'")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Check timestamp")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    endpoints_tested: dict[str, bool] = Field(..., description="Status of key endpoints")


class StatusResponse(BaseModel):
    """API status response."""
    status: str
    version: str
    timestamp: datetime


_start_time = time.time()


@app.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """Health check endpoint for monitoring and load balancers."""
    uptime = time.time() - _start_time
    endpoints_tested = {
        "analyze": True,
        "graphs": True,
        "calibration": True,
    }
    
    status = "healthy" if all(endpoints_tested.values()) else "degraded"
    logger.debug(f"Health check: status={status}, uptime_seconds={uptime:.2f}")
    
    return HealthCheckResponse(
        status=status,
        version=API_VERSION,
        timestamp=datetime.now(),
        uptime_seconds=uptime,
        endpoints_tested=endpoints_tested,
    )


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    """Get current API status."""
    return StatusResponse(
        status="operational",
        version=API_VERSION,
        timestamp=datetime.now(),
    )


class APITestResponse(BaseModel):
    """Response for API tests."""
    status: str
    endpoints: dict[str, str]
    timestamp: datetime


@app.get("/test-apis", response_model=APITestResponse)
async def test_apis() -> APITestResponse:
    """Test all available API endpoints and report status."""
    endpoints = {
        "analyze": "POST /analyze",
        "graphs_export": "POST /graphs/export",
        "graphs_metrics": "POST /graphs/metrics",
        "calibration_status": "GET /calibration/status",
        "calibration_calibrate": "POST /calibration/calibrate",
        "health": "GET /health",
        "status": "GET /status",
    }
    
    return APITestResponse(
        status="operational",
        endpoints=endpoints,
        timestamp=datetime.now(),
    )


@app.post("/analyze", response_model=AnalysisResponse)
@limiter.limit("30/minute")
async def analyze(request: Request, req: Annotated[AnalyzeRequest, Body()]) -> AnalysisResponse:
    """Analyze content for credibility and misinformation."""
    request_id = str(uuid4())
    start_time = time.time()
    logger.info(f"Analyze request started: request_id={request_id}, input_type={req.input_type}")
    
    # Validate input before processing
    try:
        validate_analyze_request(req)
        logger.debug(f"Request {request_id} validation passed")
    except ValidationError as e:
        logger.warning(f"Request {request_id} validation failed: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    
    doc = await build_document(req.input_type, req.content)
    
    # Run independent analysis tasks in parallel
    # Note: analyze_linguistic and analyze_statistical are synchronous for now.
    # We run the async tasks concurrently.
    source_task = analyze_source(doc)
    claims_task = analyze_claims(doc)
    
    # Gather async results
    source, claims = await asyncio.gather(source_task, claims_task)
    
    # Run sync analyzers
    linguistic = analyze_linguistic(doc)
    statistical = analyze_statistical(doc)
    
    aggregation = aggregate(linguistic, statistical, source, claims)
    explanation = generate_explanation(
        doc, linguistic, statistical, source, claims, aggregation
    )
    
    # Build graphs if requested
    claim_graph = None
    evidence_graph = None
    if req.include_graphs:
        claim_graph = build_claim_graph(doc, claims, source)
        # Build external corroboration data from claims for evidence graph
        external_corroboration = _build_corroboration_data(claims)
        evidence_graph = build_evidence_graph(claims, external_corroboration)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Analysis {request_id} completed in {elapsed_time:.2f}s")
    
    return AnalysisResponse(
        request_id=request_id,
        timestamp=datetime.now(),
        version=API_VERSION,
        document=doc,
        linguistic=linguistic,
        statistical=statistical,
        source=source,
        claims=claims,
        aggregation=aggregation,
        explanation=explanation,
        claim_graph=claim_graph,
        evidence_graph=evidence_graph,
    )


def _build_corroboration_data(claims) -> dict:
    """Build external corroboration data from claims for evidence graph construction.
    
    Args:
        claims: ClaimsOutput object containing extracted claims and their support status.
        
    Returns:
        dict: Mapping of claim IDs to corroboration data including confidence, support status,
              citations, and quality flags.
              
    Note:
        This function maps claim support statuses to confidence levels for use in evidence
        graph construction, enabling visualization of evidentiary relationships.
    """
    corroboration = {}
    for claim_item in claims.claim_items:
        # Determine confidence based on support status
        confidence_map = {
            "supported": 0.85,
            "unsupported": 0.15,
            "contested": 0.5,
            "unverifiable": 0.3
        }
        confidence = confidence_map.get(claim_item.support, 0.5)
        
        # Determine supported status
        supported = None
        if claim_item.support == "supported":
            supported = True
        elif claim_item.support == "unsupported":
            supported = False
        
        corroboration[claim_item.id] = {
            "confidence": confidence,
            "supported": supported,
            "citations": claim_item.citations,
            "quality_flags": claim_item.quality_flags,
        }
    return corroboration


# ============================================================================
# Graph API Endpoints
# ============================================================================

class GraphExportRequest(BaseModel):
    """Request model for graph export."""
    claim_graph: ClaimGraph | None = None
    evidence_graph: EvidenceGraph | None = None
    format: Literal["json", "graphml", "dot"] = "json"


class GraphExportResponse(BaseModel):
    """Response model for graph export."""
    format: str
    claim_graph_export: str | None = None
    evidence_graph_export: str | None = None


class GraphMetricsResponse(BaseModel):
    """Response model for graph metrics."""
    claim_graph_metrics: dict | None = None
    evidence_graph_metrics: dict | None = None


@app.post("/graphs/export", response_model=GraphExportResponse)
@limiter.limit("100/minute")
async def export_graphs(request: Request, req: Annotated[GraphExportRequest, Body()]) -> GraphExportResponse:
    """Export graphs in various formats (JSON, GraphML, DOT)."""
    claim_export = None
    evidence_export = None
    
    if req.claim_graph:
        claim_export = _graph_query_interface.export_graph(req.claim_graph, req.format)
    
    if req.evidence_graph:
        evidence_export = _graph_query_interface.export_graph(req.evidence_graph, req.format)
    
    return GraphExportResponse(
        format=req.format,
        claim_graph_export=claim_export,
        evidence_graph_export=evidence_export,
    )


class GraphMetricsRequest(BaseModel):
    """Request model for calculating graph metrics."""
    claim_graph: ClaimGraph | None = None
    evidence_graph: EvidenceGraph | None = None


@app.post("/graphs/metrics", response_model=GraphMetricsResponse)
@limiter.limit("100/minute")
async def calculate_graph_metrics(request: Request, req: Annotated[GraphMetricsRequest, Body()]) -> GraphMetricsResponse:
    """Calculate centrality and density metrics for graphs."""
    claim_metrics = None
    evidence_metrics = None
    
    if req.claim_graph:
        centrality = _graph_query_interface.calculate_node_centrality(req.claim_graph)
        density = _graph_query_interface.calculate_graph_density(req.claim_graph)
        claim_metrics = {
            "centrality": centrality,
            "density": density,
            "node_count": len(req.claim_graph.nodes),
            "edge_count": len(req.claim_graph.edges),
        }
    
    if req.evidence_graph:
        centrality = _graph_query_interface.calculate_node_centrality(req.evidence_graph)
        density = _graph_query_interface.calculate_graph_density(req.evidence_graph)
        evidence_metrics = {
            "centrality": centrality,
            "density": density,
            "claim_count": len(req.evidence_graph.claim_nodes),
            "edge_count": len(req.evidence_graph.edges),
        }
    
    return GraphMetricsResponse(
        claim_graph_metrics=claim_metrics,
        evidence_graph_metrics=evidence_metrics,
    )


# ============================================================================
# Calibration API Endpoints
# ============================================================================

class CalibrationStatusResponse(BaseModel):
    """Response model for calibration status."""
    calibration_available: bool
    model_version: str | None = None
    last_trained: str | None = None
    training_samples: int | None = None
    method: str | None = None


class CalibrateConfidenceRequest(BaseModel):
    """Request to calibrate a confidence score."""
    heuristic_confidence: float = Field(ge=0.0, le=1.0, description="Original confidence score")
    batch: list[float] | None = Field(None, description="Optional: batch of confidence scores")


class CalibrateConfidenceResponse(BaseModel):
    """Response with calibrated confidence."""
    heuristic_confidence: float
    calibrated_confidence: float
    calibration_applied: bool
    model_version: str | None = None
    batch_calibrated: list[float] | None = None


@app.get("/calibration/status", response_model=CalibrationStatusResponse)
@limiter.limit("60/minute")
async def get_calibration_status(request: Request) -> CalibrationStatusResponse:
    """Get the current status of confidence calibration model."""
    try:
        from .calibration.model import ConfidenceCalibrator
        from pathlib import Path
        
        # Check for calibration model in standard location
        model_dir = Path(__file__).parent / "calibration" / "models"
        
        if model_dir.exists():
            calibrator = ConfidenceCalibrator(model_dir=model_dir)
            current_version = calibrator.get_current_version()
            
            if current_version and calibrator.is_fitted:
                versions = calibrator.list_versions()
                current = next((v for v in versions if v.version == current_version), None)
                
                return CalibrationStatusResponse(
                    calibration_available=True,
                    model_version=current_version,
                    last_trained=current.created_at.isoformat() if current else None,
                    training_samples=current.config.get("data_size") if current else None,
                    method=calibrator.method,
                )
        
        return CalibrationStatusResponse(
            calibration_available=False,
            model_version=None,
        )
    except Exception:
        return CalibrationStatusResponse(
            calibration_available=False,
            model_version=None,
        )


@app.post("/calibration/calibrate", response_model=CalibrateConfidenceResponse)
@limiter.limit("50/minute")
async def calibrate_confidence(request: Request, req: Annotated[CalibrateConfidenceRequest, Body()]) -> CalibrateConfidenceResponse:
    """Apply calibration to a heuristic confidence score."""
    try:
        from .calibration.model import ConfidenceCalibrator
        from pathlib import Path
        
        # Try to load calibration model
        model_dir = Path(__file__).parent / "calibration" / "models"
        
        if model_dir.exists():
            calibrator = ConfidenceCalibrator(model_dir=model_dir)
            current_version = calibrator.get_current_version()
            
            if current_version and calibrator.is_fitted:
                # Apply calibration
                calibrated = calibrator.predict_proba(req.heuristic_confidence)
                
                batch_result = None
                if req.batch:
                    batch_result = calibrator.predict_proba_batch(req.batch).tolist()
                
                return CalibrateConfidenceResponse(
                    heuristic_confidence=req.heuristic_confidence,
                    calibrated_confidence=calibrated,
                    calibration_applied=True,
                    model_version=current_version,
                    batch_calibrated=batch_result,
                )
        
        # No calibration model available, return unchanged
        batch_result = None
        if req.batch:
            batch_result = req.batch  # Return unchanged
        
        return CalibrateConfidenceResponse(
            heuristic_confidence=req.heuristic_confidence,
            calibrated_confidence=req.heuristic_confidence,
            calibration_applied=False,
            model_version=None,
            batch_calibrated=batch_result,
        )
    except Exception as e:
        logger.error(f"Calibration error: {type(e).__name__}: {str(e)}")
        
        # Return unchanged on error
        batch_result = None
        if req.batch:
            batch_result = req.batch
        
        return CalibrateConfidenceResponse(
            heuristic_confidence=req.heuristic_confidence,
            calibrated_confidence=req.heuristic_confidence,
            calibration_applied=False,
            model_version=None,
            batch_calibrated=batch_result,
        )


def run(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    import uvicorn

    uvicorn.run("hawkins_truth_engine.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Hawkins Truth Engine server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    run(host=args.host, port=args.port, reload=args.reload)
