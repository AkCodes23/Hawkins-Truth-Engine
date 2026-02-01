from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, validator


InputType = Literal["raw_text", "url", "social_post"]
Verdict = Literal["Likely Real", "Suspicious", "Likely Fake"]
WorldLabel = Literal["Real World", "Upside Down"]


class CharSpan(BaseModel):
    start: int
    end: int


class Pointer(BaseModel):
    char_spans: list[CharSpan] = Field(default_factory=list)
    sentence_ids: list[int] = Field(default_factory=list)
    entity_ids: list[int] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    id: str
    module: str
    weight: float = Field(ge=0.0, le=1.0)
    value: float | None = Field(default=None, ge=0.0, le=1.0)
    severity: Literal["low", "medium", "high"]
    evidence: str
    pointers: Pointer = Field(default_factory=Pointer)
    provenance: dict[str, Any] = Field(default_factory=dict)


class LanguageInfo(BaseModel):
    top: str
    distribution: list[dict[str, str | float]] = Field(default_factory=list)
    detection_failed: bool = Field(default=False, description="Whether language detection failed")
    failure_reason: str | None = Field(default=None, description="Reason for detection failure if applicable")


class Sentence(BaseModel):
    id: int
    text: str
    char_span: CharSpan


class Token(BaseModel):
    text: str
    lemma: str | None = None
    char_span: CharSpan


class Entity(BaseModel):
    id: int
    text: str
    type: str
    sentence_id: int
    char_span: CharSpan
    normalized: str | None = None


class Attribution(BaseModel):
    speaker_entity_id: int | None = None
    verb: str
    quote_span: CharSpan
    sentence_id: int


class Document(BaseModel):
    input_type: InputType
    raw_input: str
    url: str | None = None
    domain: str | None = None
    retrieved_at: datetime | None = None
    title: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    display_text: str
    language: LanguageInfo
    sentences: list[Sentence]
    tokens: list[Token]
    entities: list[Entity]
    attributions: list[Attribution]
    preprocessing_flags: list[str] = Field(default_factory=list)
    preprocessing_provenance: dict[str, Any] = Field(default_factory=dict)


class LinguisticOutput(BaseModel):
    linguistic_risk_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    signals: list[EvidenceItem]
    highlighted_phrases: list[str]


class StatisticalOutput(BaseModel):
    statistical_risk_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: list[EvidenceItem]


class SourceIntelOutput(BaseModel):
    source_trust_score: float = Field(ge=0.0, le=1.0)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    source_flags: list[EvidenceItem]


class ClaimItem(BaseModel):
    id: str
    text: str
    type: Literal["factual", "speculative", "predictive", "opinion_presented_as_fact"]
    support: Literal["supported", "unsupported", "contested", "unverifiable"]
    reasons: list[str] = Field(default_factory=list)
    pointers: Pointer = Field(default_factory=Pointer)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    query_trace: list[dict[str, Any]] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)


class ClaimsOutput(BaseModel):
    claims: dict[str, int]
    claim_items: list[ClaimItem]
    medical_topic_detected: bool = False
    medical_topic_triggers: list[str] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)


class ReasoningStep(BaseModel):
    rule_id: str
    triggered: bool
    because: list[str] = Field(default_factory=list)
    contributed: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)


class AggregationOutput(BaseModel):
    credibility_score: int = Field(ge=0, le=100)
    verdict: Verdict
    world_label: WorldLabel
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_calibrated: bool = False
    uncertainty_flags: list[str] = Field(default_factory=list)
    reasoning_path: list[ReasoningStep] = Field(default_factory=list)


class VerdictExplanation(BaseModel):
    verdict_text: str
    evidence_bullets: list[str]
    assumptions: list[str]
    blind_spots: list[str]
    highlighted_spans: list[dict[str, Any]] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    request_id: str = Field(..., description="Unique identifier for this analysis request")
    timestamp: datetime = Field(default_factory=datetime.now, description="When the analysis was performed")
    version: str = Field(default="1.0", description="API version")
    document: Document
    linguistic: LinguisticOutput
    statistical: StatisticalOutput
    source: SourceIntelOutput
    claims: ClaimsOutput
    aggregation: AggregationOutput
    explanation: VerdictExplanation
    # Optional graph data (included when include_graphs=True)
    claim_graph: ClaimGraph | None = None
    evidence_graph: EvidenceGraph | None = None


# Graph Data Schemas


class GraphNode(BaseModel):
    """Node in a graph representing a source, claim, or entity."""

    id: str = Field(
        ...,
        description="Unique identifier (e.g., 'claim:C1', 'source:S1', 'entity:E1')",
    )
    type: Literal["source", "claim", "entity"] = Field(
        ..., description="Type of the node"
    )
    text: str = Field(..., description="Display text for the node")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Type-specific metadata"
    )
    confidence: float | None = Field(
        None, ge=0.0, le=1.0, description="Optional confidence score"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp for provenance"
    )
    updated_at: datetime = Field(
        default_factory=datetime.now, description="Last modification timestamp"
    )

    @validator("id")
    def validate_id_format(cls, v):
        """Validate that ID follows the expected format."""
        if not v:
            raise ValueError("ID cannot be empty")

        # If a namespace prefix is present, ensure the identifier part exists.
        # Do not enforce a fixed prefix set: graph/query interfaces may mint custom IDs.
        if ":" in v:
            _prefix, identifier = v.split(":", 1)
            if not identifier:
                raise ValueError("ID identifier part cannot be empty")

        return v

    @validator("updated_at", always=True)
    def set_updated_at(cls, v, values):
        """Ensure updated_at is set to current time on updates."""
        return datetime.now()

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class GraphEdge(BaseModel):
    """Edge in a graph representing a relationship between nodes."""

    id: str = Field(..., description="Unique edge identifier")
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    relationship_type: Literal[
        "MENTIONS",
        "ATTRIBUTED_TO",
        "FROM_SOURCE",
        "SUPPORTS",
        "CONTRADICTS",
        "RELATES_TO",
    ] = Field(..., description="Type of relationship")
    weight: float = Field(ge=0.0, le=1.0, description="Relationship strength")
    provenance: dict[str, Any] = Field(
        default_factory=dict, description="Evidence for this relationship"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Timestamp for provenance"
    )

    @validator("id")
    def validate_edge_id(cls, v):
        """Validate that edge ID is not empty."""
        if not v:
            raise ValueError("Edge ID cannot be empty")
        return v

    @validator("source_id", "target_id")
    def validate_node_ids(cls, v):
        """Validate that node IDs are not empty."""
        if not v:
            raise ValueError("Node IDs cannot be empty")
        return v

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class ClaimGraph(BaseModel):
    """Graph structure representing relationships between sources, claims, and entities."""

    nodes: dict[str, GraphNode] = Field(
        default_factory=dict, description="Node ID → Node mapping"
    )
    edges: dict[str, GraphEdge] = Field(
        default_factory=dict, description="Edge ID → Edge mapping"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Graph-level metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Graph creation timestamp"
    )

    @validator("nodes")
    def validate_node_consistency(cls, v):
        """Validate that node IDs in the dictionary match the node's id field."""
        for node_id, node in v.items():
            if node.id != node_id:
                raise ValueError(
                    f"Node ID mismatch: dictionary key '{node_id}' != node.id '{node.id}'"
                )
        return v

    @validator("edges")
    def validate_edge_consistency(cls, v, values):
        """Validate that edge IDs match and referenced nodes exist."""
        nodes = values.get("nodes", {})

        for edge_id, edge in v.items():
            if edge.id != edge_id:
                raise ValueError(
                    f"Edge ID mismatch: dictionary key '{edge_id}' != edge.id '{edge.id}'"
                )

            # Validate that source and target nodes exist
            if edge.source_id not in nodes:
                raise ValueError(
                    f"Edge '{edge_id}' references non-existent source node '{edge.source_id}'"
                )
            if edge.target_id not in nodes:
                raise ValueError(
                    f"Edge '{edge_id}' references non-existent target node '{edge.target_id}'"
                )

        return v

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class EvidenceGraph(BaseModel):
    """Graph structure representing evidential relationships between claims."""

    claim_nodes: dict[str, str] = Field(
        default_factory=dict, description="Claim ID → Node ID mapping"
    )
    edges: dict[str, GraphEdge] = Field(
        default_factory=dict, description="Edge ID → Edge mapping"
    )
    similarity_threshold: float = Field(
        0.5, ge=0.0, le=1.0, description="Threshold for RELATES_TO edges"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Graph-level metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.now, description="Graph creation timestamp"
    )

    @validator("edges")
    def validate_evidence_edges(cls, v):
        """Validate that edges have appropriate relationship types for evidence graph."""
        valid_types = {"SUPPORTS", "CONTRADICTS", "RELATES_TO"}

        for edge_id, edge in v.items():
            if edge.relationship_type not in valid_types:
                raise ValueError(
                    f"Edge '{edge_id}' has invalid relationship type '{edge.relationship_type}' "
                    f"for evidence graph. Must be one of: {valid_types}"
                )

        return v

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class AnalyzeRequest(BaseModel):
    input_type: InputType
    content: str
    include_graphs: bool = Field(
        default=False, description="Include graph data in response"
    )
