from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


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
    distribution: list[dict[str, Any]] = Field(default_factory=list)


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
    signals: list[EvidenceItem]
    highlighted_phrases: list[str]


class StatisticalOutput(BaseModel):
    statistical_risk_score: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceItem]


class SourceIntelOutput(BaseModel):
    source_trust_score: float = Field(ge=0.0, le=1.0)
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
    document: Document
    linguistic: LinguisticOutput
    statistical: StatisticalOutput
    source: SourceIntelOutput
    claims: ClaimsOutput
    aggregation: AggregationOutput
    explanation: VerdictExplanation


class AnalyzeRequest(BaseModel):
    input_type: InputType
    content: str
