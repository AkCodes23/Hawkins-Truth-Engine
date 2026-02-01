from __future__ import annotations

from dataclasses import dataclass

from . import config
from .schemas import AggregationOutput, ReasoningStep


@dataclass(frozen=True)
class Signals:
    linguistic_risk: float
    statistical_risk: float
    source_trust: float
    supported_claims: int
    unsupported_claims: int
    unverifiable_claims: int
    medical_topic: bool
    strong_claim_wo_attr: int


def _count_reason(claim_items, reason: str) -> int:
    """Count claims with a specific reason.
    
    Args:
        claim_items: List of ClaimItem objects to search
        reason: The reason string to count
        
    Returns:
        Count of claims containing the specified reason
    """
    n = 0
    for c in claim_items:
        if reason in (c.reasons or []):
            n += 1
    return n


def _claim_evidence_ids(
    claims, *, reason: str | None = None, support: str | None = None
) -> list[str]:
    """Get evidence IDs for claims matching optional filters.
    
    Args:
        claims: ClaimsOutput object
        reason: Optional filter by reason string
        support: Optional filter by support status (supported/unsupported/etc.)
        
    Returns:
        List of evidence IDs (formatted as 'claim:id') matching criteria
    """
    out: list[str] = []
    for c in claims.claim_items:
        if reason is not None and reason not in (c.reasons or []):
            continue
        if support is not None and c.support != support:
            continue
        out.append(f"claim:{c.id}")
    return out


def _top_item_ids(items, *, limit: int) -> list[str]:
    """Get top N evidence items ranked deterministically.
    
    Ranking prioritizes: severity (high > medium > low), then weight, then ID.
    This ensures reproducible, explainable ordering of evidence.
    
    Args:
        items: List of evidence items to rank
        limit: Maximum number of items to return
        
    Returns:
        List of up to 'limit' evidence IDs, ranked by criteria
    """
    # Deterministic: severity then weight then id.
    sev_rank = {"low": 1, "medium": 2, "high": 3}
    ranked = sorted(
        items,
        key=lambda it: (
            sev_rank.get(getattr(it, "severity", "low"), 1),
            float(getattr(it, "weight", 0.0)),
            str(getattr(it, "id", "")),
        ),
        reverse=True,
    )
    return [str(it.id) for it in ranked[: max(0, limit)]]


def aggregate(linguistic, statistical, source, claims) -> AggregationOutput:
    """Aggregate multiple analysis signals into a unified credibility assessment.
    
    Implements a rule-based reasoning system that:
    - Applies 6 deterministic reasoning rules (R1-R6)
    - Combines linguistic, statistical, and source intelligence signals
    - Factors in claim support evidence
    - Produces explainable reasoning trace
    - Handles edge cases (no claims, ambiguous cases, multi-signal risks)
    
    Args:
        linguistic: LinguisticOutput with linguistic risk signals
        statistical: StatisticalOutput with statistical risk signals
        source: SourceIntelOutput with source trust score
        claims: ClaimsOutput with extracted claims and support statuses
        
    Returns:
        AggregationOutput with credibility score (0-100), verdict, confidence,
        and detailed reasoning path for explainability.
    """
    # Handle empty claims edge case - prevents crashes when no claims are extracted
    if not claims.claim_items:
        # Return a neutral result with low confidence when there's insufficient data
        return AggregationOutput(
            credibility_score=50,  # Neutral score
            verdict="Suspicious",
            world_label="Upside Down",
            confidence=0.1,  # Very low confidence due to insufficient data
            confidence_calibrated=False,
            uncertainty_flags=["no_claims_extracted", "insufficient_data_for_analysis"],
            reasoning_path=[
                ReasoningStep(
                    rule_id="R_NO_CLAIMS",
                    triggered=True,
                    because=["no_claims_extracted_from_document"],
                    contributed={"direction": "neutral", "reason": "insufficient_data"},
                    evidence_ids=[],
                )
            ],
        )
    
    sig = Signals(
        linguistic_risk=float(linguistic.linguistic_risk_score),
        statistical_risk=float(statistical.statistical_risk_score),
        source_trust=float(source.source_trust_score),
        supported_claims=int(claims.claims.get("supported", 0)),
        unsupported_claims=int(claims.claims.get("unsupported", 0)),
        unverifiable_claims=int(claims.claims.get("unverifiable", 0)),
        medical_topic=bool(claims.medical_topic_detected),
        strong_claim_wo_attr=_count_reason(
            claims.claim_items, "strong_claim_without_attribution"
        ),
    )

    uncertainty_flags: list[str] = []
    reasoning: list[ReasoningStep] = []

    # Rule registry (explicit, deterministic)
    # R1: Low-trust + high linguistic + low claim support -> likely fake
    r1 = (
        sig.source_trust < config.REASONING_LOW_TRUST_THRESHOLD
        and sig.linguistic_risk > config.REASONING_HIGH_LINGUISTIC_RISK
        and sig.supported_claims == 0
        and (sig.unverifiable_claims + sig.unsupported_claims) >= 2
    )
    reasoning.append(
        ReasoningStep(
            rule_id="R_LOW_SOURCE_HIGH_LING_LOW_SUPPORT",
            triggered=r1,
            because=[
                f"source_trust={sig.source_trust:.2f}",
                f"linguistic_risk={sig.linguistic_risk:.2f}",
                f"supported_claims={sig.supported_claims}",
                f"unsupported_claims={sig.unsupported_claims}",
                f"unverifiable_claims={sig.unverifiable_claims}",
            ],
            contributed={"direction": "toward_fake" if r1 else "none"},
            evidence_ids=(
                (
                    _top_item_ids(source.source_flags, limit=6)
                    + _top_item_ids(linguistic.signals, limit=6)
                    + _claim_evidence_ids(claims, support="unsupported")[:4]
                    + _claim_evidence_ids(claims, support="unverifiable")[:4]
                )
                if r1
                else []
            ),
        )
    )

    # R2: Strong medical claims without attribution and no support -> high harm potential
    r2 = (
        sig.medical_topic
        and sig.strong_claim_wo_attr >= 1
        and sig.supported_claims == 0
    )
    if r2:
        uncertainty_flags.append("high_harm_potential_medical")
    reasoning.append(
        ReasoningStep(
            rule_id="R_MED_STRONG_CLAIM_NO_SUPPORT",
            triggered=r2,
            because=[
                f"medical_topic={sig.medical_topic}",
                f"strong_claim_without_attribution={sig.strong_claim_wo_attr}",
                f"supported_claims={sig.supported_claims}",
            ],
            contributed={"direction": "toward_fake" if r2 else "none"},
            evidence_ids=(
                _claim_evidence_ids(claims, reason="strong_claim_without_attribution")
                if r2
                else []
            ),
        )
    )

    # R3: High source trust dampens risk (but doesn't erase it)
    r3 = sig.source_trust > config.REASONING_HIGH_TRUST_THRESHOLD and (
        sig.linguistic_risk < config.REASONING_LOW_RISK_THRESHOLD 
        and sig.statistical_risk < config.REASONING_LOW_RISK_THRESHOLD
    )
    reasoning.append(
        ReasoningStep(
            rule_id="R_HIGH_SOURCE_LOW_RISK",
            triggered=r3,
            because=[
                f"source_trust={sig.source_trust:.2f}",
                f"linguistic_risk={sig.linguistic_risk:.2f}",
                f"statistical_risk={sig.statistical_risk:.2f}",
            ],
            contributed={"direction": "toward_real" if r3 else "none"},
            evidence_ids=(
                (
                    _top_item_ids(source.source_flags, limit=6)
                    + _top_item_ids(statistical.evidence, limit=4)
                )
                if r3
                else []
            ),
        )
    )
    
    # R4: Ambiguous case - mixed signals (contentious claims with both support and opposition)
    ambiguous = (
        sig.supported_claims >= 1 
        and sig.unsupported_claims >= 1 
        and sig.unverifiable_claims >= 1
    )
    if ambiguous:
        uncertainty_flags.append("mixed_claim_support")
    reasoning.append(
        ReasoningStep(
            rule_id="R_AMBIGUOUS_MIXED_SIGNALS",
            triggered=ambiguous,
            because=[
                f"supported_claims={sig.supported_claims}",
                f"unsupported_claims={sig.unsupported_claims}",
                f"unverifiable_claims={sig.unverifiable_claims}",
            ],
            contributed={"direction": "reduces_confidence" if ambiguous else "none"},
            evidence_ids=(
                (
                    _claim_evidence_ids(claims, support="supported")[:2]
                    + _claim_evidence_ids(claims, support="unsupported")[:2]
                    + _claim_evidence_ids(claims, support="unverifiable")[:2]
                )
                if ambiguous
                else []
            ),
        )
    )
    
    # R5: High risk signals across multiple dimensions
    high_multi_risk = (
        sig.linguistic_risk > config.REASONING_HIGH_LINGUISTIC_RISK
        and sig.statistical_risk > config.REASONING_HIGH_LINGUISTIC_RISK
        and sig.unsupported_claims >= 1
    )
    if high_multi_risk:
        uncertainty_flags.append("high_multi_signal_risk")
    reasoning.append(
        ReasoningStep(
            rule_id="R_HIGH_MULTI_RISK",
            triggered=high_multi_risk,
            because=[
                f"linguistic_risk={sig.linguistic_risk:.2f}",
                f"statistical_risk={sig.statistical_risk:.2f}",
                f"unsupported_claims={sig.unsupported_claims}",
            ],
            contributed={"direction": "toward_fake" if high_multi_risk else "none"},
            evidence_ids=(
                (
                    _top_item_ids(linguistic.signals, limit=4)
                    + _top_item_ids(statistical.evidence, limit=4)
                    + _claim_evidence_ids(claims, support="unsupported")[:2]
                )
                if high_multi_risk
                else []
            ),
        )
    )
    
    # R6: Majority claim agreement edge case
    total_claims = sig.supported_claims + sig.unsupported_claims + sig.unverifiable_claims
    claim_agreement = (
        total_claims > 0
        and max(sig.supported_claims, sig.unsupported_claims, sig.unverifiable_claims) / total_claims >= config.REASONING_CLAIM_AGREEMENT_THRESHOLD
    )
    reasoning.append(
        ReasoningStep(
            rule_id="R_CLAIM_AGREEMENT",
            triggered=claim_agreement,
            because=[
                f"total_claims={total_claims}",
                f"supported_claims={sig.supported_claims}",
                f"unsupported_claims={sig.unsupported_claims}",
                f"unverifiable_claims={sig.unverifiable_claims}",
            ],
            contributed={"direction": "increases_confidence" if claim_agreement else "none"},
            evidence_ids=[],
        )
    )

    # Compute interpretable ledger (not blind averaging):
    # - risk components are capped
    # - source trust gates the overall risk
    base_risk = min(1.0, 
        config.REASONING_LINGUISTIC_WEIGHT * sig.linguistic_risk + 
        config.REASONING_STATISTICAL_WEIGHT * sig.statistical_risk
    )
    # Source trust gate: low trust amplifies risk, high trust reduces modestly
    gate = 1.0
    if sig.source_trust < config.REASONING_LOW_TRUST_THRESHOLD:
        gate = config.REASONING_LOW_TRUST_MULTIPLIER
    elif sig.source_trust > config.REASONING_HIGH_TRUST_THRESHOLD:
        gate = config.REASONING_HIGH_TRUST_MULTIPLIER
    risk = min(1.0, base_risk * gate)

    # Claim support adjustment
    if sig.supported_claims >= 2:
        risk = max(0.0, risk - config.REASONING_SUPPORTED_CLAIMS_ADJUSTMENT)
    if sig.unverifiable_claims >= 3:
        risk = min(1.0, risk + config.REASONING_UNVERIFIABLE_CLAIMS_PENALTY)
    if sig.unsupported_claims >= 2:
        risk = min(1.0, risk + config.REASONING_UNSUPPORTED_CLAIMS_PENALTY)

    # Apply rule overrides
    if r1 or r2:
        risk = min(1.0, max(risk, config.REASONING_MIN_FAKE_RISK))
    if r3:
        risk = min(risk, config.REASONING_MAX_REAL_RISK)
    if high_multi_risk:
        risk = min(1.0, max(risk, config.REASONING_MULTIRISK_MIN_RISK))
    if ambiguous:
        risk = min(1.0, max(risk, config.REASONING_AMBIGUOUS_MIN_RISK))

    credibility_score = int(round(100 * (1.0 - risk)))
    if credibility_score >= config.VERDICT_LIKELY_REAL_THRESHOLD:
        verdict = "Likely Real"
    elif credibility_score >= config.VERDICT_SUSPICIOUS_THRESHOLD:
        verdict = "Suspicious"
    else:
        verdict = "Likely Fake"

    world_label = "Real World" if verdict == "Likely Real" else "Upside Down"

    # Confidence heuristic (POC; not calibrated)
    agreement = 1.0 - abs(sig.linguistic_risk - sig.statistical_risk)
    coverage = config.REASONING_CONFIDENCE_BASE_COVERAGE
    if sig.unverifiable_claims == 0:
        coverage += 0.2
    if sig.supported_claims >= 1:
        coverage += 0.1
    if claim_agreement:
        coverage += 0.15
    
    conf = max(
        config.REASONING_CONFIDENCE_MIN,
        min(
            config.REASONING_CONFIDENCE_MAX,
            config.REASONING_CONFIDENCE_BASE_SCORE + 0.35 * agreement + 0.30 * (coverage - config.REASONING_CONFIDENCE_BASE_COVERAGE)
        )
    )
    
    # Reduce confidence for uncertainty conditions
    if any("unavailable" in (str(f).lower()) for f in uncertainty_flags):
        conf = min(conf, 0.75)
    if ambiguous:
        conf = min(conf, 0.65)
    if "mixed_claim_support" in uncertainty_flags:
        conf = max(0.3, min(conf, 0.60))
    if high_multi_risk:
        conf = min(conf, 0.75)
    
    # Explicit final clamping to ensure valid range [0.0, 1.0]
    # This handles any edge cases from the calculations above
    conf = float(max(0.0, min(1.0, conf)))
    
    # Also validate credibility_score is in valid range [0, 100]
    credibility_score = int(max(0, min(100, credibility_score)))

    return AggregationOutput(
        credibility_score=credibility_score,
        verdict=verdict,  # type: ignore[arg-type]
        world_label=world_label,  # type: ignore[arg-type]
        confidence=conf,
        confidence_calibrated=False,
        uncertainty_flags=uncertainty_flags,
        reasoning_path=reasoning,
    )
