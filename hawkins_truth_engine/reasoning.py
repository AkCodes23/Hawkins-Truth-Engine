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
    contested_claims: int
    unverifiable_claims: int
    medical_topic: bool
    strong_claim_wo_attr: int
    high_evidence_strength_claims: int
    contradiction_detected: bool


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
        # Instead of a flat 50, calculate a base score from other signals
        # If linguistic risk is low and source trust is high, it's likely fine
        base_risk = min(1.0, config.REASONING_LINGUISTIC_WEIGHT * float(linguistic.linguistic_risk_score) +
                        config.REASONING_STATISTICAL_WEIGHT * float(statistical.statistical_risk_score))
        
        # Source trust gate
        gate = 1.0
        if float(source.source_trust_score) < config.REASONING_LOW_TRUST_THRESHOLD:
            gate = config.REASONING_LOW_TRUST_MULTIPLIER
        elif float(source.source_trust_score) > config.REASONING_HIGH_TRUST_THRESHOLD:
            gate = config.REASONING_HIGH_TRUST_MULTIPLIER
        
        risk = min(1.0, base_risk * gate)
        cred_score = int(round(100 * (1.0 - risk)))
        
        # Determine verdict based on calculated score
        if cred_score >= config.VERDICT_LIKELY_REAL_THRESHOLD:
            verdict = "Likely Real"
        elif cred_score >= config.VERDICT_SUSPICIOUS_THRESHOLD:
            verdict = "Suspicious"
        else:
            verdict = "Likely Fake"

        return AggregationOutput(
            credibility_score=cred_score,
            verdict=verdict,
            world_label="Real World" if verdict == "Likely Real" else "Upside Down",
            confidence=0.15,  # Slightly higher than 0.1 but still very low
            confidence_calibrated=False,
            uncertainty_flags=["no_claims_extracted", "insufficient_data_for_analysis"],
            reasoning_path=[
                ReasoningStep(
                    rule_id="R_NO_CLAIMS",
                    triggered=True,
                    because=[
                        "no_claims_extracted_from_document",
                        f"linguistic_risk={float(linguistic.linguistic_risk_score):.2f}",
                        f"source_trust={float(source.source_trust_score):.2f}"
                    ],
                    contributed={"direction": "neutral_base", "reason": "no_claims_to_verify"},
                    evidence_ids=[],
                )
            ],
        )
    
    # Count claims by support status
    contested_count = sum(1 for c in claims.claim_items if c.support == "contested")
    high_strength_count = sum(
        1 for c in claims.claim_items
        if c.citations and any(
            cit.get("evidence_strength", {}).get("strength_tier") in ["very_strong", "strong"]
            for cit in c.citations
        )
    )
    contradiction_detected = any(
        "potential_contradictions_detected" in c.uncertainty_flags
        or "contradicting_evidence" in " ".join(c.reasons).lower()
        for c in claims.claim_items
    )
    
    sig = Signals(
        linguistic_risk=float(linguistic.linguistic_risk_score),
        statistical_risk=float(statistical.statistical_risk_score),
        source_trust=float(source.source_trust_score),
        supported_claims=int(claims.claims.get("supported", 0)),
        unsupported_claims=int(claims.claims.get("unsupported", 0)),
        contested_claims=contested_count,
        unverifiable_claims=int(claims.claims.get("unverifiable", 0)),
        medical_topic=bool(claims.medical_topic_detected),
        strong_claim_wo_attr=_count_reason(
            claims.claim_items, "strong_claim_without_attribution"
        ),
        high_evidence_strength_claims=high_strength_count,
        contradiction_detected=contradiction_detected,
    )

    uncertainty_flags: list[str] = []
    reasoning: list[ReasoningStep] = []

    # Rule registry (explicit, deterministic)
    # R1: Low-trust + high linguistic + low claim support -> likely fake
    # Also trigger if high linguistic risk alone with no support (for raw text where source trust is neutral)
    r1 = (
        (
            sig.source_trust < config.REASONING_LOW_TRUST_THRESHOLD
            or (sig.source_trust <= 0.55 and sig.linguistic_risk > config.REASONING_HIGH_LINGUISTIC_RISK)
        )
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
    # Also includes contested claims (explicit contradictions)
    ambiguous = (
        (sig.supported_claims >= 1 and sig.unsupported_claims >= 1)
        or sig.contested_claims >= 1
        or (sig.supported_claims >= 1 and sig.unverifiable_claims >= 1 and sig.contested_claims >= 1)
    )
    if ambiguous:
        if sig.contested_claims >= 1:
            uncertainty_flags.append("contested_claims_detected")
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
    # Also trigger if high linguistic risk alone with unsupported claims (for short texts)
    high_multi_risk = (
        (
            (sig.linguistic_risk > config.REASONING_HIGH_LINGUISTIC_RISK
             and sig.statistical_risk > config.REASONING_HIGH_LINGUISTIC_RISK)
            or (sig.linguistic_risk > config.REASONING_HIGH_LINGUISTIC_RISK
                and sig.statistical_risk == 0.0  # Short text case
                and sig.unsupported_claims >= 1)
        )
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
    
    # R6: High linguistic risk with no statistical signals (short text case)
    # When text is too short for statistical analysis but has high linguistic risk
    r6_high_ling_short_text = (
        sig.linguistic_risk > config.REASONING_HIGH_LINGUISTIC_RISK
        and sig.statistical_risk == 0.0
        and sig.supported_claims == 0
        and (sig.unverifiable_claims + sig.unsupported_claims) >= 2
    )
    if r6_high_ling_short_text:
        uncertainty_flags.append("high_linguistic_risk_short_text")
    reasoning.append(
        ReasoningStep(
            rule_id="R_HIGH_LING_SHORT_TEXT",
            triggered=r6_high_ling_short_text,
            because=[
                f"linguistic_risk={sig.linguistic_risk:.2f}",
                f"statistical_risk={sig.statistical_risk:.2f}",
                f"supported_claims={sig.supported_claims}",
                f"unverifiable_claims={sig.unverifiable_claims}",
            ],
            contributed={"direction": "toward_fake" if r6_high_ling_short_text else "none"},
            evidence_ids=(
                _top_item_ids(linguistic.signals, limit=6)
                if r6_high_ling_short_text
                else []
            ),
        )
    )
    
    # R7: Majority claim agreement edge case
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

    # --- Unified Risk Calculation (Weighted by Confidence) ---
    ling_risk = sig.linguistic_risk
    ling_conf = float(linguistic.confidence_score)
    
    stat_risk = sig.statistical_risk
    stat_conf = float(statistical.confidence_score)
    
    src_trust = sig.source_trust
    src_risk = 1.0 - src_trust
    src_conf = float(source.confidence_score)
    
    # Calculate base risk using module confidence
    total_raw_weight = (
        config.REASONING_LINGUISTIC_WEIGHT * ling_conf +
        config.REASONING_STATISTICAL_WEIGHT * stat_conf +
        config.REASONING_SOURCE_WEIGHT * src_conf
    )
    
    if total_raw_weight > 0:
        base_risk = (
            config.REASONING_LINGUISTIC_WEIGHT * ling_conf * ling_risk +
            config.REASONING_STATISTICAL_WEIGHT * stat_conf * stat_risk +
            config.REASONING_SOURCE_WEIGHT * src_conf * src_risk
        ) / total_raw_weight
    else:
        base_risk = 0.5

    risk = base_risk

    # --- Claim-Based Adjustments ---
    cross_provider_count = 0
    reputable_source_count = 0
    questionable_source_count = 0
    high_diversity_count = 0
    
    for claim_item in claims.claim_items:
        if "cross_provider_validation" in claim_item.reasons:
            cross_provider_count += 1
        if "multiple_reputable_sources" in claim_item.quality_flags:
            reputable_source_count += 1
        if "questionable_sources_present" in claim_item.quality_flags:
            questionable_source_count += 1
        if "high_source_diversity" in claim_item.quality_flags:
            high_diversity_count += 1
    
    # Claim support improves score (reduces risk)
    if sig.supported_claims >= 1:
        support_boost = min(0.35, config.REASONING_SUPPORTED_CLAIMS_ADJUSTMENT * sig.supported_claims)
        if sig.high_evidence_strength_claims >= 1:
            support_boost += config.REASONING_REPUTABLE_SOURCE_BOOST
        risk = max(0.0, risk - support_boost)

    # Penalties for unsupported or contested claims
    penalty = 0.0
    if sig.unsupported_claims >= 1:
        penalty += config.REASONING_UNSUPPORTED_CLAIMS_PENALTY * sig.unsupported_claims
    if sig.contested_claims >= 1:
        penalty += config.REASONING_UNSUPPORTED_CLAIMS_PENALTY * 1.5 * sig.contested_claims
    if sig.contradiction_detected:
        penalty += 0.1
    if questionable_source_count >= 1:
        penalty += config.REASONING_QUESTIONABLE_SOURCE_PENALTY
        
    risk = min(1.0, risk + penalty)

    # --- RuleOverrides (Logical Gates) ---
    if r1 or r2 or r6_high_ling_short_text or high_multi_risk:
        risk = max(risk, config.REASONING_MIN_FAKE_RISK)
    elif r3 and sig.unsupported_claims == 0:
        risk = min(risk, config.REASONING_MAX_REAL_RISK)

    # Final Score and Verdict
    credibility_score = int(round(100 * (1.0 - risk)))
    credibility_score = int(max(0, min(100, credibility_score)))

    if credibility_score >= config.VERDICT_LIKELY_REAL_THRESHOLD:
        verdict = "Likely Real"
    elif credibility_score >= config.VERDICT_SUSPICIOUS_THRESHOLD:
        verdict = "Suspicious"
    else:
        verdict = "Likely Fake"

    world_label = "Real World" if verdict == "Likely Real" else "Upside Down"

    # --- Calibrated Confidence Calculation ---
    ling_conf = float(linguistic.confidence_score)
    stat_conf = float(statistical.confidence_score)
    src_conf = float(source.confidence_score)
    
    # 1. Module weighted avg confidence
    avg_module_conf = (ling_conf + stat_conf + src_conf) / 3.0
    
    # 2. Agreement (inverse of variance)
    relevant_risks = [sig.linguistic_risk, 1.0 - sig.source_trust]
    if stat_conf > 0.4: relevant_risks.append(sig.statistical_risk)
    
    agreement = 1.0
    if len(relevant_risks) > 1:
        # Simple stdev-based agreement
        mean_risk = sum(relevant_risks) / len(relevant_risks)
        variance = sum((x - mean_risk)**2 for x in relevant_risks) / len(relevant_risks)
        agreement = 1.0 - (variance ** 0.5) * 2.0 # Scale to 0-1
    
    # 3. Data coverage
    total_claims = len(claims.claim_items)
    reputable_source_count = sum(1 for c in claims.claim_items if "multiple_reputable_sources" in c.quality_flags)
    data_coverage = min(1.0, (total_claims / 5.0) * 0.5 + (reputable_source_count / 3.0) * 0.5)
    
    conf = (avg_module_conf * 0.4 + agreement * 0.3 + data_coverage * 0.3)
    
    # Penalty for conflicts
    if ambiguous or sig.contested_claims >= 1:
        conf *= 0.7
        
    conf = float(max(0.1, min(1.0, conf)))

    return AggregationOutput(
        credibility_score=credibility_score,
        verdict=verdict,  # type: ignore[arg-type]
        world_label=world_label,  # type: ignore[arg-type]
        confidence=conf,
        confidence_calibrated=True,
        uncertainty_flags=uncertainty_flags,
        reasoning_path=reasoning,
    )
