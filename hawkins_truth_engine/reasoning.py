from __future__ import annotations

from dataclasses import dataclass

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
    n = 0
    for c in claim_items:
        if reason in (c.reasons or []):
            n += 1
    return n


def _claim_evidence_ids(
    claims, *, reason: str | None = None, support: str | None = None
) -> list[str]:
    out: list[str] = []
    for c in claims.claim_items:
        if reason is not None and reason not in (c.reasons or []):
            continue
        if support is not None and c.support != support:
            continue
        out.append(f"claim:{c.id}")
    return out


def _top_item_ids(items, *, limit: int) -> list[str]:
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
        sig.source_trust < 0.35
        and sig.linguistic_risk > 0.65
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
    r3 = sig.source_trust > 0.75 and (
        sig.linguistic_risk < 0.45 and sig.statistical_risk < 0.45
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

    # Compute interpretable ledger (not blind averaging):
    # - risk components are capped
    # - source trust gates the overall risk
    base_risk = min(1.0, 0.55 * sig.linguistic_risk + 0.45 * sig.statistical_risk)
    # Source trust gate: low trust amplifies risk, high trust reduces modestly
    gate = 1.0
    if sig.source_trust < 0.35:
        gate = 1.25
    elif sig.source_trust > 0.75:
        gate = 0.85
    risk = min(1.0, base_risk * gate)

    # Claim support adjustment
    if sig.supported_claims >= 2:
        risk = max(0.0, risk - 0.20)
    if sig.unverifiable_claims >= 3:
        risk = min(1.0, risk + 0.10)
    if sig.unsupported_claims >= 2:
        risk = min(1.0, risk + 0.15)

    # Apply rule overrides
    if r1 or r2:
        risk = min(1.0, max(risk, 0.80))
    if r3:
        risk = min(risk, 0.35)

    credibility_score = int(round(100 * (1.0 - risk)))
    if credibility_score >= 70:
        verdict = "Likely Real"
    elif credibility_score >= 40:
        verdict = "Suspicious"
    else:
        verdict = "Likely Fake"

    world_label = "Real World" if verdict == "Likely Real" else "Upside Down"

    # Confidence heuristic (POC; not calibrated)
    agreement = 1.0 - abs(sig.linguistic_risk - sig.statistical_risk)
    coverage = 0.6
    if sig.unverifiable_claims == 0:
        coverage += 0.2
    if sig.supported_claims >= 1:
        coverage += 0.1
    conf = max(0.05, min(0.95, 0.35 + 0.35 * agreement + 0.30 * (coverage - 0.6)))
    if any("unavailable" in (str(f).lower()) for f in uncertainty_flags):
        conf = min(conf, 0.75)

    return AggregationOutput(
        credibility_score=credibility_score,
        verdict=verdict,  # type: ignore[arg-type]
        world_label=world_label,  # type: ignore[arg-type]
        confidence=conf,
        confidence_calibrated=False,
        uncertainty_flags=uncertainty_flags,
        reasoning_path=reasoning,
    )
