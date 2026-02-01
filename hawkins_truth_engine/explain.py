from __future__ import annotations

from .schemas import VerdictExplanation


def generate_explanation(
    doc, linguistic, statistical, source, claims, aggregation
) -> VerdictExplanation:
    verdict_text = (
        f"World: {aggregation.world_label} | Verdict: {aggregation.verdict} "
        f"({aggregation.confidence * 100:.0f}% confidence)"
    )

    bullets: list[str] = []
    # Pick top evidence across modules (simple severity/weight heuristic)
    items = []
    items.extend(linguistic.signals)
    items.extend(statistical.evidence)
    items.extend(source.source_flags)
    scored = []
    for it in items:
        sev = {"low": 1, "medium": 2, "high": 3}.get(it.severity, 1)
        score = sev * 10 + it.weight * 5 + (it.value or 0) * 5
        scored.append((score, it))

    seen = set()
    for score, it in sorted(scored, key=lambda t: t[0], reverse=True):
        if it.evidence in seen:
            continue
        seen.add(it.evidence)
        bullets.append(f"{it.module}: {it.evidence}")
        if len(bullets) >= 6:
            break

    # Claim summary bullets
    bullets.append(
        f"claims: supported={claims.claims.get('supported', 0)}, unverifiable={claims.claims.get('unverifiable', 0)}"
    )
    if claims.medical_topic_detected:
        bullets.append(
            f"medical: topic cues detected ({', '.join(sorted(set(claims.medical_topic_triggers))[:5])})"
        )

    assumptions = [
        "Online evidence providers (GDELT/RDAP/NCBI) returned representative results.",
        "Heuristic claim extraction approximates atomic claims for this POC.",
    ]
    blind_spots = [
        "No guaranteed real-time fact verification; evidence absence is not proof of falsity.",
        "Non-English content and domain-specific jargon reduce extraction reliability.",
        "PubMed abstracts may not resolve claim specificity (dose/population/outcome).",
    ]
    if aggregation.uncertainty_flags:
        assumptions.append(
            "Uncertainty flags are treated as first-class outputs and may lower confidence."
        )

    highlighted_spans = []
    for it in linguistic.signals:
        for sp in it.pointers.char_spans:
            highlighted_spans.append({"start": sp.start, "end": sp.end, "label": it.id})

    return VerdictExplanation(
        verdict_text=verdict_text,
        evidence_bullets=bullets,
        assumptions=assumptions,
        blind_spots=blind_spots,
        highlighted_spans=highlighted_spans,
    )
