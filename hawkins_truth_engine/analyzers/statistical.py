from __future__ import annotations

from collections import Counter
from math import log

import numpy as np

from ..schemas import EvidenceItem, StatisticalOutput


def _lexical_diversity(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / max(1, len(tokens))


def _repetition_ratio(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    c = Counter(tokens)
    top = c.most_common(5)
    return sum(v for _, v in top) / max(1, len(tokens))


def analyze_statistical(doc) -> StatisticalOutput:
    toks = [t.text.lower() for t in doc.tokens if t.text]
    evidence: list[EvidenceItem] = []

    diversity = _lexical_diversity(toks)
    repetition = _repetition_ratio(toks)

    if diversity < 0.22 and len(toks) > 200:
        evidence.append(
            EvidenceItem(
                id="low_lexical_diversity",
                module="statistical",
                weight=0.18,
                value=min(1.0, (0.22 - diversity) / 0.22),
                severity="medium",
                evidence=f"Low lexical diversity ({diversity:.3f}) for length={len(toks)}.",
                provenance={"diversity": diversity, "tokens": len(toks)},
            )
        )

    if repetition > 0.12 and len(toks) > 120:
        evidence.append(
            EvidenceItem(
                id="high_repetition",
                module="statistical",
                weight=0.22,
                value=min(1.0, (repetition - 0.12) / 0.20),
                severity="medium" if repetition < 0.22 else "high",
                evidence=f"High repetition ratio among top tokens ({repetition:.3f}).",
                provenance={"repetition": repetition},
            )
        )

    # Simple "burstiness" proxy: many short sentences with similar length.
    sent_lens = np.array([len(s.text.split()) for s in doc.sentences], dtype=float)
    if len(sent_lens) >= 8:
        cv = float(sent_lens.std() / max(1.0, sent_lens.mean()))
        if cv < 0.35:
            evidence.append(
                EvidenceItem(
                    id="uniform_sentence_length",
                    module="statistical",
                    weight=0.10,
                    value=min(1.0, (0.35 - cv) / 0.35),
                    severity="low",
                    evidence=f"Unusually uniform sentence lengths (CV={cv:.3f}).",
                    provenance={"cv": cv},
                )
            )

    # Word frequency distribution irregularity proxy: low entropy.
    if len(toks) >= 200:
        c = Counter(toks)
        total = sum(c.values())
        probs = np.array([v / total for v in c.values()], dtype=float)
        entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        # normalize by log2(V)
        v = len(c)
        norm = entropy / max(1e-6, log(v, 2))
        if norm < 0.72:
            evidence.append(
                EvidenceItem(
                    id="low_token_entropy",
                    module="statistical",
                    weight=0.15,
                    value=min(1.0, (0.72 - norm) / 0.72),
                    severity="medium",
                    evidence=f"Low token distribution entropy (normalized={norm:.3f}).",
                    provenance={"entropy_norm": norm, "vocab": v},
                )
            )

    risk = 0.0
    for e in evidence:
        risk += e.weight * (e.value if e.value is not None else 0.5)
    risk = max(0.0, min(1.0, risk))
    return StatisticalOutput(statistical_risk_score=risk, evidence=evidence)
