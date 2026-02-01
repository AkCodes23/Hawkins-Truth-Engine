from __future__ import annotations

import re

from ..schemas import EvidenceItem, LinguisticOutput, Pointer
from ..utils import find_spans


_CLICKBAIT_PHRASES = [
    "you won't believe",
    "shocking",
    "what happened next",
    "doctors hate",
    "miracle",
    "secret",
    "exposed",
]

_CONSPIRACY_PHRASES = [
    "they don't want you to know",
    "mainstream media",
    "cover-up",
    "deep state",
    "big pharma",
]

_URGENCY_WORDS = {"urgent", "now", "immediately", "warning", "alert", "breaking"}
_CERTAINTY = {"proves", "definitely", "guaranteed", "always", "never", "undeniable"}
_HEDGES = {"may", "might", "could", "possibly", "suggests", "appears"}


def analyze_linguistic(doc) -> LinguisticOutput:
    text = doc.display_text
    signals: list[EvidenceItem] = []
    highlights: list[str] = []

    # Clickbait punctuation / caps
    exclam = text.count("!")
    qmarks = text.count("?")
    caps_tokens = sum(1 for t in doc.tokens if t.text.isupper() and len(t.text) >= 3)
    cap_ratio = (caps_tokens / max(1, len(doc.tokens)))
    punct_score = min(1.0, (exclam + qmarks) / 10.0)
    caps_score = min(1.0, cap_ratio * 8.0)
    if punct_score > 0.25:
        signals.append(
            EvidenceItem(
                id="clickbait_punct",
                module="linguistic",
                weight=0.10,
                value=punct_score,
                severity="medium" if punct_score < 0.6 else "high",
                evidence=f"High punctuation intensity (!/? count={exclam+qmarks}).",
                pointers=Pointer(char_spans=[]),
                provenance={"exclam": exclam, "qmarks": qmarks},
            )
        )
    if caps_score > 0.25:
        signals.append(
            EvidenceItem(
                id="clickbait_caps",
                module="linguistic",
                weight=0.10,
                value=caps_score,
                severity="medium" if caps_score < 0.6 else "high",
                evidence=f"Unusually high ALL-CAPS token ratio ({cap_ratio:.3f}).",
                pointers=Pointer(char_spans=[]),
                provenance={"caps_tokens": caps_tokens, "total_tokens": len(doc.tokens)},
            )
        )

    # Phrase-based clickbait and conspiracy framing
    lower = text.lower()
    for phrase in _CLICKBAIT_PHRASES:
        if phrase in lower:
            spans = find_spans(text, phrase)
            signals.append(
                EvidenceItem(
                    id=f"clickbait_phrase::{phrase}",
                    module="linguistic",
                    weight=0.12,
                    value=0.8,
                    severity="high",
                    evidence=f"Clickbait phrase detected: '{phrase}'.",
                    pointers=Pointer(char_spans=spans),
                    provenance={},
                )
            )
            highlights.append(phrase)
    for phrase in _CONSPIRACY_PHRASES:
        if phrase in lower:
            spans = find_spans(text, phrase)
            signals.append(
                EvidenceItem(
                    id=f"conspiracy_phrase::{phrase}",
                    module="linguistic",
                    weight=0.15,
                    value=0.9,
                    severity="high",
                    evidence=f"Conspiracy framing phrase detected: '{phrase}'.",
                    pointers=Pointer(char_spans=spans),
                    provenance={},
                )
            )
            highlights.append(phrase)

    # Urgency / emotion cues (lexicon)
    urgency_hits = [w for w in _URGENCY_WORDS if re.search(rf"\\b{re.escape(w)}\\b", lower)]
    if urgency_hits:
        signals.append(
            EvidenceItem(
                id="urgency_lexicon",
                module="linguistic",
                weight=0.08,
                value=min(1.0, len(urgency_hits) / 4.0),
                severity="medium",
                evidence=f"Urgency cues present: {', '.join(sorted(set(urgency_hits)))}.",
                pointers=Pointer(char_spans=[]),
                provenance={"hits": urgency_hits},
            )
        )

    # Hedging vs certainty imbalance
    cert = sum(1 for w in _CERTAINTY if re.search(rf"\\b{re.escape(w)}\\b", lower))
    hedge = sum(1 for w in _HEDGES if re.search(rf"\\b{re.escape(w)}\\b", lower))
    imbalance = 0.0
    if cert + hedge > 0:
        imbalance = max(0.0, (cert - hedge) / max(1, cert + hedge))
    if imbalance > 0.35 and cert >= 2:
        signals.append(
            EvidenceItem(
                id="certainty_imbalance",
                module="linguistic",
                weight=0.15,
                value=min(1.0, imbalance),
                severity="high" if imbalance > 0.6 else "medium",
                evidence=f"High certainty language without comparable hedging (certainty={cert}, hedges={hedge}).",
                pointers=Pointer(char_spans=[]),
                provenance={"certainty": cert, "hedges": hedge},
            )
        )

    # Anonymous authority cues
    anon_markers = ["experts say", "scientists say", "sources say", "researchers say"]
    anon_hits = [p for p in anon_markers if p in lower]
    if anon_hits and not doc.entities:
        signals.append(
            EvidenceItem(
                id="anonymous_authority",
                module="linguistic",
                weight=0.12,
                value=0.75,
                severity="medium",
                evidence=f"Anonymous authority cues without named entities: {', '.join(anon_hits)}.",
                pointers=Pointer(char_spans=[]),
                provenance={},
            )
        )

    # Risk score: bounded sum of weighted values (NOT final credibility; only linguistic risk).
    # This is allowed because it is within-module and we expose every contributing signal.
    risk = 0.0
    for s in signals:
        risk += s.weight * (s.value if s.value is not None else 0.5)
    risk = max(0.0, min(1.0, risk))

    return LinguisticOutput(
        linguistic_risk_score=risk,
        signals=signals,
        highlighted_phrases=highlights,
    )
