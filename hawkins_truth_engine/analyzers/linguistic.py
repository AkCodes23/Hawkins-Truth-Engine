from __future__ import annotations

import re
from typing import Callable

from .. import config
from ..schemas import EvidenceItem, LinguisticOutput, Pointer
from ..utils import find_spans


# Expanded clickbait phrases with context requirements
# Format: (phrase, weight, requires_context_check)
_CLICKBAIT_PHRASES = [
    # High-confidence clickbait (no context needed)
    ("you won't believe", 0.14, False),
    ("what happened next will", 0.14, False),
    ("this one weird trick", 0.15, False),
    ("share before it's deleted", 0.15, False),
    ("they don't want you to see", 0.14, False),
    ("is breaking the internet", 0.12, False),
    ("will blow your mind", 0.13, False),
    ("the truth about", 0.08, True),  # Context-dependent
    ("what they're not telling you", 0.14, False),
    ("you need to know this", 0.10, True),
    ("gone viral", 0.08, True),
    ("spread this everywhere", 0.13, False),
    ("wake up people", 0.12, False),
    ("exposed", 0.08, True),  # Context-dependent
    ("bombshell", 0.10, True),
    ("jaw-dropping", 0.09, False),
    ("unbelievable", 0.07, True),
    # Medium-confidence clickbait
    ("shocking", 0.08, True),  # Context-dependent ("shocking pink" vs "shocking news")
    ("miracle", 0.09, True),   # Context-dependent ("miracle cure" vs "economic miracle")
    ("secret", 0.06, True),    # Context-dependent
    ("doctors hate", 0.14, True),  # Check if followed by "this" or "him/her"
    ("scientists baffled", 0.12, False),
    ("insane", 0.06, True),
    ("incredible", 0.05, True),
    ("mind-blowing", 0.10, False),
]

# Expanded conspiracy phrases
_CONSPIRACY_PHRASES = [
    ("they don't want you to know", 0.16, False),
    ("mainstream media lies", 0.15, False),
    ("mainstream media won't tell you", 0.15, False),
    ("cover-up", 0.12, True),  # Context-dependent
    ("deep state", 0.14, False),
    ("big pharma", 0.11, True),  # Context-dependent
    ("new world order", 0.14, False),
    ("globalist", 0.10, True),
    ("sheeple", 0.15, False),
    ("wake up", 0.08, True),  # Very context-dependent
    ("plandemic", 0.16, False),
    ("scamdemic", 0.16, False),
    ("false flag", 0.14, False),
    ("controlled opposition", 0.13, False),
    ("puppet masters", 0.13, False),
    ("hidden agenda", 0.12, False),
    ("the elites", 0.09, True),
    ("lamestream media", 0.14, False),
    ("msm lies", 0.14, False),
    ("do your own research", 0.10, True),  # Context-dependent
    ("follow the money", 0.07, True),
    ("connect the dots", 0.06, True),
    ("open your eyes", 0.09, True),
    ("think for yourself", 0.06, True),
]

# Sensationalism indicators
_SENSATIONAL_PHRASES = [
    ("breaking news", 0.05, True),  # Often legitimate
    ("just in", 0.04, True),
    ("developing story", 0.03, True),
    ("exclusive", 0.05, True),
    ("urgent", 0.06, True),
    ("alert", 0.05, True),
    ("warning", 0.05, True),
    ("must read", 0.08, False),
    ("must see", 0.08, False),
    ("must watch", 0.08, False),
    ("game changer", 0.06, True),
    ("unprecedented", 0.04, True),
]

# Fear-mongering phrases
_FEAR_PHRASES = [
    ("will kill you", 0.12, True),
    ("deadly", 0.06, True),
    ("dangerous", 0.04, True),
    ("terrifying", 0.08, True),
    ("horrifying", 0.08, True),
    ("catastrophic", 0.06, True),
    ("end of the world", 0.10, True),
    ("collapse imminent", 0.11, False),
    ("crisis", 0.03, True),
    ("emergency", 0.04, True),
]

_URGENCY_WORDS = {
    "urgent", "now", "immediately", "warning", "alert", "breaking",
    "act now", "limited time", "hurry", "don't wait", "before it's too late",
    "last chance", "final warning", "time is running out",
}

_CERTAINTY = {
    "proves", "definitely", "guaranteed", "always", "never", "undeniable",
    "100%", "absolutely", "certainly", "without a doubt", "no question",
    "proven fact", "irrefutable", "unquestionable", "indisputable",
}

_HEDGES = {
    "may", "might", "could", "possibly", "suggests", "appears",
    "perhaps", "likely", "probably", "reportedly", "allegedly",
    "according to", "it seems", "evidence suggests", "research indicates",
}

# Context patterns that negate clickbait/conspiracy detection
_NEGATING_CONTEXTS = {
    "doctors hate": [
        r"doctors hate (having|working|dealing|waiting)",  # Legitimate usage
        r"doctors hate (paperwork|bureaucracy|insurance)",
    ],
    "shocking": [
        r"shocking (pink|blue|color)",  # Color description
        r"(electric|static) shocking",  # Electrical context
    ],
    "miracle": [
        r"miracle (on|at) (34th|ice|the)",  # Movie/sports references
        r"economic miracle",
        r"miracle mile",
    ],
    "secret": [
        r"(top|state|trade|military) secret",  # Legitimate usage
        r"secret (service|santa|agent|garden)",
    ],
    "exposed": [
        r"exposed (to|at|by) (the|a|an)",  # Physical exposure
        r"film exposed",
        r"exposed brick",
        r"(skin|area) exposed",
    ],
    "wake up": [
        r"wake up (early|late|at|in the)",  # Literal meaning
        r"wake up (call|time)",
    ],
    "cover-up": [
        r"(makeup|cover-up|concealer)",  # Cosmetics
    ],
    "big pharma": [
        r"big pharma (companies|industry|sector)",  # Neutral reference possible
    ],
    "the elites": [
        r"(sports|athletic|military) elites?",  # Legitimate usage
    ],
}


def _check_context(text: str, phrase: str, lower_text: str) -> bool:
    """Check if a phrase appears in a non-clickbait/conspiracy context.

    Returns True if the phrase should be flagged (suspicious context),
    False if the phrase appears in an innocent context.
    """
    if phrase not in _NEGATING_CONTEXTS:
        return True  # No negating patterns, flag it

    for pattern in _NEGATING_CONTEXTS[phrase]:
        if re.search(pattern, lower_text, re.IGNORECASE):
            return False  # Found innocent context, don't flag

    return True  # No innocent context found, flag it


def _detect_phrases(
    text: str,
    lower_text: str,
    phrase_list: list[tuple[str, float, bool]],
    category: str,
) -> tuple[list[EvidenceItem], list[str]]:
    """Detect phrases with optional context checking.

    Args:
        text: Original text
        lower_text: Lowercase text for matching
        phrase_list: List of (phrase, weight, requires_context_check) tuples
        category: Category name for evidence IDs

    Returns:
        Tuple of (signals list, highlights list)
    """
    signals: list[EvidenceItem] = []
    highlights: list[str] = []

    for phrase, weight, requires_context in phrase_list:
        if phrase in lower_text:
            # If context check required, verify it's not innocent usage
            if requires_context and not _check_context(text, phrase, lower_text):
                continue  # Skip - appears in innocent context

            spans = find_spans(text, phrase)
            signals.append(
                EvidenceItem(
                    id=f"{category}_phrase::{phrase}",
                    module="linguistic",
                    weight=weight,
                    value=0.85,
                    severity="high" if weight >= 0.12 else "medium",
                    evidence=f"{category.replace('_', ' ').title()} phrase detected: '{phrase}'.",
                    pointers=Pointer(char_spans=spans),
                    provenance={"context_checked": requires_context},
                )
            )
            highlights.append(phrase)

    return signals, highlights


def _word_boundary_search(pattern: str, text: str) -> bool:
    """Search for a word with proper word boundaries.

    Fixed version that doesn't double-escape.
    """
    # Use raw string for word boundary, escape the pattern content
    regex = r"\b" + re.escape(pattern) + r"\b"
    return bool(re.search(regex, text, re.IGNORECASE))


def _count_word_matches(word_set: set[str], text: str) -> list[str]:
    """Count matches of words from a set in text using proper word boundaries."""
    matches = []
    for word in word_set:
        if _word_boundary_search(word, text):
            matches.append(word)
    return matches


def analyze_linguistic(doc) -> LinguisticOutput:
    """Analyze text for linguistic patterns associated with misinformation.

    Detects linguistic red flags including:
    - Clickbait punctuation and ALL-CAPS text
    - Clickbait phrases (e.g., "you won't believe") with context awareness
    - Conspiracy framing language (e.g., "deep state", "cover-up")
    - Sensationalism indicators
    - Fear-mongering language
    - Urgency cues (e.g., "urgent", "breaking")
    - Certainty language imbalance (overuse of absolutes)
    - Anonymous authority claims

    Args:
        doc: Document object with text analysis (tokens, entities, etc.)

    Returns:
        LinguisticOutput with linguistic risk score (0-1), supporting evidence items,
        and highlighted suspicious phrases.
    """
    text = doc.display_text
    lower = text.lower()
    signals: list[EvidenceItem] = []
    highlights: list[str] = []

    # Thresholds from config
    punct_threshold = config.LING_CLICKBAIT_PUNCT_THRESHOLD
    caps_threshold = config.LING_CLICKBAIT_CAPS_THRESHOLD

    # Clickbait punctuation / caps
    exclam = text.count("!")
    qmarks = text.count("?")
    # Multiple consecutive punctuation is more suspicious
    multi_punct = len(re.findall(r"[!?]{2,}", text))

    caps_tokens = sum(1 for t in doc.tokens if t.text.isupper() and len(t.text) >= 3)
    cap_ratio = caps_tokens / max(1, len(doc.tokens))

    # Adjust scoring - multiple punctuation is more concerning
    punct_score = min(1.0, (exclam + qmarks) / 10.0 + multi_punct * 0.15)
    caps_score = min(1.0, cap_ratio * 8.0)

    if punct_score > punct_threshold:
        signals.append(
            EvidenceItem(
                id="clickbait_punct",
                module="linguistic",
                weight=0.10,
                value=punct_score,
                severity="medium" if punct_score < 0.6 else "high",
                evidence=f"High punctuation intensity ({exclam}! {qmarks}? {multi_punct} multi).",
                pointers=Pointer(char_spans=[]),
                provenance={
                    "exclamation": exclam,
                    "question": qmarks,
                    "multi_punct": multi_punct,
                    "threshold": punct_threshold,
                },
            )
        )

    if caps_score > caps_threshold:
        signals.append(
            EvidenceItem(
                id="clickbait_caps",
                module="linguistic",
                weight=0.10,
                value=caps_score,
                severity="medium" if caps_score < 0.6 else "high",
                evidence=f"Unusually high ALL-CAPS token ratio ({cap_ratio:.3f}).",
                pointers=Pointer(char_spans=[]),
                provenance={
                    "caps_tokens": caps_tokens,
                    "total_tokens": len(doc.tokens),
                    "threshold": caps_threshold,
                },
            )
        )

    # Detect phrases with context awareness
    clickbait_signals, clickbait_highlights = _detect_phrases(
        text, lower, _CLICKBAIT_PHRASES, "clickbait"
    )
    signals.extend(clickbait_signals)
    highlights.extend(clickbait_highlights)

    conspiracy_signals, conspiracy_highlights = _detect_phrases(
        text, lower, _CONSPIRACY_PHRASES, "conspiracy"
    )
    signals.extend(conspiracy_signals)
    highlights.extend(conspiracy_highlights)

    sensational_signals, sensational_highlights = _detect_phrases(
        text, lower, _SENSATIONAL_PHRASES, "sensational"
    )
    signals.extend(sensational_signals)
    highlights.extend(sensational_highlights)

    fear_signals, fear_highlights = _detect_phrases(
        text, lower, _FEAR_PHRASES, "fear"
    )
    signals.extend(fear_signals)
    highlights.extend(fear_highlights)

    # Urgency / emotion cues (lexicon) - FIXED: proper word boundary regex
    urgency_hits = _count_word_matches(_URGENCY_WORDS, lower)
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

    # Hedging vs certainty imbalance - FIXED: proper word boundary regex
    cert_hits = _count_word_matches(_CERTAINTY, lower)
    hedge_hits = _count_word_matches(_HEDGES, lower)
    cert = len(cert_hits)
    hedge = len(hedge_hits)

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
                provenance={
                    "certainty": cert,
                    "hedges": hedge,
                    "certainty_words": cert_hits,
                    "hedge_words": hedge_hits,
                },
            )
        )

    # Anonymous authority cues - expanded list
    anon_markers = [
        "experts say", "scientists say", "sources say", "researchers say",
        "studies show", "research shows", "doctors say", "officials say",
        "insiders reveal", "sources confirm", "reports indicate",
        "according to sources", "sources claim", "many people say",
        "some say", "they say", "people are saying",
    ]
    anon_hits = [p for p in anon_markers if p in lower]

    # Only flag if no named entities AND multiple anonymous references
    # OR if specific high-risk patterns exist
    high_risk_anon = ["insiders reveal", "sources confirm", "people are saying", "many people say"]
    has_high_risk_anon = any(p in lower for p in high_risk_anon)

    if anon_hits and (not doc.entities or has_high_risk_anon):
        weight = 0.12 if has_high_risk_anon else 0.08
        severity = "high" if has_high_risk_anon else "medium"
        signals.append(
            EvidenceItem(
                id="anonymous_authority",
                module="linguistic",
                weight=weight,
                value=0.75 if has_high_risk_anon else 0.5,
                severity=severity,
                evidence=f"Anonymous authority cues: {', '.join(anon_hits)}.",
                pointers=Pointer(char_spans=[]),
                provenance={"hits": anon_hits, "has_named_entities": bool(doc.entities)},
            )
        )

    # Emotional manipulation detection
    emotional_caps_phrases = re.findall(r"\b[A-Z]{4,}\b", text)
    if len(emotional_caps_phrases) >= 3:
        signals.append(
            EvidenceItem(
                id="emotional_caps_emphasis",
                module="linguistic",
                weight=0.08,
                value=min(1.0, len(emotional_caps_phrases) / 8.0),
                severity="medium",
                evidence=f"Multiple ALL-CAPS words for emphasis ({len(emotional_caps_phrases)} found).",
                pointers=Pointer(char_spans=[]),
                provenance={"caps_words": emotional_caps_phrases[:10]},
            )
        )

    # Risk score: bounded sum of weighted values (NOT final credibility; only linguistic risk).
    # Cap individual signal contribution to prevent single signal from dominating
    risk = 0.0
    for s in signals:
        contribution = s.weight * (s.value if s.value is not None else 0.5)
        risk += min(0.25, contribution)  # Cap individual contribution
    risk = max(0.0, min(1.0, risk))

    return LinguisticOutput(
        linguistic_risk_score=risk,
        confidence_score=1.0,  # Linguistic analysis is generally applicable regardless of length
        signals=signals,
        highlighted_phrases=highlights,
    )
