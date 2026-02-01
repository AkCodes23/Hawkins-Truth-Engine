from __future__ import annotations

from collections import Counter
from math import log

import numpy as np

from .. import config
from ..schemas import EvidenceItem, StatisticalOutput


def _lexical_diversity(tokens: list[str]) -> float:
    """Calculate type-token ratio (lexical diversity).

    Returns ratio of unique tokens to total tokens.
    Higher values indicate more diverse vocabulary.
    """
    if not tokens:
        return 0.0
    return len(set(tokens)) / max(1, len(tokens))


def _repetition_ratio(tokens: list[str]) -> float:
    """Calculate repetition ratio based on top 5 most frequent tokens.

    Returns proportion of text occupied by the 5 most common words.
    Higher values indicate more repetitive content.
    """
    if not tokens:
        return 0.0
    c = Counter(tokens)
    top = c.most_common(5)
    return sum(v for _, v in top) / max(1, len(tokens))


def _bigram_repetition(tokens: list[str]) -> float:
    """Calculate bigram repetition ratio.

    Repeated bigrams can indicate templated/generated text.
    """
    if len(tokens) < 4:
        return 0.0

    bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    c = Counter(bigrams)
    repeated = sum(count for count in c.values() if count > 1)
    return repeated / max(1, len(bigrams))


def _sentence_length_variance(sentences: list) -> tuple[float, float]:
    """Calculate sentence length statistics.

    Returns (mean_length, coefficient_of_variation).
    Very uniform sentence lengths can indicate generated content.
    """
    if not sentences:
        return 0.0, 0.0

    lengths = np.array([len(s.text.split()) for s in sentences], dtype=float)
    if len(lengths) < 2:
        return float(lengths.mean()) if len(lengths) == 1 else 0.0, 0.0

    mean_len = float(lengths.mean())
    std_len = float(lengths.std())
    cv = std_len / max(1.0, mean_len)

    return mean_len, cv


def _short_text_indicators(text: str, tokens: list[str]) -> list[dict]:
    """Analyze indicators specific to short texts.

    For texts too short for statistical analysis, we look for
    other patterns that might indicate low-quality content.
    """
    indicators = []

    # Check for excessive punctuation density
    punct_count = sum(1 for c in text if c in "!?.,;:")
    if len(text) > 0:
        punct_density = punct_count / len(text)
        if punct_density > 0.1:  # More than 10% punctuation
            indicators.append({
                "type": "high_punct_density",
                "value": punct_density,
                "weight": 0.08,
            })

    # Check for ALL CAPS ratio in short text
    if tokens:
        caps_ratio = sum(1 for t in tokens if t.isupper() and len(t) >= 2) / len(tokens)
        if caps_ratio > 0.3:  # More than 30% caps
            indicators.append({
                "type": "high_caps_ratio",
                "value": caps_ratio,
                "weight": 0.10,
            })

    # Check for very short sentences (fragmentary writing)
    words = text.split()
    if len(words) >= 3:
        avg_word_len = sum(len(w) for w in words) / len(words)
        if avg_word_len < 3.5:  # Very short average word length
            indicators.append({
                "type": "short_avg_word_length",
                "value": avg_word_len,
                "weight": 0.05,
            })

    # Check for repeated words in short text
    if len(tokens) >= 5:
        unique_ratio = len(set(tokens)) / len(tokens)
        if unique_ratio < 0.5:  # Less than 50% unique words
            indicators.append({
                "type": "low_uniqueness_short_text",
                "value": unique_ratio,
                "weight": 0.08,
            })

    return indicators


def analyze_statistical(doc) -> StatisticalOutput:
    """Analyze text for statistical patterns associated with misinformation.

    Detects patterns including:
    - Low lexical diversity (repetitive vocabulary)
    - High repetition of common words/phrases
    - Uniform sentence lengths (potential template content)
    - Low token entropy (irregular word distribution)
    - Short text specific indicators

    For short texts (< 50 tokens), uses specialized short-text analysis
    instead of returning zero risk.

    Args:
        doc: Document object with tokens, sentences, etc.

    Returns:
        StatisticalOutput with statistical risk score (0-1) and evidence items.
    """
    toks = [t.text.lower() for t in doc.tokens if t.text]
    evidence: list[EvidenceItem] = []

    # Configuration thresholds
    diversity_threshold = config.STAT_LOW_DIVERSITY_THRESHOLD
    repetition_threshold = config.STAT_HIGH_REPETITION_THRESHOLD
    uniform_threshold = config.STAT_UNIFORM_SENTENCE_THRESHOLD
    entropy_threshold = config.STAT_LOW_ENTROPY_THRESHOLD

    # Track if we have enough data for full analysis
    is_short_text = len(toks) < 50
    is_medium_text = 50 <= len(toks) < 120
    has_statistical_signals = False

    # For short texts, use specialized analysis
    if is_short_text:
        short_indicators = _short_text_indicators(doc.display_text, toks)

        for ind in short_indicators:
            evidence.append(
                EvidenceItem(
                    id=f"short_text_{ind['type']}",
                    module="statistical",
                    weight=ind["weight"],
                    value=min(1.0, ind["value"] * 2),  # Scale up for visibility
                    severity="low",
                    evidence=f"Short text indicator: {ind['type']} = {ind['value']:.3f}",
                    provenance={"token_count": len(toks), "indicator": ind},
                )
            )
            has_statistical_signals = True

        # Add uncertainty flag for short texts
        if not short_indicators:
            evidence.append(
                EvidenceItem(
                    id="insufficient_text_length",
                    module="statistical",
                    weight=0.0,  # Don't contribute to risk
                    value=0.0,
                    severity="low",
                    evidence=f"Text too short for full statistical analysis ({len(toks)} tokens).",
                    provenance={"token_count": len(toks), "minimum_recommended": 50},
                )
            )

    # Lexical diversity - adjust threshold for medium texts
    diversity = _lexical_diversity(toks)
    adjusted_diversity_threshold = diversity_threshold
    min_tokens_for_diversity = 100  # Lowered from 200

    if is_medium_text:
        # For medium texts, use slightly relaxed threshold
        adjusted_diversity_threshold = diversity_threshold * 0.9
        min_tokens_for_diversity = 50

    if diversity < adjusted_diversity_threshold and len(toks) >= min_tokens_for_diversity:
        evidence.append(
            EvidenceItem(
                id="low_lexical_diversity",
                module="statistical",
                weight=0.18,
                value=min(1.0, (adjusted_diversity_threshold - diversity) / adjusted_diversity_threshold),
                severity="medium",
                evidence=f"Low lexical diversity ({diversity:.3f}) for length={len(toks)}.",
                provenance={
                    "diversity": diversity,
                    "tokens": len(toks),
                    "threshold": adjusted_diversity_threshold,
                },
            )
        )
        has_statistical_signals = True

    # Repetition analysis - adjust for medium texts
    repetition = _repetition_ratio(toks)
    min_tokens_for_repetition = 60  # Lowered from 120

    if is_medium_text:
        min_tokens_for_repetition = 40

    if repetition > repetition_threshold and len(toks) >= min_tokens_for_repetition:
        evidence.append(
            EvidenceItem(
                id="high_repetition",
                module="statistical",
                weight=0.22,
                value=min(1.0, (repetition - repetition_threshold) / 0.20),
                severity="medium" if repetition < repetition_threshold + 0.10 else "high",
                evidence=f"High repetition ratio among top tokens ({repetition:.3f}).",
                provenance={"repetition": repetition, "threshold": repetition_threshold},
            )
        )
        has_statistical_signals = True

    # Bigram repetition (works for medium texts too)
    if len(toks) >= 30:
        bigram_rep = _bigram_repetition(toks)
        if bigram_rep > 0.15:  # More than 15% repeated bigrams
            evidence.append(
                EvidenceItem(
                    id="high_bigram_repetition",
                    module="statistical",
                    weight=0.12,
                    value=min(1.0, bigram_rep / 0.3),
                    severity="medium" if bigram_rep < 0.25 else "high",
                    evidence=f"High bigram repetition ({bigram_rep:.3f}).",
                    provenance={"bigram_repetition": bigram_rep},
                )
            )
            has_statistical_signals = True

    # Sentence length uniformity - lower threshold
    min_sentences = 5  # Lowered from 8
    if len(doc.sentences) >= min_sentences:
        mean_len, cv = _sentence_length_variance(doc.sentences)

        if cv < uniform_threshold:
            evidence.append(
                EvidenceItem(
                    id="uniform_sentence_length",
                    module="statistical",
                    weight=0.10,
                    value=min(1.0, (uniform_threshold - cv) / uniform_threshold),
                    severity="low",
                    evidence=f"Unusually uniform sentence lengths (CV={cv:.3f}, mean={mean_len:.1f} words).",
                    provenance={
                        "cv": cv,
                        "mean_length": mean_len,
                        "sentence_count": len(doc.sentences),
                        "threshold": uniform_threshold,
                    },
                )
            )
            has_statistical_signals = True

    # Token entropy - adjust for medium texts
    min_tokens_for_entropy = 100  # Lowered from 200

    if len(toks) >= min_tokens_for_entropy:
        c = Counter(toks)
        total = sum(c.values())
        probs = np.array([v / total for v in c.values()], dtype=float)
        entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        # Normalize by log2(V)
        v = len(c)
        norm = entropy / max(1e-6, log(v, 2))

        if norm < entropy_threshold:
            evidence.append(
                EvidenceItem(
                    id="low_token_entropy",
                    module="statistical",
                    weight=0.15,
                    value=min(1.0, (entropy_threshold - norm) / entropy_threshold),
                    severity="medium",
                    evidence=f"Low token distribution entropy (normalized={norm:.3f}).",
                    provenance={
                        "entropy_norm": norm,
                        "vocab": v,
                        "threshold": entropy_threshold,
                    },
                )
            )
            has_statistical_signals = True

    # Calculate risk score
    risk = 0.0
    for e in evidence:
        risk += e.weight * (e.value if e.value is not None else 0.5)

    # Calculate confidence based on text length
    # < 50 tokens: very low confidence
    # 50-120: medium confidence
    # > 120: high confidence
    if len(toks) < 50:
        confidence = 0.2
    elif len(toks) < 120:
        confidence = 0.5 + (len(toks) - 50) / 70 * 0.3  # 0.5 to 0.8
    else:
        confidence = min(1.0, 0.8 + (len(toks) - 120) / 100 * 0.2)  # 0.8 to 1.0

    # For short texts without signals, set a neutral baseline
    # instead of 0.0 or 0.15 (to avoid biasing the reasoning engine)
    if is_short_text and not has_statistical_signals:
        risk = 0.35  # Neutral risk baseline for unknown short text

    risk = max(0.0, min(1.0, risk))

    return StatisticalOutput(
        statistical_risk_score=risk, 
        confidence_score=confidence,
        evidence=evidence
    )
