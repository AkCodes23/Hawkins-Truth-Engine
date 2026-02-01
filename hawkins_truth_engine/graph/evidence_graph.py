"""
Evidence Graph construction module for the Hawkins Truth Engine.

This module provides functionality to build evidence graphs that represent
evidential relationships between claims based on similarity and external corroboration.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from ..schemas import (
    ClaimItem,
    ClaimsOutput,
    EvidenceGraph,
    GraphEdge,
)


def calculate_claim_similarity(claim1: ClaimItem, claim2: ClaimItem) -> float:
    """
    Calculates semantic similarity between two claims using enhanced text analysis.
    
    Args:
        claim1: First claim to compare
        claim2: Second claim to compare
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if claim1.id == claim2.id:
        return 1.0
    
    # Enhanced text normalization
    def normalize_text(text: str) -> str:
        """Normalize text for better comparison."""
        # Convert to lowercase
        text = text.lower()
        
        # Handle contractions
        contractions = {
            "n't": " not",
            "'re": " are", 
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'s": ""  # Remove possessive 's
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove punctuation and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    # Normalize both claim texts
    text1 = normalize_text(claim1.text)
    text2 = normalize_text(claim2.text)
    
    # Tokenize into words
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    # Remove common stop words that don't contribute to semantic meaning
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
        'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 
        'may', 'might', 'can', 'this', 'that', 'these', 'those', 'it', 'they', 
        'them', 'their', 'there', 'here', 'where', 'when', 'how', 'why', 'what'
    }
    
    words1 = words1 - stop_words
    words2 = words2 - stop_words
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity (intersection over union)
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    jaccard_similarity = intersection / union if union > 0 else 0.0
    
    # Calculate additional similarity factors
    
    # 1. Length similarity (claims of very different lengths are less likely to be similar)
    len1, len2 = len(words1), len(words2)
    length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0
    
    # 2. Key term overlap (boost similarity if important terms overlap)
    # Look for named entities, numbers, and important keywords
    important_patterns = [
        r'\b\d+(?:\.\d+)?%?\b',  # Numbers and percentages
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns (simplified)
        r'\b(?:study|research|report|data|evidence|analysis|survey|poll)\b',  # Research terms
        r'\b(?:increase|decrease|rise|fall|grow|decline|improve|worsen)\b',  # Change terms
        r'\b(?:cause|effect|result|lead|due|because|since)\b',  # Causal terms
    ]
    
    important_terms1 = set()
    important_terms2 = set()
    
    for pattern in important_patterns:
        important_terms1.update(re.findall(pattern, text1, re.IGNORECASE))
        important_terms2.update(re.findall(pattern, text2, re.IGNORECASE))
    
    # Normalize important terms
    important_terms1 = {term.lower() for term in important_terms1}
    important_terms2 = {term.lower() for term in important_terms2}
    
    important_overlap = 0.0
    if important_terms1 and important_terms2:
        important_intersection = len(important_terms1 & important_terms2)
        important_union = len(important_terms1 | important_terms2)
        important_overlap = important_intersection / important_union if important_union > 0 else 0.0
    
    # 3. Claim type similarity (claims of same type are more likely to be related)
    type_similarity = 1.0 if claim1.type == claim2.type else 0.7
    
    # Combine all similarity factors with weights
    final_similarity = (
        jaccard_similarity * 0.5 +      # Base text similarity
        length_ratio * 0.1 +            # Length similarity
        important_overlap * 0.3 +       # Important terms overlap
        type_similarity * 0.1           # Type similarity
    )
    
    return min(final_similarity, 1.0)


def determine_evidence_relationship(claim1: ClaimItem, claim2: ClaimItem,
                                  similarity: float,
                                  corroboration: dict) -> str | None:
    """
    Enhanced relationship determination with improved logic for SUPPORTS, CONTRADICTS, and RELATES_TO.
    
    Args:
        claim1: First claim
        claim2: Second claim  
        similarity: Similarity score between claims
        corroboration: External corroboration data
        
    Returns:
        Relationship type ("SUPPORTS", "CONTRADICTS", "RELATES_TO") or None
    """
    # Minimum similarity threshold for any relationship
    if similarity < 0.15:  # Lowered threshold for better coverage
        return None
    
    # Get support statuses and quality information
    support1 = claim1.support
    support2 = claim2.support
    quality1 = claim1.quality_flags or []
    quality2 = claim2.quality_flags or []
    citations1 = claim1.citations or []
    citations2 = claim2.citations or []
    
    # Get external corroboration confidence for both claims
    corr1 = corroboration.get(claim1.id, {})
    corr2 = corroboration.get(claim2.id, {})
    
    corr1_confidence = corr1.get('confidence', 0.5)
    corr2_confidence = corr2.get('confidence', 0.5)
    corr1_supported = corr1.get('supported', None)
    corr2_supported = corr2.get('supported', None)
    
    # Calculate quality scores for enhanced decision making
    def calculate_quality_score(quality_flags: list, citations: list) -> float:
        """Calculate quality score based on flags and citations."""
        score = 0.5  # Base score
        
        if "high_quality" in quality_flags:
            score += 0.3
        elif "low_quality" in quality_flags:
            score -= 0.2
            
        if "verified_source" in quality_flags:
            score += 0.2
        elif "questionable_source" in quality_flags:
            score -= 0.2
            
        # Citation bonus
        if citations:
            score += min(len(citations) * 0.05, 0.2)  # Max 0.2 bonus
            
        return max(0.0, min(score, 1.0))
    
    quality_score1 = calculate_quality_score(quality1, citations1)
    quality_score2 = calculate_quality_score(quality2, citations2)
    avg_quality = (quality_score1 + quality_score2) / 2
    
    # Enhanced similarity thresholds based on quality
    high_similarity = similarity >= (0.6 - avg_quality * 0.1)  # Quality can lower threshold
    moderate_similarity = similarity >= (0.35 - avg_quality * 0.05)
    low_similarity = similarity >= 0.2
    
    # ENHANCED SUPPORTS relationship logic
    if high_similarity:
        # Strong support: Both claims have consistent support status
        if support1 == support2 and support1 in ["supported", "unsupported"]:
            # External corroboration also consistent
            if (corr1_supported is not None and corr2_supported is not None and 
                corr1_supported == corr2_supported):
                return "SUPPORTS"
            # High quality claims with strong internal consistency
            elif avg_quality > 0.7 and corr1_confidence > 0.6 and corr2_confidence > 0.6:
                return "SUPPORTS"
            # Even moderate quality with very high similarity
            elif similarity > 0.8 and avg_quality > 0.5:
                return "SUPPORTS"
        
        # Graduated support: One claim strongly supported, other moderately supported
        support_hierarchy = {"supported": 3, "contested": 2, "unverifiable": 1, "unsupported": 0}
        support1_level = support_hierarchy.get(support1, 1)
        support2_level = support_hierarchy.get(support2, 1)
        
        # Claims with similar support levels
        if abs(support1_level - support2_level) <= 1 and min(support1_level, support2_level) >= 2:
            if corr1_confidence > 0.5 and corr2_confidence > 0.5 and avg_quality > 0.6:
                return "SUPPORTS"
    
    # Moderate similarity support cases
    elif moderate_similarity:
        # Same support status with good quality
        if support1 == support2 and support1 in ["supported", "contested"]:
            if avg_quality > 0.7 and (corr1_confidence + corr2_confidence) / 2 > 0.6:
                return "SUPPORTS"
        
        # Complementary evidence: different types but consistent direction
        if (claim1.type != claim2.type and 
            support1 in ["supported", "contested"] and support2 in ["supported", "contested"]):
            if avg_quality > 0.6 and similarity > 0.45:
                return "SUPPORTS"
    
    # ENHANCED CONTRADICTS relationship logic
    if moderate_similarity:
        # Direct contradiction in support status
        contradiction_pairs = [
            ("supported", "unsupported"),
            ("unsupported", "supported"),
            ("supported", "contested"),  # Added contested as potential contradiction
            ("contested", "unsupported")
        ]
        
        if (support1, support2) in contradiction_pairs or (support2, support1) in contradiction_pairs:
            # External corroboration confirms contradiction
            if (corr1_supported is not None and corr2_supported is not None and 
                corr1_supported != corr2_supported):
                return "CONTRADICTS"
            # Strong confidence in contradictory assessments
            elif corr1_confidence > 0.6 and corr2_confidence > 0.6 and avg_quality > 0.5:
                return "CONTRADICTS"
            # High similarity with clear contradiction
            elif similarity > 0.7 and avg_quality > 0.6:
                return "CONTRADICTS"
        
        # Semantic contradiction detection (enhanced)
        contradiction_indicators = [
            ("increase", "decrease"), ("rise", "fall"), ("grow", "decline"),
            ("improve", "worsen"), ("positive", "negative"), ("true", "false"),
            ("real", "fake"), ("valid", "invalid"), ("correct", "incorrect"),
            ("effective", "ineffective"), ("safe", "dangerous"), ("beneficial", "harmful")
        ]
        
        text1_lower = claim1.text.lower()
        text2_lower = claim2.text.lower()
        
        for pos_term, neg_term in contradiction_indicators:
            if ((pos_term in text1_lower and neg_term in text2_lower) or
                (neg_term in text1_lower and pos_term in text2_lower)):
                if similarity > 0.4 and avg_quality > 0.5:
                    return "CONTRADICTS"
    
    # Low similarity contradiction (strong semantic opposition)
    elif low_similarity and similarity > 0.25:
        # Look for direct negation patterns
        negation_patterns = [
            ("is", "is not"), ("are", "are not"), ("will", "will not"),
            ("can", "cannot"), ("should", "should not"), ("does", "does not")
        ]
        
        text1_lower = claim1.text.lower()
        text2_lower = claim2.text.lower()
        
        for pos_pattern, neg_pattern in negation_patterns:
            if ((pos_pattern in text1_lower and neg_pattern in text2_lower) or
                (neg_pattern in text1_lower and pos_pattern in text2_lower)):
                if avg_quality > 0.6 and (corr1_confidence + corr2_confidence) / 2 > 0.6:
                    return "CONTRADICTS"
    
    # ENHANCED RELATES_TO relationship logic (topical similarity without clear evidence relationship)
    if moderate_similarity:
        # Both claims are unverifiable but topically related
        if support1 == "unverifiable" and support2 == "unverifiable":
            return "RELATES_TO"
        
        # Mixed support statuses without clear pattern
        mixed_statuses = {support1, support2}
        if len(mixed_statuses) > 1 and ("contested" in mixed_statuses or "unverifiable" in mixed_statuses):
            return "RELATES_TO"
        
        # Similar claims with uncertain external corroboration
        if (corr1_confidence < 0.6 or corr2_confidence < 0.6) and similarity >= 0.4:
            return "RELATES_TO"
        
        # Claims of different types but similar content (enhanced)
        if claim1.type != claim2.type and similarity >= 0.4:
            # Boost relationship for certain type combinations
            related_type_pairs = [
                ("factual", "opinion_presented_as_fact"),
                ("factual", "speculation"),
                ("opinion_presented_as_fact", "speculation")
            ]
            
            if ((claim1.type, claim2.type) in related_type_pairs or 
                (claim2.type, claim1.type) in related_type_pairs):
                return "RELATES_TO"
            elif similarity >= 0.5:  # Higher threshold for unrelated types
                return "RELATES_TO"
        
        # Temporal or causal relationships
        temporal_indicators = ["before", "after", "during", "while", "when", "since", "until"]
        causal_indicators = ["because", "due to", "caused by", "leads to", "results in", "affects"]
        
        text1_words = claim1.text.lower().split()
        text2_words = claim2.text.lower().split()
        
        has_temporal = any(word in temporal_indicators for word in text1_words + text2_words)
        has_causal = any(word in causal_indicators for word in text1_words + text2_words)
        
        if (has_temporal or has_causal) and similarity >= 0.3:
            return "RELATES_TO"
    
    # Low similarity topical relationships
    elif low_similarity:
        # Domain-specific relationships (same domain, different specifics)
        domain_keywords = {
            "health": ["health", "medical", "disease", "treatment", "patient", "doctor", "hospital"],
            "climate": ["climate", "weather", "temperature", "warming", "environment", "carbon"],
            "economy": ["economy", "economic", "financial", "market", "business", "trade", "money"],
            "politics": ["political", "government", "policy", "election", "vote", "candidate"],
            "technology": ["technology", "digital", "computer", "internet", "software", "data"]
        }
        
        text1_lower = claim1.text.lower()
        text2_lower = claim2.text.lower()
        
        for domain, keywords in domain_keywords.items():
            count1 = sum(1 for keyword in keywords if keyword in text1_lower)
            count2 = sum(1 for keyword in keywords if keyword in text2_lower)
            
            if count1 >= 2 and count2 >= 2:  # Both claims have multiple domain keywords
                return "RELATES_TO"
    
    # Default: no clear relationship
    return None


def create_evidence_edges(claims: list[ClaimItem], 
                         corroboration: dict) -> list[GraphEdge]:
    """
    Creates evidence relationship edges between claims with performance optimizations.
    
    Args:
        claims: List of claims to analyze
        corroboration: External corroboration data
        
    Returns:
        List of evidence relationship edges
    """
    edges: list[GraphEdge] = []
    edge_counter = 0
    
    def next_edge_id() -> str:
        nonlocal edge_counter
        edge_counter += 1
        return f"evidence_edge:{edge_counter}"
    
    # Performance optimization: Pre-compute claim features for faster comparison
    claim_features = {}
    for claim in claims:
        # Pre-normalize text for similarity calculation
        normalized_text = _normalize_text_for_similarity(claim.text)
        words = set(normalized_text.split())
        
        # Remove stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 
            'may', 'might', 'can', 'this', 'that', 'these', 'those', 'it', 'they', 
            'them', 'their', 'there', 'here', 'where', 'when', 'how', 'why', 'what'
        }
        words = words - stop_words
        
        # Pre-extract important terms
        important_terms = _extract_important_terms(normalized_text)
        
        claim_features[claim.id] = {
            'words': words,
            'important_terms': important_terms,
            'text_length': len(words),
            'normalized_text': normalized_text
        }
    
    # Performance optimization: Early filtering based on basic similarity
    def quick_similarity_check(claim1_id: str, claim2_id: str) -> float:
        """Quick similarity check to filter out obviously unrelated claims."""
        features1 = claim_features[claim1_id]
        features2 = claim_features[claim2_id]
        
        words1 = features1['words']
        words2 = features2['words']
        
        if not words1 or not words2:
            return 0.0
        
        # Quick Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    # Performance optimization: Batch process claims in chunks for large datasets
    total_pairs = len(claims) * (len(claims) - 1) // 2
    processed_pairs = 0
    
    # Compare each pair of claims with optimizations
    for i, claim1 in enumerate(claims):
        for j, claim2 in enumerate(claims):
            if i >= j:  # Avoid duplicate comparisons and self-comparison
                continue
            
            processed_pairs += 1
            
            # Performance optimization: Quick similarity pre-filter
            quick_sim = quick_similarity_check(claim1.id, claim2.id)
            if quick_sim < 0.1:  # Skip obviously unrelated claims
                continue
            
            # Calculate full similarity only for potentially related claims
            similarity = calculate_claim_similarity_optimized(claim1, claim2, claim_features)
            
            # Determine relationship type
            relationship_type = determine_evidence_relationship(
                claim1, claim2, similarity, corroboration
            )
            
            if relationship_type is None:
                continue
            
            # Calculate edge weight based on enhanced design formula
            similarity_score = similarity
            
            # Enhanced evidence strength calculation
            def calculate_evidence_strength(claim: ClaimItem) -> float:
                """Calculate enhanced evidence strength for a claim."""
                base_strength = {
                    "supported": 0.85,      # Increased from 0.8
                    "unsupported": 0.15,    # Decreased from 0.2 for clearer distinction
                    "contested": 0.5,       # Neutral
                    "unverifiable": 0.25    # Decreased from 0.3
                }.get(claim.support, 0.5)
                
                # Quality adjustments (enhanced)
                quality_multiplier = 1.0
                if claim.quality_flags:
                    if "high_quality" in claim.quality_flags:
                        quality_multiplier *= 1.3  # Increased from 1.2
                    elif "low_quality" in claim.quality_flags:
                        quality_multiplier *= 0.7  # Decreased from 0.8
                    
                    if "verified_source" in claim.quality_flags:
                        quality_multiplier *= 1.2
                    elif "questionable_source" in claim.quality_flags:
                        quality_multiplier *= 0.6
                    
                    if "peer_reviewed" in claim.quality_flags:
                        quality_multiplier *= 1.25
                    elif "unreviewed" in claim.quality_flags:
                        quality_multiplier *= 0.8
                
                # Citation adjustments (enhanced)
                citation_multiplier = 1.0
                if claim.citations:
                    num_citations = len(claim.citations)
                    # Logarithmic scaling for citations
                    import math
                    citation_bonus = min(math.log(num_citations + 1) * 0.1, 0.3)  # Max 0.3 bonus
                    citation_multiplier = 1.0 + citation_bonus
                    
                    # Bonus for diverse citation types
                    citation_types = set()
                    for citation in claim.citations:
                        if isinstance(citation, dict) and 'type' in citation:
                            citation_types.add(citation['type'])
                    
                    if len(citation_types) > 1:
                        citation_multiplier *= 1.1  # Diversity bonus
                
                # Apply multipliers
                final_strength = base_strength * quality_multiplier * citation_multiplier
                
                return min(final_strength, 1.0)
            
            evidence_strength1 = calculate_evidence_strength(claim1)
            evidence_strength2 = calculate_evidence_strength(claim2)
            
            # Relationship-specific evidence strength calculation
            if relationship_type == "SUPPORTS":
                # For supporting relationships, use the average but boost if both are strong
                evidence_strength = (evidence_strength1 + evidence_strength2) / 2
                if evidence_strength1 > 0.7 and evidence_strength2 > 0.7:
                    evidence_strength *= 1.15  # Boost for mutual strong evidence
            elif relationship_type == "CONTRADICTS":
                # For contradicting relationships, use the higher strength (stronger contradiction)
                evidence_strength = max(evidence_strength1, evidence_strength2)
                # Boost if both claims have strong but opposing evidence
                if min(evidence_strength1, evidence_strength2) > 0.6:
                    evidence_strength *= 1.1
            else:  # RELATES_TO
                # For topical relationships, use average with slight penalty for uncertainty
                evidence_strength = (evidence_strength1 + evidence_strength2) / 2 * 0.9
            
            # Enhanced corroboration confidence calculation
            corr1 = corroboration.get(claim1.id, {})
            corr2 = corroboration.get(claim2.id, {})
            corr_confidence1 = corr1.get('confidence', 0.5)
            corr_confidence2 = corr2.get('confidence', 0.5)
            
            # Relationship-specific corroboration weighting
            if relationship_type == "SUPPORTS":
                # For supports, both should have similar corroboration
                corroboration_confidence = (corr_confidence1 + corr_confidence2) / 2
                # Bonus if both are highly confident and consistent
                corr1_supported = corr1.get('supported', None)
                corr2_supported = corr2.get('supported', None)
                if (corr1_supported is not None and corr2_supported is not None and 
                    corr1_supported == corr2_supported and 
                    corr_confidence1 > 0.7 and corr_confidence2 > 0.7):
                    corroboration_confidence *= 1.2
            elif relationship_type == "CONTRADICTS":
                # For contradicts, use average but boost if confidently opposing
                corroboration_confidence = (corr_confidence1 + corr_confidence2) / 2
                corr1_supported = corr1.get('supported', None)
                corr2_supported = corr2.get('supported', None)
                if (corr1_supported is not None and corr2_supported is not None and 
                    corr1_supported != corr2_supported and 
                    corr_confidence1 > 0.6 and corr_confidence2 > 0.6):
                    corroboration_confidence *= 1.15
            else:  # RELATES_TO
                # For relates_to, average with slight penalty for uncertainty
                corroboration_confidence = (corr_confidence1 + corr_confidence2) / 2 * 0.95
            
            # Ensure corroboration confidence is in valid range
            corroboration_confidence = max(0.0, min(corroboration_confidence, 1.0))
            
            # Enhanced edge weight formula with relationship-specific adjustments
            base_weight = (
                similarity_score * 0.35 +           # Slightly reduced similarity weight
                evidence_strength * 0.45 +          # Increased evidence weight
                corroboration_confidence * 0.2      # Maintained corroboration weight
            )
            
            # Relationship-specific weight adjustments
            relationship_multipliers = {
                "SUPPORTS": 1.0,        # No adjustment for supports
                "CONTRADICTS": 1.05,    # Slight boost for contradictions (important to identify)
                "RELATES_TO": 0.9       # Slight penalty for topical relationships
            }
            
            weight = base_weight * relationship_multipliers.get(relationship_type, 1.0)
            
            # Quality-based final adjustment
            avg_quality = (calculate_evidence_strength(claim1) + calculate_evidence_strength(claim2)) / 2
            if avg_quality > 0.8:
                weight *= 1.05  # Boost for high-quality claim pairs
            elif avg_quality < 0.3:
                weight *= 0.9   # Penalty for low-quality claim pairs
            
            # Ensure weight is in valid range
            weight = max(0.0, min(weight, 1.0))
            
            # Create edge (bidirectional relationship, but we create one edge)
            edge = GraphEdge(
                id=next_edge_id(),
                source_id=f"claim:{claim1.id}",
                target_id=f"claim:{claim2.id}",
                relationship_type=relationship_type,
                weight=weight,
                provenance={
                    "similarity_score": similarity_score,
                    "evidence_strength": evidence_strength,
                    "evidence_strength_claim1": evidence_strength1,
                    "evidence_strength_claim2": evidence_strength2,
                    "corroboration_confidence": corroboration_confidence,
                    "corr_confidence_claim1": corr_confidence1,
                    "corr_confidence_claim2": corr_confidence2,
                    "claim1_support": claim1.support,
                    "claim2_support": claim2.support,
                    "claim1_type": claim1.type,
                    "claim2_type": claim2.type,
                    "claim1_quality": claim1.quality_flags or [],
                    "claim2_quality": claim2.quality_flags or [],
                    "claim1_citations": len(claim1.citations or []),
                    "claim2_citations": len(claim2.citations or []),
                    "relationship_determination_method": "enhanced_evidence_analysis_v2",
                    "quick_similarity_prefilter": quick_sim
                },
                created_at=datetime.now()
            )
            
            edges.append(edge)
    
    return edges


def _normalize_text_for_similarity(text: str) -> str:
    """Normalize text for similarity calculation (extracted for reuse)."""
    # Convert to lowercase
    text = text.lower()
    
    # Handle contractions
    contractions = {
        "n't": " not",
        "'re": " are", 
        "'ve": " have",
        "'ll": " will",
        "'d": " would",
        "'s": ""  # Remove possessive 's
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Remove punctuation and normalize whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def _extract_important_terms(text: str) -> set:
    """Extract important terms from text (extracted for reuse)."""
    important_patterns = [
        r'\b\d+(?:\.\d+)?%?\b',  # Numbers and percentages
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Proper nouns (simplified)
        r'\b(?:study|research|report|data|evidence|analysis|survey|poll)\b',  # Research terms
        r'\b(?:increase|decrease|rise|fall|grow|decline|improve|worsen)\b',  # Change terms
        r'\b(?:cause|effect|result|lead|due|because|since)\b',  # Causal terms
    ]
    
    important_terms = set()
    for pattern in important_patterns:
        important_terms.update(re.findall(pattern, text, re.IGNORECASE))
    
    # Normalize important terms
    return {term.lower() for term in important_terms}


def calculate_claim_similarity_optimized(claim1: ClaimItem, claim2: ClaimItem, 
                                       claim_features: dict) -> float:
    """
    Optimized version of claim similarity calculation using pre-computed features.
    
    Args:
        claim1: First claim to compare
        claim2: Second claim to compare
        claim_features: Pre-computed features for all claims
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if claim1.id == claim2.id:
        return 1.0
    
    # Use pre-computed features
    features1 = claim_features[claim1.id]
    features2 = claim_features[claim2.id]
    
    words1 = features1['words']
    words2 = features2['words']
    important_terms1 = features1['important_terms']
    important_terms2 = features2['important_terms']
    
    if not words1 or not words2:
        return 0.0
    
    # Calculate Jaccard similarity (intersection over union)
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    jaccard_similarity = intersection / union if union > 0 else 0.0
    
    # Calculate additional similarity factors
    
    # 1. Length similarity (claims of very different lengths are less likely to be similar)
    len1, len2 = features1['text_length'], features2['text_length']
    length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0.0
    
    # 2. Important term overlap
    important_overlap = 0.0
    if important_terms1 and important_terms2:
        important_intersection = len(important_terms1 & important_terms2)
        important_union = len(important_terms1 | important_terms2)
        important_overlap = important_intersection / important_union if important_union > 0 else 0.0
    
    # 3. Claim type similarity (claims of same type are more likely to be related)
    type_similarity = 1.0 if claim1.type == claim2.type else 0.7
    
    # Combine all similarity factors with weights
    final_similarity = (
        jaccard_similarity * 0.5 +      # Base text similarity
        length_ratio * 0.1 +            # Length similarity
        important_overlap * 0.3 +       # Important terms overlap
        type_similarity * 0.1           # Type similarity
    )
    
    return min(final_similarity, 1.0)


def build_evidence_graph(claims_output: ClaimsOutput, 
                        external_corroboration: dict) -> EvidenceGraph:
    """
    Constructs evidence graph from claims and corroboration data with enhanced performance tracking.
    
    Args:
        claims_output: Output from claims analysis containing claim items
        external_corroboration: External corroboration data for claims
        
    Returns:
        EvidenceGraph containing claim relationships
    """
    import time
    start_time = time.time()
    
    # Input validation - handle invalid inputs gracefully
    if claims_output is None or not hasattr(claims_output, 'claim_items'):
        return EvidenceGraph(
            claim_nodes={},
            edges={},
            similarity_threshold=0.4,
            metadata={
                "num_claims": 0,
                "num_edges": 0,
                "construction_time": time.time() - start_time,
                "error": "Invalid claims_output provided",
                "builder_version": "2.0_enhanced",
            },
            created_at=datetime.now()
        )
    
    # Handle None corroboration
    if external_corroboration is None:
        external_corroboration = {}
    
    claims = claims_output.claim_items
    
    # Performance optimization: Skip graph construction for very small datasets
    if len(claims) < 2:
        return EvidenceGraph(
            claim_nodes={claim.id: f"claim:{claim.id}" for claim in claims},
            edges={},
            similarity_threshold=0.4,
            metadata={
                "num_claims": len(claims),
                "num_edges": 0,
                "construction_time": time.time() - start_time,
                "edges_per_claim_pair": 0.0,
                "relationship_distribution": {"SUPPORTS": 0, "CONTRADICTS": 0, "RELATES_TO": 0},
                "average_edge_weight": 0.0,
                "builder_version": "2.0_enhanced",
                "optimization_notes": "Skipped construction for dataset with < 2 claims"
            },
            created_at=datetime.now()
        )
    
    # Create claim node mapping
    claim_nodes = {claim.id: f"claim:{claim.id}" for claim in claims}
    
    # Create evidence edges with enhanced algorithms
    evidence_edges = create_evidence_edges(claims, external_corroboration)
    
    # Convert edges list to dictionary
    edges_dict = {edge.id: edge for edge in evidence_edges}
    
    # Calculate enhanced performance metrics
    construction_time = time.time() - start_time
    total_possible_pairs = len(claims) * (len(claims) - 1) // 2
    
    # Calculate relationship distribution
    relationship_distribution = _calculate_relationship_distribution(evidence_edges)
    
    # Calculate average edge weight
    avg_edge_weight = sum(edge.weight for edge in evidence_edges) / max(len(evidence_edges), 1)
    
    # Calculate quality metrics
    quality_metrics = _calculate_quality_metrics(evidence_edges, claims)
    
    # Calculate performance metrics
    performance_metrics = {
        "edges_per_second": len(evidence_edges) / max(construction_time, 0.001),
        "pairs_processed_per_second": total_possible_pairs / max(construction_time, 0.001),
        "edge_creation_rate": len(evidence_edges) / max(total_possible_pairs, 1),
        "memory_efficiency_score": _calculate_memory_efficiency(claims, evidence_edges)
    }
    
    # Create evidence graph with enhanced metadata
    evidence_graph = EvidenceGraph(
        claim_nodes=claim_nodes,
        edges=edges_dict,
        similarity_threshold=0.35,  # Updated threshold for enhanced algorithm
        metadata={
            "num_claims": len(claims),
            "num_edges": len(evidence_edges),
            "construction_time": construction_time,
            "total_possible_pairs": total_possible_pairs,
            "edges_per_claim_pair": len(evidence_edges) / max(total_possible_pairs, 1),
            "relationship_distribution": relationship_distribution,
            "average_edge_weight": avg_edge_weight,
            "quality_metrics": quality_metrics,
            "performance_metrics": performance_metrics,
            "builder_version": "2.0_enhanced",
            "algorithm_enhancements": [
                "enhanced_supports_detection",
                "improved_contradicts_identification", 
                "better_topical_similarity",
                "sophisticated_edge_weighting",
                "performance_optimizations"
            ]
        },
        created_at=datetime.now()
    )
    
    return evidence_graph


def _calculate_quality_metrics(edges: list[GraphEdge], claims: list[ClaimItem]) -> dict[str, Any]:
    """Calculate quality metrics for the evidence graph."""
    if not edges:
        return {
            "high_confidence_edges": 0,
            "low_confidence_edges": 0,
            "avg_similarity_score": 0.0,
            "avg_evidence_strength": 0.0,
            "avg_corroboration_confidence": 0.0,
            "quality_distribution": {"high": 0, "medium": 0, "low": 0}
        }
    
    high_confidence_edges = sum(1 for edge in edges if edge.weight > 0.7)
    low_confidence_edges = sum(1 for edge in edges if edge.weight < 0.3)
    
    similarity_scores = [edge.provenance.get("similarity_score", 0.0) for edge in edges]
    evidence_strengths = [edge.provenance.get("evidence_strength", 0.0) for edge in edges]
    corr_confidences = [edge.provenance.get("corroboration_confidence", 0.0) for edge in edges]
    
    # Quality distribution based on edge weights
    quality_dist = {"high": 0, "medium": 0, "low": 0}
    for edge in edges:
        if edge.weight > 0.7:
            quality_dist["high"] += 1
        elif edge.weight > 0.4:
            quality_dist["medium"] += 1
        else:
            quality_dist["low"] += 1
    
    return {
        "high_confidence_edges": high_confidence_edges,
        "low_confidence_edges": low_confidence_edges,
        "avg_similarity_score": sum(similarity_scores) / len(similarity_scores),
        "avg_evidence_strength": sum(evidence_strengths) / len(evidence_strengths),
        "avg_corroboration_confidence": sum(corr_confidences) / len(corr_confidences),
        "quality_distribution": quality_dist,
        "edge_weight_std": _calculate_standard_deviation([edge.weight for edge in edges])
    }


def _calculate_memory_efficiency(claims: list[ClaimItem], edges: list[GraphEdge]) -> float:
    """Calculate a memory efficiency score for the graph construction."""
    # Simple heuristic: ratio of useful edges to total possible edges
    total_possible = len(claims) * (len(claims) - 1) // 2
    if total_possible == 0:
        return 1.0
    
    # Weight by edge quality (higher weight edges are more "useful")
    useful_edges = sum(edge.weight for edge in edges)
    max_possible_useful = total_possible * 1.0  # If all edges had weight 1.0
    
    return useful_edges / max(max_possible_useful, 1.0)


def _calculate_standard_deviation(values: list[float]) -> float:
    """Calculate standard deviation of a list of values."""
    if not values:
        return 0.0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def _calculate_relationship_distribution(edges: list[GraphEdge]) -> dict[str, int]:
    """Calculate distribution of relationship types in edges."""
    distribution = {"SUPPORTS": 0, "CONTRADICTS": 0, "RELATES_TO": 0}
    
    for edge in edges:
        if edge.relationship_type in distribution:
            distribution[edge.relationship_type] += 1
    
    return distribution