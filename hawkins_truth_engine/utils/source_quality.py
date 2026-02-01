"""
Source Quality Assessment Module

Evaluates the credibility and trustworthiness of external sources used for
claim verification. This module provides functions to assess source quality
based on domain reputation, journal credibility, and source diversity.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse
from typing import Literal
from difflib import SequenceMatcher

# Reputable domain patterns (TLDs and known institutions)
_REPUTABLE_TLDS = {".edu", ".gov", ".org", ".ac.uk", ".ac.jp", ".ac.cn", ".gov.uk", ".gov.au"}

# Expanded reputable news domains
_REPUTABLE_NEWS_DOMAINS = {
    # Major wire services
    "reuters.com", "ap.org", "apnews.com", "afp.com",
    # UK/Europe
    "bbc.com", "bbc.co.uk", "theguardian.com", "ft.com", "economist.com",
    "telegraph.co.uk", "independent.co.uk", "thetimes.co.uk",
    "dw.com", "france24.com", "euronews.com", "spiegel.de",
    # US major outlets
    "nytimes.com", "washingtonpost.com", "wsj.com", "npr.org",
    "cnn.com", "abcnews.go.com", "cbsnews.com", "nbcnews.com",
    "usatoday.com", "latimes.com", "chicagotribune.com", "bostonglobe.com",
    "politico.com", "thehill.com", "axios.com", "theatlantic.com",
    "newyorker.com", "time.com", "newsweek.com", "usnews.com",
    # Business/Finance
    "bloomberg.com", "cnbc.com", "marketwatch.com", "fortune.com",
    "forbes.com", "businessinsider.com", "ft.com",
    # Science/Tech
    "nature.com", "science.org", "sciencedaily.com", "scientificamerican.com",
    "wired.com", "arstechnica.com", "techcrunch.com", "theverge.com",
    # Fact-checkers
    "snopes.com", "factcheck.org", "politifact.com", "fullfact.org",
    # International
    "aljazeera.com", "scmp.com", "japantimes.co.jp", "straitstimes.com",
    "thehindu.com", "abc.net.au", "cbc.ca", "globalnews.ca",
    # Medical/Health
    "mayoclinic.org", "webmd.com", "healthline.com", "medscape.com",
    "nih.gov", "cdc.gov", "who.int",
}

# Expanded questionable domain patterns (known misinformation sources)
_QUESTIONABLE_DOMAINS = {
    # Known misinformation sites
    "infowars.com", "naturalnews.com", "beforeitsnews.com",
    "thepeoplesvoice.tv", "worldtruth.tv", "yournewswire.com",
    "newspunch.com", "realrawnews.com", "thegatewaypundit.com",
    "zerohedge.com",  # Often unreliable
    "rt.com", "sputniknews.com",  # State propaganda
    "globalresearch.ca", "activistpost.com",
    "collective-evolution.com", "wakingtimes.com",
    "davidicke.com", "vaccineimpact.com",
    "greenmedinfo.com", "mercola.com",  # Medical misinformation
    # Satire sites sometimes shared as real
    "theonion.com", "babylonbee.com", "clickhole.com",
    "worldnewsdailyreport.com", "satirewire.com",
}

# Satire sites (separate category - not malicious but should be noted)
_SATIRE_DOMAINS = {
    "theonion.com", "babylonbee.com", "clickhole.com",
    "worldnewsdailyreport.com", "satirewire.com", "thedailymash.co.uk",
    "newsthump.com", "borowitz-report.com", "thebeaverton.com",
}

# Reputable medical/scientific journal patterns
_REPUTABLE_JOURNALS = {
    "nature", "science", "cell", "lancet", "nejm", "jama", "bmj",
    "plos", "pnas", "nature medicine", "science translational",
    "new england journal", "british medical journal",
    "annals of internal medicine", "circulation", "blood",
    "journal of clinical oncology", "journal of the american",
    "cochrane", "systematic review", "meta-analysis",
}

# Domain age thresholds (in days)
_MIN_REPUTABLE_AGE = 365  # At least 1 year old
_MIN_QUESTIONABLE_AGE = 90  # Less than 90 days is suspicious


def assess_domain_quality(domain: str | None) -> dict[str, any]:
    """
    Assess the quality and credibility of a domain.

    Args:
        domain: Domain name (e.g., "example.com") or None

    Returns:
        Dict with:
        - quality_score: float [0.0, 1.0] - overall quality score
        - reputation_tier: Literal["reputable", "neutral", "questionable", "unknown"]
        - indicators: list[str] - quality indicators
        - trust_boost: float - trust adjustment factor
        - is_satire: bool - whether the domain is a known satire site
    """
    if not domain:
        return {
            "quality_score": 0.3,
            "reputation_tier": "unknown",
            "indicators": ["no_domain_available"],
            "trust_boost": 0.0,
            "is_satire": False,
        }

    domain_lower = domain.lower().strip()

    # Remove www. prefix for matching
    if domain_lower.startswith("www."):
        domain_lower = domain_lower[4:]

    quality_score = 0.5  # Start neutral
    indicators: list[str] = []
    trust_boost = 0.0

    # Check for satire first
    if domain_lower in _SATIRE_DOMAINS:
        return {
            "quality_score": 0.3,  # Not necessarily low quality, but needs context
            "reputation_tier": "neutral",
            "indicators": ["satire_site"],
            "trust_boost": -0.15,
            "is_satire": True,
        }

    # Check for reputable TLDs
    has_reputable_tld = any(domain_lower.endswith(tld) for tld in _REPUTABLE_TLDS)
    if has_reputable_tld:
        quality_score += 0.2
        trust_boost += 0.15
        indicators.append("reputable_tld")

    # Check for known reputable news domains
    if domain_lower in _REPUTABLE_NEWS_DOMAINS:
        quality_score = min(1.0, quality_score + 0.3)
        trust_boost += 0.25
        indicators.append("reputable_news_source")

    # Check for subdomain of reputable domain
    for reputable in _REPUTABLE_NEWS_DOMAINS:
        if domain_lower.endswith("." + reputable):
            quality_score = min(1.0, quality_score + 0.25)
            trust_boost += 0.20
            indicators.append("reputable_subdomain")
            break

    # Check for questionable domains
    if domain_lower in _QUESTIONABLE_DOMAINS:
        quality_score = max(0.0, quality_score - 0.5)
        trust_boost -= 0.35
        indicators.append("questionable_source")

    # Check for suspicious patterns in domain
    suspicious_patterns = [
        r"news\d+",  # news123.com
        r"\d{3,}",  # Multiple digits
        r"-(breaking|viral|exposed|truth|real)-",
        r"^(real|true|honest|patriot)",
        r"(insider|leaked|banned|censored)",
    ]
    for pattern in suspicious_patterns:
        if re.search(pattern, domain_lower):
            quality_score = max(0.0, quality_score - 0.1)
            trust_boost -= 0.05
            indicators.append("suspicious_domain_pattern")
            break

    # Determine reputation tier
    if quality_score >= 0.7:
        reputation_tier: Literal["reputable", "neutral", "questionable", "unknown"] = "reputable"
    elif quality_score <= 0.3:
        reputation_tier = "questionable"
    else:
        reputation_tier = "neutral"

    return {
        "quality_score": max(0.0, min(1.0, quality_score)),
        "reputation_tier": reputation_tier,
        "indicators": indicators,
        "trust_boost": trust_boost,
        "is_satire": False,
    }


def assess_journal_quality(journal_name: str | None) -> dict[str, any]:
    """
    Assess the quality of a scientific journal.

    Args:
        journal_name: Journal name or None

    Returns:
        Dict with:
        - quality_score: float [0.0, 1.0]
        - reputation_tier: Literal["reputable", "neutral", "unknown"]
        - indicators: list[str]
        - trust_boost: float
    """
    if not journal_name:
        return {
            "quality_score": 0.5,
            "reputation_tier": "unknown",
            "indicators": ["no_journal_name"],
            "trust_boost": 0.0,
        }

    journal_lower = journal_name.lower()
    quality_score = 0.5
    indicators: list[str] = []
    trust_boost = 0.0

    # Check for reputable journal patterns
    for pattern in _REPUTABLE_JOURNALS:
        if pattern in journal_lower:
            quality_score = min(1.0, quality_score + 0.35)
            trust_boost += 0.30
            indicators.append("reputable_journal")
            break

    # Check for peer-reviewed indicators
    if any(term in journal_lower for term in ["peer review", "peer-reviewed", "refereed"]):
        quality_score = min(1.0, quality_score + 0.1)
        trust_boost += 0.1
        indicators.append("peer_reviewed")

    # Check for preprint servers (less authoritative than peer-reviewed)
    if any(term in journal_lower for term in ["preprint", "arxiv", "biorxiv", "medrxiv"]):
        indicators.append("preprint")
        # Preprints are okay but less authoritative
        quality_score = max(0.4, quality_score - 0.1)

    reputation_tier: Literal["reputable", "neutral", "unknown"]
    if quality_score >= 0.7:
        reputation_tier = "reputable"
    elif quality_score <= 0.3:
        reputation_tier = "unknown"
    else:
        reputation_tier = "neutral"

    return {
        "quality_score": max(0.0, min(1.0, quality_score)),
        "reputation_tier": reputation_tier,
        "indicators": indicators,
        "trust_boost": trust_boost,
    }


def extract_domain_from_url(url: str | None) -> str | None:
    """Extract domain from a URL string."""
    if not url:
        return None
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split("/")[0]
        # Remove port if present
        domain = domain.split(":")[0]
        return domain.lower().strip()
    except Exception:
        return None


def assess_citation_quality(citation: dict[str, any]) -> dict[str, any]:
    """
    Assess the quality of a citation from external sources.

    Args:
        citation: Citation dict with keys like 'url', 'domain', 'journal', 'title', etc.

    Returns:
        Dict with quality assessment metrics
    """
    domain = citation.get("domain") or extract_domain_from_url(citation.get("url"))
    journal = citation.get("journal") or citation.get("fulljournalname")

    domain_assessment = assess_domain_quality(domain)
    journal_assessment = assess_journal_quality(journal)

    # Combine assessments
    quality_score = (domain_assessment["quality_score"] * 0.6 +
                     journal_assessment["quality_score"] * 0.4)
    trust_boost = domain_assessment["trust_boost"] + journal_assessment["trust_boost"]

    indicators = domain_assessment["indicators"] + journal_assessment["indicators"]

    # Additional quality checks
    if citation.get("pmid"):  # PubMed ID indicates peer-reviewed source
        quality_score = min(1.0, quality_score + 0.15)
        trust_boost += 0.12
        indicators.append("pubmed_indexed")

    if citation.get("pubtypes") and any("review" in str(pt).lower() for pt in citation.get("pubtypes", [])):
        quality_score = min(1.0, quality_score + 0.05)
        trust_boost += 0.05
        indicators.append("review_article")

    # Check for DOI (indicates formal publication)
    if citation.get("doi"):
        quality_score = min(1.0, quality_score + 0.05)
        indicators.append("has_doi")

    reputation_tier: Literal["reputable", "neutral", "questionable", "unknown"]
    if quality_score >= 0.7:
        reputation_tier = "reputable"
    elif quality_score <= 0.3:
        reputation_tier = "questionable"
    else:
        reputation_tier = "neutral"

    return {
        "quality_score": max(0.0, min(1.0, quality_score)),
        "reputation_tier": reputation_tier,
        "indicators": indicators,
        "trust_boost": trust_boost,
        "domain_assessment": domain_assessment,
        "journal_assessment": journal_assessment,
        "is_satire": domain_assessment.get("is_satire", False),
    }


def calculate_source_diversity(citations: list[dict[str, any]]) -> dict[str, any]:
    """
    Calculate source diversity metrics for a set of citations.

    Args:
        citations: List of citation dicts

    Returns:
        Dict with diversity metrics:
        - unique_domains: int
        - unique_providers: int (PubMed, GDELT, Tavily)
        - diversity_score: float [0.0, 1.0]
        - domain_list: list[str]
    """
    if not citations:
        return {
            "unique_domains": 0,
            "unique_providers": 0,
            "diversity_score": 0.0,
            "domain_list": [],
        }

    domains = set()
    providers = set()

    for citation in citations:
        domain = citation.get("domain") or extract_domain_from_url(citation.get("url"))
        if domain:
            domains.add(domain)

        provider = citation.get("provider", "unknown")
        providers.add(provider)

    unique_domains = len(domains)
    unique_providers = len(providers)

    # Diversity score: higher when more unique sources from different providers
    # Normalize by number of citations
    total_citations = len(citations)
    if total_citations > 0:
        domain_diversity = min(1.0, unique_domains / total_citations)
        provider_diversity = min(1.0, unique_providers / 3.0)  # Max 3 providers
        diversity_score = (domain_diversity * 0.7 + provider_diversity * 0.3)
    else:
        diversity_score = 0.0

    return {
        "unique_domains": unique_domains,
        "unique_providers": unique_providers,
        "diversity_score": diversity_score,
        "domain_list": sorted(list(domains)),
    }


def assess_source_authority(citation: dict[str, any]) -> float:
    """
    Assess the authority level of a source based on its type and characteristics.

    Authority hierarchy:
    - Peer-reviewed journals (PubMed): 1.0
    - Government/Educational (.gov, .edu): 0.9
    - Reputable news sources: 0.8
    - General news: 0.6
    - Web search results: 0.5
    - Blogs/Unknown: 0.3
    - Questionable sources: 0.1

    Args:
        citation: Citation dict

    Returns:
        Authority score [0.0, 1.0]
    """
    # PubMed sources are highest authority (peer-reviewed)
    if citation.get("pmid") or citation.get("provider") == "pubmed":
        return 1.0

    # Government and educational domains
    domain = citation.get("domain") or extract_domain_from_url(citation.get("url"))
    if domain:
        domain_lower = domain.lower()
        # Remove www prefix
        if domain_lower.startswith("www."):
            domain_lower = domain_lower[4:]

        # Check government/educational TLDs
        if any(domain_lower.endswith(tld) for tld in [".gov", ".edu", ".ac.uk", ".ac.jp", ".gov.uk"]):
            return 0.9

        # Check reputable news
        if domain_lower in _REPUTABLE_NEWS_DOMAINS:
            return 0.8

        # Check questionable sources
        if domain_lower in _QUESTIONABLE_DOMAINS:
            return 0.1

        # Check satire
        if domain_lower in _SATIRE_DOMAINS:
            return 0.2

    # Journal-based assessment
    journal = citation.get("journal") or citation.get("fulljournalname")
    if journal:
        journal_lower = journal.lower()
        if any(pattern in journal_lower for pattern in _REPUTABLE_JOURNALS):
            return 0.95

    # Default based on provider
    provider = citation.get("provider", "unknown")
    if provider == "pubmed":
        return 1.0
    elif provider == "gdelt":
        return 0.6  # News sources
    elif provider == "tavily":
        return 0.5  # Web search (mixed quality)

    return 0.3  # Unknown/low authority


def analyze_temporal_relevance(citations: list[dict[str, any]], claim_date: str | None = None) -> dict[str, any]:
    """
    Analyze temporal relevance of citations.

    Args:
        citations: List of citation dicts
        claim_date: Optional date of the claim (for recency analysis)

    Returns:
        Dict with temporal metrics
    """
    from datetime import datetime, timedelta

    if not citations:
        return {
            "avg_age_days": None,
            "newest_age_days": None,
            "oldest_age_days": None,
            "recency_score": 0.0,
            "temporal_relevance": "unknown",
        }

    ages = []
    now = datetime.now()

    for citation in citations:
        # Try to extract date from various fields
        date_str = (
            citation.get("pubdate") or
            citation.get("seendate") or
            citation.get("published_date") or
            citation.get("date")
        )

        if date_str:
            try:
                # Try parsing various date formats
                if isinstance(date_str, str):
                    # Common formats: "2024-01-15", "2024/01/15", "Jan 15, 2024"
                    for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%b %d, %Y", "%Y-%m", "%Y"]:
                        try:
                            pub_date = datetime.strptime(date_str[:10], fmt)
                            age_days = (now - pub_date).days
                            if 0 <= age_days <= 3650:  # Reasonable range (0-10 years)
                                ages.append(age_days)
                            break
                        except ValueError:
                            continue
            except Exception:
                pass

    if not ages:
        return {
            "avg_age_days": None,
            "newest_age_days": None,
            "oldest_age_days": None,
            "recency_score": 0.5,  # Neutral if no dates
            "temporal_relevance": "unknown",
        }

    avg_age = sum(ages) / len(ages)
    newest_age = min(ages)
    oldest_age = max(ages)

    # Recency score: newer is better
    # Normalize: 0 days = 1.0, 365 days = 0.5, 730+ days = 0.0
    recency_score = max(0.0, min(1.0, 1.0 - (avg_age / 730.0)))

    # Categorize temporal relevance
    if avg_age <= 30:
        temporal_relevance = "very_recent"
    elif avg_age <= 180:
        temporal_relevance = "recent"
    elif avg_age <= 365:
        temporal_relevance = "moderate"
    elif avg_age <= 730:
        temporal_relevance = "old"
    else:
        temporal_relevance = "very_old"

    return {
        "avg_age_days": int(avg_age),
        "newest_age_days": newest_age,
        "oldest_age_days": oldest_age,
        "recency_score": recency_score,
        "temporal_relevance": temporal_relevance,
    }


def _semantic_similarity(text1: str, text2: str) -> float:
    """Calculate simple semantic similarity between two texts.

    Uses multiple methods for better accuracy:
    - Word overlap (Jaccard)
    - Sequence matching
    - Key phrase overlap

    Returns similarity score [0.0, 1.0]
    """
    if not text1 or not text2:
        return 0.0

    t1_lower = text1.lower()
    t2_lower = text2.lower()

    # Extract significant words (4+ chars)
    words1 = set(re.findall(r"\b[a-z]{4,}\b", t1_lower))
    words2 = set(re.findall(r"\b[a-z]{4,}\b", t2_lower))

    if not words1 or not words2:
        return 0.0

    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    jaccard = intersection / union if union > 0 else 0.0

    # Sequence matcher for phrase similarity
    seq_ratio = SequenceMatcher(None, t1_lower, t2_lower).ratio()

    # Key number/figure overlap
    numbers1 = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text1))
    numbers2 = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text2))
    number_overlap = 0.0
    if numbers1 and numbers2:
        number_overlap = len(numbers1 & numbers2) / len(numbers1 | numbers2)

    # Weighted combination
    similarity = jaccard * 0.5 + seq_ratio * 0.3 + number_overlap * 0.2

    return min(1.0, similarity)


def detect_contradictions(citations: list[dict[str, any]], claim: str) -> dict[str, any]:
    """
    Detect potential contradictions in citations using improved analysis.

    Uses multiple methods:
    - Keyword-based contradiction detection
    - Semantic similarity between citation content
    - Negation pattern detection

    Args:
        citations: List of citation dicts
        claim: The claim text being verified

    Returns:
        Dict with contradiction metrics
    """
    if not citations:
        return {
            "contradiction_score": 0.0,
            "supporting_count": 0,
            "contradicting_count": 0,
            "neutral_count": 0,
            "contradicting_indicators": [],
        }

    # Expanded contradiction keywords with context
    contradicting_patterns = [
        (r"\b(debunk|debunked|debunking)\b", 0.9),
        (r"\b(false|fake|hoax|fabricat)\b", 0.85),
        (r"\b(misleading|misinformation|disinformation)\b", 0.85),
        (r"\b(incorrect|inaccurate|wrong)\b", 0.7),
        (r"\b(disproven|refuted|rejected)\b", 0.85),
        (r"\b(no evidence|lacks evidence|insufficient evidence)\b", 0.75),
        (r"\b(myth|misconception)\b", 0.7),
        (r"\b(contrary to|contradicts|disputes)\b", 0.8),
        (r"\bnot true\b", 0.85),
        (r"\bhas been denied\b", 0.75),
    ]

    supporting_patterns = [
        (r"\b(confirm|confirmed|confirms)\b", 0.8),
        (r"\b(verified|verifies|verification)\b", 0.85),
        (r"\b(proven|proved|proof)\b", 0.7),
        (r"\b(true|accurate|correct)\b", 0.6),
        (r"\b(supports?|supported|supporting)\b", 0.7),
        (r"\b(evidence shows|data shows|study shows)\b", 0.75),
        (r"\b(consistent with|in line with)\b", 0.7),
        (r"\bresearch indicates\b", 0.65),
    ]

    supporting_count = 0
    contradicting_count = 0
    neutral_count = 0
    contradicting_indicators = []
    stance_scores = []

    for citation in citations:
        # Combine all text from citation
        text_to_analyze = " ".join([
            citation.get("title", ""),
            citation.get("content", ""),
            " ".join(citation.get("snippets", [])),
        ]).lower()

        if not text_to_analyze.strip():
            neutral_count += 1
            continue

        # Calculate contradiction and support scores
        contradiction_score = 0.0
        support_score = 0.0

        for pattern, weight in contradicting_patterns:
            if re.search(pattern, text_to_analyze, re.IGNORECASE):
                contradiction_score = max(contradiction_score, weight)

        for pattern, weight in supporting_patterns:
            if re.search(pattern, text_to_analyze, re.IGNORECASE):
                support_score = max(support_score, weight)

        # Check for negation of claim terms in citation
        claim_key_terms = set(re.findall(r"\b[a-z]{4,}\b", claim.lower()))
        for term in claim_key_terms:
            if re.search(rf"\b(not|no|never|neither)\b[^.]*\b{re.escape(term)}\b", text_to_analyze):
                contradiction_score = max(contradiction_score, 0.6)

        # Determine stance
        if contradiction_score > support_score and contradiction_score >= 0.5:
            contradicting_count += 1
            contradicting_indicators.append(
                f"contradicting_signal_in_{citation.get('provider', 'unknown')}"
            )
            stance_scores.append(-contradiction_score)
        elif support_score > contradiction_score and support_score >= 0.5:
            supporting_count += 1
            stance_scores.append(support_score)
        else:
            neutral_count += 1
            stance_scores.append(0.0)

    # Check for semantic contradictions between citations
    semantic_contradictions = 0
    if len(citations) >= 2:
        snippets_list = []
        for c in citations:
            snippets = c.get("snippets", [])
            if snippets:
                snippets_list.append(" ".join(snippets))

    # Compare pairs of snippets for contradictory content
    for i in range(len(snippets_list)):
        for j in range(i + 1, len(snippets_list)):
            s1 = snippets_list[i]
            s2 = snippets_list[j]
            sim = _semantic_similarity(s1, s2)
            
            # If talking about same thing, check negation
            if sim > 0.4:
                neg1 = len(re.findall(r"\b(not|no|never|neither|without|deny|denies|denied)\b", s1.lower()))
                neg2 = len(re.findall(r"\b(not|no|never|neither|without|deny|denies|denied)\b", s2.lower()))
                if abs(neg1 - neg2) >= 1:
                    semantic_contradictions += 1

    # Calculate overall contradiction score
    total = len(citations)
    if total > 0:
        contradiction_ratio = contradicting_count / total

        # Factor in semantic contradictions (more weight)
        semantic_factor = min(0.5, semantic_contradictions * 0.2)

        # If both supporting and contradicting exist, it's contested
        if supporting_count > 0 and contradicting_count > 0:
            base_score = 0.5 + (contradiction_ratio * 0.3)
        else:
            base_score = contradiction_ratio

        contradiction_score = min(1.0, base_score + semantic_factor)
        
        # Special case: single source but it's explicitly a "debunking" source
        if total == 1 and contradicting_count == 1:
            contradiction_score = max(contradiction_score, 0.7)
    else:
        contradiction_score = 0.0

    return {
        "contradiction_score": contradiction_score,
        "supporting_count": supporting_count,
        "contradicting_count": contradicting_count,
        "neutral_count": neutral_count,
        "contradicting_indicators": contradicting_indicators,
        "semantic_contradictions": semantic_contradictions,
    }


def calculate_evidence_strength(
    citations: list[dict[str, any]],
    claim: str,
    medical_topic: bool = False,
) -> dict[str, any]:
    """
    Calculate overall evidence strength for a claim.

    Combines multiple factors:
    - Source quality and authority
    - Source diversity
    - Temporal relevance
    - Contradiction detection
    - Medical topic special requirements

    Args:
        citations: List of citation dicts
        claim: Claim text
        medical_topic: Whether this is a medical claim

    Returns:
        Dict with evidence strength metrics
    """
    if not citations:
        return {
            "strength_score": 0.0,
            "strength_tier": "very_weak",
            "factors": {
                "source_count": 0,
                "quality_score": 0.0,
                "authority_score": 0.0,
                "diversity_score": 0.0,
                "recency_score": 0.0,
                "contradiction_penalty": 0.0,
            },
        }

    # Assess each citation
    quality_scores = []
    authority_scores = []

    for citation in citations:
        quality_assessment = assess_citation_quality(citation)
        quality_scores.append(quality_assessment["quality_score"])
        authority_scores.append(assess_source_authority(citation))

    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
    avg_authority = sum(authority_scores) / len(authority_scores) if authority_scores else 0.0

    # Source diversity
    diversity = calculate_source_diversity(citations)
    diversity_score = diversity["diversity_score"]

    # Temporal relevance
    temporal = analyze_temporal_relevance(citations)
    recency_score = temporal["recency_score"]

    # Contradiction detection
    contradictions = detect_contradictions(citations, claim)
    contradiction_penalty = contradictions["contradiction_score"] * 0.35  # Increased penalty

    # Count factors
    source_count = len(citations)
    reputable_count = sum(1 for qs in quality_scores if qs >= 0.7)
    high_authority_count = sum(1 for as_ in authority_scores if as_ >= 0.8)

    # Calculate base strength with improved weighting
    base_strength = (
        min(1.0, source_count / 4.0) * 0.15 +  # Source count (normalized to 4)
        avg_quality * 0.25 +                    # Quality
        avg_authority * 0.25 +                  # Authority
        diversity_score * 0.15 +                # Diversity
        recency_score * 0.10 +                  # Recency
        (reputable_count / max(1, source_count)) * 0.10  # Reputable ratio
    )

    # Medical claims - relaxed requirements but still stricter than general
    if medical_topic:
        if high_authority_count == 0:
            # No high-authority source - moderate penalty instead of severe
            base_strength *= 0.75  # Relaxed from 0.6
        if source_count < 2:
            base_strength *= 0.8  # Relaxed from 0.7

    # Apply contradiction penalty
    strength_score = max(0.0, min(1.0, base_strength - contradiction_penalty))

    # Determine strength tier
    if strength_score >= 0.75:
        strength_tier = "very_strong"
    elif strength_score >= 0.55:
        strength_tier = "strong"
    elif strength_score >= 0.35:
        strength_tier = "moderate"
    elif strength_score >= 0.15:
        strength_tier = "weak"
    else:
        strength_tier = "very_weak"

    return {
        "strength_score": strength_score,
        "strength_tier": strength_tier,
        "factors": {
            "source_count": source_count,
            "quality_score": avg_quality,
            "authority_score": avg_authority,
            "diversity_score": diversity_score,
            "recency_score": recency_score,
            "contradiction_penalty": contradiction_penalty,
            "reputable_count": reputable_count,
            "high_authority_count": high_authority_count,
        },
    }
