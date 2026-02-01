from __future__ import annotations

import logging
import re
from datetime import datetime

from ..external.rdap import rdap_domain
from ..external.whois import whois_domain
from ..schemas import EvidenceItem, SourceIntelOutput

logger = logging.getLogger(__name__)


def _rdap_domain_age_days(rdap_json: dict) -> int | None:
    """Extract domain age in days from RDAP response."""
    events = rdap_json.get("events") or []
    reg = None
    for e in events:
        if e.get("eventAction") in {"registration", "registered"}:
            reg = e.get("eventDate")
            break
    if not reg:
        return None
    try:
        dt = datetime.fromisoformat(reg.replace("Z", "+00:00"))
        return max(0, (datetime.now(dt.tzinfo) - dt).days)
    except Exception:
        return None


def _analyze_text_credibility_signals(doc) -> tuple[list[EvidenceItem], float]:
    """Analyze raw text for credibility signals when no domain is available.

    For raw text/social posts, we can still assess credibility based on:
    - Presence of citations/references
    - Attribution quality (named sources vs anonymous)
    - Professional writing indicators
    - Social media specific patterns

    Returns:
        Tuple of (evidence_items, trust_adjustment)
    """
    flags: list[EvidenceItem] = []
    trust_adjustment = 0.0
    text = doc.display_text
    lower = text.lower()

    # Check for citations/references (positive signal)
    citation_patterns = [
        r"according to [\w\s]+,",
        r"(said|stated|reported) [\w\s]+ (university|institute|hospital|journal)",
        r"\(\d{4}\)",  # Year citations like (2024)
        r"doi:\s*10\.\d+",
        r"https?://",  # URLs as sources
        r"published in [\w\s]+",
        r"study (by|from|at) [\w\s]+",
    ]

    citation_count = 0
    for pattern in citation_patterns:
        if re.search(pattern, lower, re.IGNORECASE):
            citation_count += 1

    if citation_count >= 2:
        flags.append(
            EvidenceItem(
                id="has_citations",
                module="source",
                weight=0.15,
                value=0.3,  # Positive (lower value = good)
                severity="low",
                evidence=f"Text contains {citation_count} citation/reference patterns.",
                provenance={"citation_count": citation_count},
            )
        )
        trust_adjustment += 0.15
    elif citation_count == 0:
        flags.append(
            EvidenceItem(
                id="no_citations",
                module="source",
                weight=0.12,
                value=0.7,
                severity="medium",
                evidence="No citations or references found in text.",
                provenance={},
            )
        )
        trust_adjustment -= 0.10

    # Check for named vs anonymous sources
    named_entity_count = len(doc.entities) if doc.entities else 0
    has_attributions = len(doc.attributions) if doc.attributions else 0

    if named_entity_count >= 3 and has_attributions >= 1:
        flags.append(
            EvidenceItem(
                id="attributed_claims",
                module="source",
                weight=0.12,
                value=0.25,
                severity="low",
                evidence=f"Claims attributed to {named_entity_count} named entities.",
                provenance={
                    "named_entities": named_entity_count,
                    "attributions": has_attributions,
                },
            )
        )
        trust_adjustment += 0.12
    elif named_entity_count == 0 and has_attributions == 0:
        # Anonymous content with no attributions
        flags.append(
            EvidenceItem(
                id="anonymous_content",
                module="source",
                weight=0.15,
                value=0.75,
                severity="medium",
                evidence="No named sources or attributions in content.",
                provenance={},
            )
        )
        trust_adjustment -= 0.12

    # Check for social media indicators (neutral to slightly negative)
    social_patterns = [
        (r"#\w+", "hashtags"),
        (r"@\w+", "mentions"),
        (r"RT\s*:", "retweet"),
        (r"(share|repost|viral)", "viral_language"),
        (r"(follow|subscribe|like)", "engagement_bait"),
    ]

    social_indicators = []
    for pattern, name in social_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            social_indicators.append(name)

    if len(social_indicators) >= 3:
        flags.append(
            EvidenceItem(
                id="social_media_format",
                module="source",
                weight=0.10,
                value=0.6,
                severity="low",
                evidence=f"Social media format indicators: {', '.join(social_indicators)}.",
                provenance={"indicators": social_indicators},
            )
        )
        trust_adjustment -= 0.05  # Slight penalty for social media format

    # Professional writing indicators (expanded)
    professional_patterns = [
        r"(research|study|analysis|report|investigation) (shows|indicates|suggests|found|reveals)",
        r"(university|institute|organization|foundation|agency|department)",
        r"(published|peer-reviewed|journal|source|cited)",
        r"(data|statistics|evidence|findings|results|proof)",
        r"(according to|reported by|verified by)",
    ]

    professional_count = sum(
        1 for p in professional_patterns if re.search(p, lower, re.IGNORECASE)
    )

    if professional_count >= 3:
        flags.append(
            EvidenceItem(
                id="professional_indicators",
                module="source",
                weight=0.15,
                value=0.20,
                severity="low",
                evidence="Contains professional/academic writing style indicators.",
                provenance={"indicator_count": professional_count},
            )
        )
        trust_adjustment += 0.15
    elif professional_count >= 1:
        trust_adjustment += 0.05

    # Structure indicators (paragraphs, length)
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) >= 3 and len(doc.tokens) >= 150:
        flags.append(
            EvidenceItem(
                id="structured_content",
                module="source",
                weight=0.08,
                value=0.3,
                severity="low",
                evidence="Content is well-structured with multiple paragraphs.",
                provenance={"paragraph_count": len(paragraphs)},
            )
        )
        trust_adjustment += 0.10

    # Check for clickbait/sensationalism markers (often correlate with low source trust)
    sensational_patterns = [
        r"!!!+",
        r"\?\?\?+",
        r"(WAKE UP|SHARE THIS|BREAKING|URGENT|ACT NOW)",
        r"(they|government|media) (don't want|won't tell|are hiding)",
        r"(magical|miraculous|secret|exposed|bombshell)",
    ]

    sensational_count = sum(
        1 for p in sensational_patterns if re.search(p, text, re.IGNORECASE)
    )

    if sensational_count >= 2:
        flags.append(
            EvidenceItem(
                id="sensationalist_style",
                module="source",
                weight=0.20,
                value=0.85,
                severity="high",
                evidence="Content uses sensationalist or manipulative writing style.",
                provenance={"pattern_count": sensational_count},
            )
        )
        trust_adjustment -= 0.20

    return flags, trust_adjustment


async def analyze_source(doc) -> SourceIntelOutput:
    """Analyze source credibility for a document.

    For URL inputs:
    - Checks domain age via RDAP/WHOIS
    - Validates domain status
    - Checks author and publication metadata

    For raw text/social posts:
    - Analyzes text-based credibility signals
    - Checks for citations and attributions
    - Evaluates writing quality indicators

    Args:
        doc: Document object with domain, author, text content, etc.

    Returns:
        SourceIntelOutput with trust score (0-1) and evidence flags.
    """
    flags: list[EvidenceItem] = []
    trust = 0.5  # Start neutral

    # Handle non-URL inputs (raw_text, social_post)
    if not doc.domain:
        # Analyze text-based signals instead of just returning neutral
        text_flags, text_adjustment = _analyze_text_credibility_signals(doc)
        flags.extend(text_flags)
        trust += text_adjustment

        # Add metadata about input type
        input_type = doc.input_type if hasattr(doc, "input_type") else "unknown"
        flags.append(
            EvidenceItem(
                id="no_domain",
                module="source",
                weight=0.10,
                value=0.5,  # Neutral
                severity="low",
                evidence=f"No domain available ({input_type} input). Text-based analysis applied.",
                provenance={"input_type": input_type, "analysis_method": "text_based"},
            )
        )

        # Ensure trust is bounded
        trust = max(0.15, min(0.85, trust))
        return SourceIntelOutput(source_trust_score=trust, source_flags=flags)

    # Domain-based analysis for URL inputs
    try:
        rdap = await rdap_domain(doc.domain)
        rdap_data = rdap.get("data") or {}
        age_days = _rdap_domain_age_days(rdap_data)

        if age_days is not None and age_days < 90:
            flags.append(
                EvidenceItem(
                    id="young_domain",
                    module="source",
                    weight=0.35,
                    value=min(1.0, (90 - age_days) / 90),
                    severity="high",
                    evidence=f"Young domain (age ~{age_days} days via RDAP).",
                    provenance={
                        "rdap_url": rdap["request"]["url"],
                        "age_days": age_days,
                    },
                )
            )
        if age_days is not None and age_days >= 365:
            trust += 0.10
        if age_days is not None and age_days >= 1825:  # 5+ years
            trust += 0.05  # Additional bonus for established domains

        # RDAP status flags
        statuses = rdap_data.get("status") or []
        if any(
            "clienthold" in s.lower() or "serverhold" in s.lower() for s in statuses
        ):
            flags.append(
                EvidenceItem(
                    id="domain_hold_status",
                    module="source",
                    weight=0.25,
                    value=0.8,
                    severity="high",
                    evidence="Domain has hold status in RDAP (potentially unstable).",
                    provenance={
                        "status": statuses,
                        "rdap_url": rdap["request"]["url"],
                    },
                )
            )
    except Exception as e:
        logger.debug(
            f"RDAP lookup failed for {doc.domain}: {type(e).__name__}, trying WHOIS fallback"
        )
        # Try WHOIS fallback when RDAP fails
        try:
            whois = await whois_domain(doc.domain)
            if whois.get("success"):
                age_days = whois.get("data", {}).get("age_days")
                if age_days is not None and age_days < 90:
                    flags.append(
                        EvidenceItem(
                            id="young_domain_whois",
                            module="source",
                            weight=0.35,
                            value=min(1.0, (90 - age_days) / 90),
                            severity="high",
                            evidence=f"Young domain (age ~{age_days} days via WHOIS fallback).",
                            provenance={"source": "whois", "age_days": age_days},
                        )
                    )
                if age_days is not None and age_days >= 365:
                    trust += 0.10
            else:
                raise Exception(whois.get("error", "WHOIS lookup failed"))
        except Exception as whois_error:
            logger.warning(
                f"WHOIS fallback also failed for {doc.domain}: {str(whois_error)}"
            )
            flags.append(
                EvidenceItem(
                    id="rdap_unavailable",
                    module="source",
                    weight=0.15,  # Reduced from 0.20 - lookup failure shouldn't penalize too much
                    value=0.6,
                    severity="medium",
                    evidence="RDAP and WHOIS lookups failed; source age/stability unknown.",
                    provenance={
                        "rdap_error": str(e),
                        "whois_error": str(whois_error),
                    },
                )
            )

    # Author metadata check
    if not doc.author:
        flags.append(
            EvidenceItem(
                id="missing_author",
                module="source",
                weight=0.12,  # Slightly reduced
                value=0.7,
                severity="medium",
                evidence="Missing author/byline metadata.",
                provenance={},
            )
        )
        trust -= 0.08

    # Publication date check
    if not doc.published_at:
        flags.append(
            EvidenceItem(
                id="missing_pubdate",
                module="source",
                weight=0.08,  # Slightly reduced
                value=0.6,
                severity="low",
                evidence="Missing publication date metadata.",
                provenance={},
            )
        )
        trust -= 0.04

    # Also analyze text content for URL-based documents
    if doc.display_text:
        text_flags, text_adjustment = _analyze_text_credibility_signals(doc)
        # Apply text-based adjustment at reduced weight for URL inputs
        # (domain analysis is primary, text is secondary)
        flags.extend(text_flags)
        trust += text_adjustment * 0.5

    # Calculate final trust score with penalty from flags
    penalty = 0.0
    for f in flags:
        penalty += f.weight * (f.value if f.value is not None else 0.5)

    trust = max(0.05, min(0.95, trust - 0.5 * penalty))
    
    # Confidence is lower if RDAP/WHOIS failed
    confidence = 1.0
    if doc.domain and any(f.id == "rdap_unavailable" for f in flags):
        confidence = 0.6
    elif not doc.domain:
        confidence = 0.8  # Text-based analysis is fairly consistent but less certain than domain age
        
    return SourceIntelOutput(
        source_trust_score=trust, 
        confidence_score=confidence,
        source_flags=flags
    )
