from __future__ import annotations

import logging
import re
from typing import Literal

from rapidfuzz import fuzz

from ..config import GROQ_API_KEY, PUBMED_MAX_ABSTRACTS, TAVILY_API_KEY

logger = logging.getLogger(__name__)
from ..external.gdelt import gdelt_doc_search
from ..external.groq import extract_claims_with_llm, is_groq_available
from ..external.ncbi import pubmed_efetch_abstract, pubmed_esearch, pubmed_esummary
from ..external.tavily import tavily_search
from ..schemas import ClaimItem, ClaimsOutput, Pointer
from ..utils import find_spans
from ..utils.source_quality import (
    assess_citation_quality,
    assess_source_authority,
    calculate_source_diversity,
    calculate_evidence_strength,
    detect_contradictions,
    extract_domain_from_url,
)


_MED_TERMS = {
    "cure",
    "treat",
    "treatment",
    "vaccine",
    "vaccines",
    "side effect",
    "adverse",
    "cancer",
    "covid",
    "diabetes",
    "autism",
    "antibiotic",
    "ivermectin",
    "hydroxychloroquine",
}

_STRONG_MED_CLAIM = re.compile(
    r"\b(cures?|guaranteed|100%|no side effects|miracle)\b", re.IGNORECASE
)


def _medical_topic_triggers(text: str) -> list[str]:
    lower = text.lower()
    hits = []
    for t in _MED_TERMS:
        if t in lower:
            hits.append(t)
    return hits


def _claim_candidates(sentences: list[str]) -> list[str]:
    # POC: treat declarative sentences longer than a threshold as claims.
    cands: list[str] = []
    for s in sentences:
        ss = s.strip()
        if len(ss) < 25:
            continue
        if ss.endswith("?"):
            continue
        cands.append(ss)
    return cands[:12]


async def _claim_candidates_llm(doc) -> tuple[list[dict], list[str]]:
    """
    Extract claims using Groq LLM for more intelligent extraction.
    
    Args:
        doc: Document object with display_text and sentences
        
    Returns:
        Tuple of (list of claim dicts from LLM, list of risk indicators)
    """
    if not is_groq_available():
        return [], []
    
    try:
        sentences = [s.text for s in doc.sentences]
        result = await extract_claims_with_llm(doc.display_text, sentences)
        
        if result.get("error"):
            logger.warning(f"LLM claim extraction failed: {result['error']}")
            return [], []
        
        claims = result.get("claims", [])
        risk_indicators = result.get("risk_indicators", [])
        
        logger.info(f"LLM extracted {len(claims)} claims")
        return claims, risk_indicators
        
    except Exception as e:
        logger.warning(f"LLM claim extraction error: {type(e).__name__}: {e}")
        return [], []


ClaimType = Literal["factual", "speculative", "predictive", "opinion_presented_as_fact"]


def _claim_type(sentence: str) -> ClaimType:
    lower = sentence.lower()
    if any(w in lower for w in ("will ", "going to", "by 20")):
        return "predictive"
    if any(w in lower for w in ("might", "may", "could", "possibly", "suggest")):
        return "speculative"
    if any(w in lower for w in ("i think", "we believe", "in my opinion")):
        return "opinion_presented_as_fact"
    return "factual"


async def _llm_veracity_fallback(claim: str) -> dict:
    """Use LLM to estimate veracity when external APIs are unavailable.
    
    This is a fallback mechanism and results are marked with low confidence.
    """
    if not is_groq_available():
        return {"support": "unverifiable", "reason": "llm_unavailable"}
    
    from ..external.groq import groq_chat_completion
    
    prompt = f"""Analyze the veracity of the following claim based on common knowledge as of late 2024.
    Claim: "{claim}"
    
    Return a JSON object only with exactly these keys:
    - support: "supported", "unsupported", or "unverifiable"
    - reason: brief explanation
    - confidence: 0.0 to 1.0
    """
    
    try:
        # Note: We use a very low temperature for factual checks
        result = await groq_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        
        from ..external.groq import get_completion_text
        text = get_completion_text(result)
        
        if not text:
            return {"support": "unverifiable", "reason": "llm_no_response"}
            
        import json
        # Extract JSON from response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            # Reduce confidence because it's a model hallucination risk
            data["confidence"] = data.get("confidence", 0.5) * 0.5 
            return data
            
        return {"support": "unverifiable", "reason": "llm_parsing_failed"}
    except Exception as e:
        logger.warning(f"LLM veracity fallback failed: {e}")
        return {"support": "unverifiable", "reason": "llm_exception"}


def _snippet_relevance(snippet: str, claim: str) -> float:
    """
    Enhanced snippet relevance scoring using multiple factors.
    """
    # Filter out very short words
    cs = {w for w in re.findall(r"\b[a-z0-9]{4,}\b", claim.lower())}
    ss = {w for w in re.findall(r"\b[a-z0-9]{4,}\b", snippet.lower())}
    
    if not cs:
        return 0.0
    
    # Jaccard similarity (intersection over union)
    intersection = len(cs & ss)
    union = len(cs | ss)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Coverage (how much of the claim is covered by snippet)
    coverage = intersection / len(cs) if cs else 0.0
    
    # Fuzzy match for paraphrasing
    fuzzy_score = fuzz.partial_ratio(claim.lower(), snippet.lower()) / 100.0
    
    # Check for exact matches of long phrases (3+ words)
    phrase_match = 0.0
    claim_words = claim.lower().split()
    if len(claim_words) >= 4:
        for i in range(len(claim_words) - 3):
            phrase = " ".join(claim_words[i:i+4])
            if phrase in snippet.lower():
                phrase_match = 0.8
                break

    # Important terms (numbers, medical terms)
    important_patterns = [
        r'\b\d+(?:\.\d+)?%?\b',  # Numbers and percentages
        r'\b(cure|treat|vaccine|disease|study|research|evidence|death|cases|risk)\b',
    ]
    
    claim_important = set()
    snippet_important = set()
    
    for pattern in important_patterns:
        claim_important.update(re.findall(pattern, claim.lower()))
        snippet_important.update(re.findall(pattern, snippet.lower()))
    
    important_overlap = 0.0
    if claim_important:
        important_intersection = len(claim_important & snippet_important)
        important_overlap = important_intersection / len(claim_important)
    
    # Combined relevance score
    relevance = (
        jaccard * 0.2 +
        coverage * 0.2 +
        fuzzy_score * 0.2 +
        phrase_match * 0.2 +
        important_overlap * 0.2
    )
    
    return min(1.0, relevance)


async def _pubmed_evidence_for_claim(claim: str) -> dict:
    """Fetch PubMed citations that may support or refute a claim."""
    # Query: use claim text directly; in a full system we'd construct fielded queries.
    out: dict = {
        "citations": [],
        "query_trace": [],
        "quality_flags": [],
        "uncertainty_flags": [],
    }
    term = claim
    try:
        sr = await pubmed_esearch(term=term)
        
        # Check if the API returned an error
        if "error" in sr:
            out["uncertainty_flags"].append("ncbi_unavailable")
            out["query_trace"].append({"provider": "ncbi", "error": sr["error"]})
            return out
        
        pmids = (sr.get("data") or {}).get("esearchresult", {}).get("idlist", [])
        out["query_trace"].append(
            {
                "provider": "ncbi",
                "db": "pubmed",
                "term": term,
                "pmids": pmids[:PUBMED_MAX_ABSTRACTS],
            }
        )
        if not pmids:
            return out
        pmids = pmids[:PUBMED_MAX_ABSTRACTS]
        summ = await pubmed_esummary(pmids)
        
        # Check if esummary returned an error
        if "error" in summ:
            out["uncertainty_flags"].append("ncbi_esummary_failed")
            out["query_trace"].append({"provider": "ncbi", "error": summ["error"]})
            return out
        
        sum_data = (summ.get("data") or {}).get("result", {})
        for pmid in pmids:
            item = sum_data.get(str(pmid), {})
            title = item.get("title")
            journal = item.get("fulljournalname") or item.get("source")
            pubdate = item.get("pubdate")
            pubtypes = item.get("pubtype") or []
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            snippets: list[str] = []
            # POC: fetch abstracts per PMID to avoid mis-associating text.
            try:
                abs_one = await pubmed_efetch_abstract([str(pmid)])
                
                # Check if efetch returned an error
                if "error" in abs_one:
                    out["uncertainty_flags"].append("pubmed_abstract_fetch_failed")
                else:
                    out["query_trace"].append(
                        {
                            "provider": "ncbi",
                            "db": "pubmed",
                            "pmid": str(pmid),
                            "efetch_url": abs_one["request"].get("url", ""),
                        }
                    )
                    abs_text = (abs_one.get("data") or "").strip()
                    lines = [ln.strip() for ln in abs_text.split("\n") if ln.strip()]
                    for ln in lines:
                        if _snippet_relevance(ln, claim) >= 0.25:
                            snippets.append(ln)
                        if len(snippets) >= 3:
                            break
            except Exception:
                out["uncertainty_flags"].append("pubmed_abstract_fetch_failed")
            out["citations"].append(
                {
                    "pmid": str(pmid),
                    "title": title,
                    "journal": journal,
                    "pubdate": pubdate,
                    "pubtypes": pubtypes,
                    "url": url,
                    "snippets": snippets[:3],
                }
            )
    except Exception as e:
        out["uncertainty_flags"].append("ncbi_unavailable")
        out["query_trace"].append({"provider": "ncbi", "error": str(e)})
    return out


async def _gdelt_evidence_for_claim(claim: str) -> dict:
    """Fetch GDELT news corroboration for a claim."""
    out: dict = {"neighbors": [], "query_trace": [], "uncertainty_flags": []}
    try:
        r = await gdelt_doc_search(query=claim, maxrecords=10, retries=2)
        
        # Check if the API returned an error
        if "error" in r:
            error_type = r["error"]
            # Only flag as unavailable for persistent errors, not temporary ones
            if "timeout" in error_type or "connection_failed" in error_type or "http_error_5" in error_type:
                out["uncertainty_flags"].append("gdelt_unavailable")
            out["query_trace"].append({"provider": "gdelt", "error": error_type, "retried": True})
            return out
        
        # Handle both possible response formats
        data = r.get("data") or {}
        articles = data.get("articles") or []
        
        # If no articles in 'articles', check if data itself is a list
        if not articles and isinstance(data, list):
            articles = data
        
        out["query_trace"].append(
            {"provider": "gdelt", "url": r.get("request", {}).get("url", ""), "count": len(articles)}
        )
        
        for a in articles[:5]:
            if isinstance(a, dict):
                out["neighbors"].append(
                    {
                        "url": a.get("url"),
                        "title": a.get("title"),
                        "domain": a.get("domain"),
                        "seendate": a.get("seendate"),
                    }
                )
    except Exception as e:
        out["uncertainty_flags"].append("gdelt_unavailable")
        out["query_trace"].append({"provider": "gdelt", "error": str(e), "exception_type": type(e).__name__})
    return out


async def _tavily_evidence_for_claim(claim: str) -> dict:
    """Fetch Tavily web search corroboration for a claim."""
    out: dict = {"neighbors": [], "query_trace": [], "uncertainty_flags": []}
    if not TAVILY_API_KEY:
        return out
    try:
        r = await tavily_search(query=claim)
        
        # Check if the API returned an error
        if "error" in r:
            out["uncertainty_flags"].append("tavily_unavailable")
            out["query_trace"].append({"provider": "tavily", "error": r["error"]})
            return out
        
        results = (r.get("data") or {}).get("results") or []
        out["query_trace"].append(
            {
                "provider": "tavily",
                "endpoint": r["request"].get("endpoint", ""),
                "count": len(results),
            }
        )
        for item in results[:5]:
            out["neighbors"].append(
                {
                    "url": item.get("url"),
                    "title": item.get("title"),
                    "content": item.get("content"),
                    "score": item.get("score"),
                }
            )
    except Exception as e:
        out["uncertainty_flags"].append("tavily_unavailable")
        out["query_trace"].append({"provider": "tavily", "error": str(e)})
    return out


def _deduplicate_claims(candidates: list[str], threshold: float = 0.85) -> list[str]:
    """Remove duplicate or near-duplicate claims from candidates.
    
    Uses fuzzy string matching to identify and remove duplicate claims
    that are extremely similar. Keeps only the first occurrence.
    
    Args:
        candidates: List of claim candidate strings
        threshold: Similarity threshold (0-1) for considering claims duplicates
        
    Returns:
        Deduplicated list of claims
    """
    if not candidates:
        return candidates
    
    deduped = []
    seen_indices = set()
    
    for i, claim in enumerate(candidates):
        if i in seen_indices:
            continue
        
        deduped.append(claim)
        
        # Mark similar claims as duplicates
        for j in range(i + 1, len(candidates)):
            if j not in seen_indices:
                similarity = fuzz.ratio(claim.lower(), candidates[j].lower()) / 100.0
                if similarity >= threshold:
                    logger.debug(f"Deduplicating claim (similarity={similarity:.2f}): '{candidates[j][:50]}...'")
                    seen_indices.add(j)
    
    return deduped


async def analyze_claims(doc) -> ClaimsOutput:
    """
    Analyze document for factual claims using LLM (if available) or heuristics.
    
    Args:
        doc: Document object with sentences and display_text
        
    Returns:
        ClaimsOutput with extracted and analyzed claims
    """
    triggers = _medical_topic_triggers(doc.display_text)
    medical = bool(triggers)
    # Aggregate provider uncertainty across claim items for easier UI surfacing.
    uncertainty_flags: list[str] = []
    llm_risk_indicators: list[str] = []

    # Try LLM-based claim extraction first (more intelligent)
    llm_claims, llm_risk_indicators = await _claim_candidates_llm(doc)
    
    if llm_claims:
        # Use LLM-extracted claims
        candidates = []
        llm_claim_metadata = {}  # Store LLM metadata for each claim
        
        for llm_claim in llm_claims:
            claim_text = llm_claim.get("text", "").strip()
            if claim_text and len(claim_text) >= 20:
                candidates.append(claim_text)
                llm_claim_metadata[claim_text] = {
                    "type": llm_claim.get("type", "factual"),
                    "verifiable": llm_claim.get("verifiable", True),
                    "topics": llm_claim.get("topics", []),
                    "confidence": llm_claim.get("confidence", 0.5),
                }
        
        logger.info(f"Using {len(candidates)} LLM-extracted claims")
        
        # Add LLM risk indicators to uncertainty flags
        if llm_risk_indicators:
            for indicator in llm_risk_indicators:
                if indicator and f"llm_risk:{indicator}" not in uncertainty_flags:
                    uncertainty_flags.append(f"llm_risk:{indicator}")
    else:
        # Fallback to heuristic claim extraction
        candidates = _claim_candidates([s.text for s in doc.sentences])
        llm_claim_metadata = {}
        logger.info(f"Using {len(candidates)} heuristic-extracted claims (LLM unavailable)")
    
    # Deduplicate claims before processing
    candidates = _deduplicate_claims(candidates, threshold=0.85)
    claim_items: list[ClaimItem] = []
    supported = 0
    unsupported = 0
    unverifiable = 0

    for idx, c in enumerate(candidates):
        cid = f"C{idx + 1}"
        
        # Use LLM type if available, otherwise use heuristic
        if c in llm_claim_metadata:
            llm_meta = llm_claim_metadata[c]
            llm_type = llm_meta.get("type", "factual")
            # Map LLM type to our types
            type_map = {
                "factual": "factual",
                "speculative": "speculative", 
                "predictive": "predictive",
                "opinion": "opinion_presented_as_fact",
            }
            ctype = type_map.get(llm_type, "factual")
        else:
            ctype = _claim_type(c)
        
        pointers = Pointer(
            char_spans=find_spans(doc.display_text, c[: min(len(c), 80)], max_spans=1)
        )

        reasons: list[str] = []
        support = "unverifiable"
        citations: list[dict] = []
        query_trace: list[dict] = []
        qflags: list[str] = []
        uflags: list[str] = []

        # Basic unsupported assertion heuristic (internal, not external fact checking).
        strong_claim_wo_attr = (
            bool(_STRONG_MED_CLAIM.search(c)) and not doc.attributions
        )
        if strong_claim_wo_attr:
            reasons.append("strong_claim_without_attribution")

        # Online evidence - collect from all sources
        pubmed_citations = []
        gdelt_citations = []
        tavily_citations = []
        
        # Medical evidence (PubMed)
        if medical:
            pub = await _pubmed_evidence_for_claim(c)
            pubmed_citations = pub.get("citations", [])
            citations.extend(pubmed_citations)
            query_trace.extend(pub.get("query_trace", []))
            qflags.extend(pub.get("quality_flags", []))
            uflags.extend(pub.get("uncertainty_flags", []))
            # Mark PubMed citations with provider
            for cit in pubmed_citations:
                cit["provider"] = "pubmed"

        # News corroboration (GDELT) as a general (non-medical) corroboration hint.
        gd = await _gdelt_evidence_for_claim(c)
        if gd.get("neighbors"):
            query_trace.extend(gd.get("query_trace", []))
            gdelt_citations = [{**n, "provider": "gdelt"} for n in gd.get("neighbors", [])]
            citations.extend(gdelt_citations)
            reasons.append("related_news_coverage_exists")
        if gd.get("uncertainty_flags"):
            uflags.extend(gd.get("uncertainty_flags", []))

        # Optional web search corroboration (Tavily).
        tv = await _tavily_evidence_for_claim(c)
        if tv.get("neighbors"):
            query_trace.extend(tv.get("query_trace", []))
            tavily_citations = [{**n, "provider": "tavily"} for n in tv.get("neighbors", [])]
            citations.extend(tavily_citations)
            reasons.append("related_web_results_exist")
        if tv.get("uncertainty_flags"):
            uflags.extend(tv.get("uncertainty_flags", []))

        for f in uflags:
            if f not in uncertainty_flags:
                uncertainty_flags.append(f)

        # Check if ALL external APIs failed - apply fallback behavior
        all_apis_failed = (
            "ncbi_unavailable" in uflags and
            "gdelt_unavailable" in uflags and
            ("tavily_unavailable" in uflags or not TAVILY_API_KEY)
        )
        
        if all_apis_failed:
            # When all APIs fail, mark as unverifiable and add special flags
            if "all_external_apis_unavailable" not in uflags:
                uflags.append("all_external_apis_unavailable")
            if "all_external_apis_unavailable" not in uncertainty_flags:
                uncertainty_flags.append("all_external_apis_unavailable")
            # Force support to unverifiable since we cannot verify claims
            support = "unverifiable"
            if "no_external_verification_available" not in reasons:
                reasons.append("no_external_verification_available")
            quality_flags_to_add = ["degraded_verification"]
            for qf in quality_flags_to_add:
                if qf not in qflags:
                    qflags.append(qf)
        else:
            # ENHANCED CROSS-REFERENCING LOGIC WITH EVIDENCE STRENGTH ANALYSIS
            # Assess source quality and authority for all citations
            quality_assessments = []
            reputable_count = 0
            questionable_count = 0
            high_authority_count = 0
            
            for citation in citations:
                assessment = assess_citation_quality(citation)
                authority = assess_source_authority(citation)
                quality_assessments.append(assessment)
                citation["quality_assessment"] = assessment
                citation["authority_score"] = authority
                
                if assessment["reputation_tier"] == "reputable":
                    reputable_count += 1
                elif assessment["reputation_tier"] == "questionable":
                    questionable_count += 1
                
                if authority >= 0.8:  # High authority (peer-reviewed, .gov, .edu, reputable news)
                    high_authority_count += 1
            
            # Calculate comprehensive evidence strength
            evidence_strength = calculate_evidence_strength(citations, c, medical)
            citation["evidence_strength"] = evidence_strength
            
            # Detect contradictions
            contradictions = detect_contradictions(citations, c)
            if contradictions["contradiction_score"] > 0.3:
                uflags.append("potential_contradictions_detected")
                reasons.append(f"contradicting_sources_found_{contradictions['contradicting_count']}")
            
            # Calculate source diversity
            diversity_metrics = calculate_source_diversity(citations)
            if diversity_metrics["diversity_score"] > 0:
                qflags.append(f"source_diversity_score_{diversity_metrics['diversity_score']:.2f}")
            
            # Multi-source agreement analysis
            providers_found = set()
            if pubmed_citations:
                providers_found.add("pubmed")
            if gdelt_citations:
                providers_found.add("gdelt")
            if tavily_citations:
                providers_found.add("tavily")
            
            multi_provider_agreement = len(providers_found) >= 2
            high_diversity = diversity_metrics["diversity_score"] >= 0.6
            
            # Enhanced support classification using evidence strength
            strength_score = evidence_strength["strength_score"]
            strength_tier = evidence_strength["strength_tier"]
            
            # Count high-quality supporting evidence with improved snippet relevance
            snippetful = []
            high_quality_snippetful = []
            high_authority_snippetful = []
            
            for citation in citations:
                snippets = citation.get("snippets", [])
                if snippets:
                    # Calculate relevance for each snippet
                    relevant_snippets = [
                        s for s in snippets 
                        if _snippet_relevance(s, c) >= 0.25
                    ]
                    if relevant_snippets:
                        snippetful.append(citation)
                        # Check quality and authority
                        quality_tier = citation.get("quality_assessment", {}).get("reputation_tier", "neutral")
                        authority = citation.get("authority_score", 0.0)
                        
                        if quality_tier == "reputable":
                            high_quality_snippetful.append(citation)
                        if authority >= 0.8:
                            high_authority_snippetful.append(citation)
            
            # SUPPORTED: Use evidence strength as primary indicator
            if strength_tier in ["very_strong", "strong"]:
                support = "supported"
                reasons.append(f"evidence_strength_{strength_tier}")
                
                # Add specific reasons
                if len(high_authority_snippetful) >= 2:
                    reasons.append("multiple_high_authority_sources_with_snippets")
                elif len(high_quality_snippetful) >= 2:
                    reasons.append("multiple_reputable_sources_with_snippets")
                elif len(snippetful) >= 2:
                    reasons.append("multiple_sources_with_snippets")
                
                if multi_provider_agreement:
                    reasons.append("cross_provider_validation")
                if high_diversity:
                    reasons.append("diverse_source_agreement")
                if high_authority_count >= 1:
                    reasons.append("high_authority_sources_present")
                    
            elif strength_tier == "moderate":
                # Moderate evidence - check for additional factors
                if len(snippetful) >= 2 and reputable_count >= 1:
                    support = "supported"
                    reasons.append("moderate_evidence_multiple_sources")
                    if multi_provider_agreement:
                        reasons.append("cross_provider_validation")
                elif citations and multi_provider_agreement and reputable_count >= 1:
                    support = "supported"
                    reasons.append("moderate_evidence_cross_validated")
                else:
                    support = "unverifiable"
                    reasons.append("moderate_evidence_insufficient_validation")
                    
            elif citations:
                # UNVERIFIABLE: Some evidence but weak
                support = "unverifiable"
                if snippetful:
                    reasons.append("sources_found_but_insufficient_snippets")
                elif contradictions["contradiction_score"] > 0.3:
                    support = "contested"
                    reasons.append("contradicting_evidence_found")
                else:
                    reasons.append("sources_found_but_no_relevant_snippets")
            else:
                # UNSUPPORTED: No evidence found, especially for strong claims
                if strong_claim_wo_attr:
                    support = "unsupported"
                    reasons.append("strong_claim_no_attribution_no_evidence")
                else:
                    support = "unverifiable"
                    reasons.append("no_external_evidence_found")
            
            # Handle contradictions - downgrade support if significant contradictions
            if contradictions["contradiction_score"] > 0.5:
                if support == "supported":
                    support = "contested"
                    reasons.append("support_downgraded_due_to_contradictions")
                elif support != "contested":
                    support = "unverifiable"
                    reasons.append("contradicting_evidence_detected")
            
            # Flag questionable sources
            if questionable_count > 0:
                qflags.append("questionable_sources_present")
                if questionable_count >= reputable_count:
                    reasons.append("more_questionable_than_reputable_sources")
                    # Downgrade support if questionable sources dominate
                    if support == "supported" and reputable_count < 2:
                        support = "unverifiable"
                        reasons.append("support_downgraded_due_to_questionable_sources")
            
            # Medical claims require stricter validation but allow high-authority news
            if medical and support == "supported":
                # Check for academic/peer-reviewed sources
                has_academic = high_authority_count >= 1 and any(
                    cit.get("provider") == "pubmed" or 
                    any(pt in str(cit.get("pubtypes", "")).lower() for pt in ["journal article", "clinical trial"])
                    for cit in citations
                )
                
                # Check for high-authority news outlets if no academic sources
                has_high_auth_news = any(
                    cit.get("authority_score", 0.0) >= 0.8 and cit.get("provider") != "pubmed"
                    for cit in citations
                )
                
                if not has_academic and has_high_auth_news:
                    # Allow high-authority news but with a warning/lower confidence
                    reasons.append("supported_by_high_authority_news_only")
                    # Note: We keep support="supported" but reasoning engine will see the reason
                elif not has_academic and not has_high_auth_news:
                    support = "unverifiable"
                    reasons.append("medical_claim_requires_authoritative_sources")
                elif len(snippetful) < 1:
                    # Medical claims MUST have at least one snippet
                    support = "unverifiable"
                    reasons.append("medical_claim_requires_supporting_snippets")

        # LLM Fallback veracity check if external APIs were unavailable or inconclusive
        # AND it's a verifiable claim
        is_verifiable = True
        if c in llm_claim_metadata:
            is_verifiable = llm_claim_metadata[c].get("verifiable", True)
            
        if support == "unverifiable" and is_verifiable and is_groq_available():
            llm_fallback = await _llm_veracity_fallback(c)
            if llm_fallback.get("support") in ["supported", "unsupported"]:
                # We don't fully trust the LLM, so we use a "likely" verdict
                # and don't change the official support status to "supported"
                # instead, we add it to reasons and flags
                reasons.append(f"llm_fallback_indicates_{llm_fallback['support']}")
                uflags.append(f"llm_veracity_estimate_{llm_fallback['support']}")
                if "llm_fallback_used" not in query_trace:
                    query_trace.append({
                        "provider": "llm_fallback",
                        "estimated_support": llm_fallback["support"],
                        "confidence": llm_fallback.get("confidence", 0.0),
                        "reason": llm_fallback.get("reason", "")
                    })
            
            # Add quality flags based on assessments
            if reputable_count >= 2:
                qflags.append("multiple_reputable_sources")
            if high_authority_count >= 1:
                qflags.append("high_authority_sources")
            if high_diversity:
                qflags.append("high_source_diversity")
            if multi_provider_agreement:
                qflags.append("cross_provider_validation")
            if strength_tier in ["very_strong", "strong"]:
                qflags.append(f"evidence_strength_{strength_tier}")

        if support == "supported":
            supported += 1
        elif support == "unsupported":
            unsupported += 1
        elif support == "contested":
            # Contested claims are counted separately but also affect unverifiable
            unverifiable += 1  # Count as unverifiable for backward compatibility
        else:
            unverifiable += 1

        claim_items.append(
            ClaimItem(
                id=cid,
                text=c,
                type=ctype,
                support=support,  # type: ignore[arg-type]
                reasons=reasons,
                pointers=pointers,
                citations=citations,
                query_trace=query_trace,
                quality_flags=qflags,
                uncertainty_flags=uflags,
            )
        )

    # Count contested claims separately
    contested = sum(1 for c in claim_items if c.support == "contested")
    
    return ClaimsOutput(
        claims={
            "supported": supported,
            "unsupported": unsupported,
            "contested": contested,
            "unverifiable": unverifiable,
        },
        claim_items=claim_items,
        medical_topic_detected=medical,
        medical_topic_triggers=triggers,
        uncertainty_flags=uncertainty_flags,
    )
