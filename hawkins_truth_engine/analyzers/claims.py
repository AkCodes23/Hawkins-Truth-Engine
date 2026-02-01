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


def _snippet_relevance(snippet: str, claim: str) -> float:
    # POC lexical overlap.
    cs = {w for w in re.findall(r"[a-z0-9]{4,}", claim.lower())}
    ss = {w for w in re.findall(r"[a-z0-9]{4,}", snippet.lower())}
    if not cs or not ss:
        return 0.0
    return len(cs & ss) / len(cs)


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
        r = await gdelt_doc_search(query=claim, maxrecords=10)
        
        # Check if the API returned an error
        if "error" in r:
            out["uncertainty_flags"].append("gdelt_unavailable")
            out["query_trace"].append({"provider": "gdelt", "error": r["error"]})
            return out
        
        articles = (r.get("data") or {}).get("articles") or []
        out["query_trace"].append(
            {"provider": "gdelt", "url": r["request"].get("url", ""), "count": len(articles)}
        )
        for a in articles[:5]:
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
        out["query_trace"].append({"provider": "gdelt", "error": str(e)})
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

        # Online evidence
        if medical:
            pub = await _pubmed_evidence_for_claim(c)
            citations.extend(pub.get("citations", []))
            query_trace.extend(pub.get("query_trace", []))
            qflags.extend(pub.get("quality_flags", []))
            uflags.extend(pub.get("uncertainty_flags", []))

            # Very conservative classification: "supported" only if we have at least 2 citations with snippets.
            snippetful = [x for x in citations if x.get("snippets")]
            if len(snippetful) >= 2:
                support = "supported"
                reasons.append("multiple_pubmed_snippets")
            elif citations:
                support = "unverifiable"
                reasons.append("pubmed_hits_but_no_clear_snippets")
            else:
                # If the claim is framed as strong medical efficacy/safety and lacks attribution,
                # treat it as an unsupported assertion (not a claim of falsity).
                support = "unsupported" if strong_claim_wo_attr else "unverifiable"
                reasons.append("no_pubmed_hits")

        # News corroboration (GDELT) as a general (non-medical) corroboration hint.
        gd = await _gdelt_evidence_for_claim(c)
        if gd.get("neighbors"):
            query_trace.extend(gd.get("query_trace", []))
            citations.extend(
                [{**n, "provider": "gdelt"} for n in gd.get("neighbors", [])]
            )
            reasons.append("related_news_coverage_exists")
        if gd.get("uncertainty_flags"):
            uflags.extend(gd.get("uncertainty_flags", []))

        # Optional web search corroboration (Tavily).
        tv = await _tavily_evidence_for_claim(c)
        if tv.get("neighbors"):
            query_trace.extend(tv.get("query_trace", []))
            citations.extend(
                [{**n, "provider": "tavily"} for n in tv.get("neighbors", [])]
            )
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

        if support == "supported":
            supported += 1
        elif support == "unsupported":
            unsupported += 1
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

    return ClaimsOutput(
        claims={
            "supported": supported,
            "unsupported": unsupported,
            "unverifiable": unverifiable,
        },
        claim_items=claim_items,
        medical_topic_detected=medical,
        medical_topic_triggers=triggers,
        uncertainty_flags=uncertainty_flags,
    )
