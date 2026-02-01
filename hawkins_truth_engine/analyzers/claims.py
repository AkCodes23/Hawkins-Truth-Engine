from __future__ import annotations

import re
from typing import Literal

from ..config import PUBMED_MAX_ABSTRACTS
from ..external.gdelt import gdelt_doc_search
from ..external.ncbi import pubmed_efetch_abstract, pubmed_esearch, pubmed_esummary
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
                out["query_trace"].append(
                    {
                        "provider": "ncbi",
                        "db": "pubmed",
                        "pmid": str(pmid),
                        "efetch_url": abs_one["request"]["url"],
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
        out["uncertainty_flags"].append(str(e))
    return out


async def _gdelt_evidence_for_claim(claim: str) -> dict:
    out: dict = {"neighbors": [], "query_trace": [], "uncertainty_flags": []}
    try:
        r = await gdelt_doc_search(query=claim, maxrecords=10)
        articles = (r.get("data") or {}).get("articles") or []
        out["query_trace"].append(
            {"provider": "gdelt", "url": r["request"]["url"], "count": len(articles)}
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
        out["uncertainty_flags"].append(str(e))
    return out


async def analyze_claims(doc) -> ClaimsOutput:
    triggers = _medical_topic_triggers(doc.display_text)
    medical = bool(triggers)
    uncertainty_flags: list[str] = []

    candidates = _claim_candidates([s.text for s in doc.sentences])
    claim_items: list[ClaimItem] = []
    supported = 0
    unsupported = 0
    unverifiable = 0

    for idx, c in enumerate(candidates):
        cid = f"C{idx + 1}"
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
