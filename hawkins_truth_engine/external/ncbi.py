from __future__ import annotations

from typing import Any

import httpx

from ..config import (
    HTTP_TIMEOUT_SECS,
    NCBI_API_KEY,
    NCBI_EMAIL,
    NCBI_TOOL,
    PUBMED_RETMAX,
)


def _base_params() -> dict[str, str]:
    p = {"tool": NCBI_TOOL}
    if NCBI_EMAIL:
        p["email"] = NCBI_EMAIL
    if NCBI_API_KEY:
        p["api_key"] = NCBI_API_KEY
    return p


async def pubmed_esearch(term: str, retmax: int | None = None) -> dict[str, Any]:
    """
    Search PubMed for articles matching the search term.
    
    Args:
        term: Search term/query
        retmax: Maximum number of results to return
        
    Returns:
        Dict with 'request' and 'data' keys, or 'error' key on failure
    """
    rm = PUBMED_RETMAX if retmax is None else retmax
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": str(rm),
        **_base_params(),
    }
    timeout = httpx.Timeout(HTTP_TIMEOUT_SECS)
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return {"request": {"url": str(r.url)}, "data": r.json()}
    except httpx.TimeoutException:
        return {"request": {"url": url}, "error": "timeout", "data": {"esearchresult": {"idlist": []}}}
    except httpx.HTTPStatusError as e:
        return {"request": {"url": str(e.request.url)}, "error": f"http_error_{e.response.status_code}", "data": {"esearchresult": {"idlist": []}}}
    except httpx.ConnectError as e:
        return {"request": {"url": url}, "error": f"connection_failed: {e}", "data": {"esearchresult": {"idlist": []}}}
    except Exception as e:
        return {"request": {"url": url}, "error": f"ncbi_error: {type(e).__name__}: {e}", "data": {"esearchresult": {"idlist": []}}}


async def pubmed_esummary(pmids: list[str]) -> dict[str, Any]:
    """
    Get summary information for PubMed article IDs.
    
    Args:
        pmids: List of PubMed IDs to look up
        
    Returns:
        Dict with 'request' and 'data' keys, or 'error' key on failure
    """
    if not pmids:
        return {"request": {"url": ""}, "data": {"result": {}}}
        
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
        **_base_params(),
    }
    timeout = httpx.Timeout(HTTP_TIMEOUT_SECS)
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return {"request": {"url": str(r.url)}, "data": r.json()}
    except httpx.TimeoutException:
        return {"request": {"url": url}, "error": "timeout", "data": {"result": {}}}
    except httpx.HTTPStatusError as e:
        return {"request": {"url": str(e.request.url)}, "error": f"http_error_{e.response.status_code}", "data": {"result": {}}}
    except httpx.ConnectError as e:
        return {"request": {"url": url}, "error": f"connection_failed: {e}", "data": {"result": {}}}
    except Exception as e:
        return {"request": {"url": url}, "error": f"ncbi_error: {type(e).__name__}: {e}", "data": {"result": {}}}


async def pubmed_efetch_abstract(pmids: list[str]) -> dict[str, Any]:
    """
    Fetch full abstracts for PubMed article IDs.
    
    Args:
        pmids: List of PubMed IDs to fetch abstracts for
        
    Returns:
        Dict with 'request' and 'data' keys, or 'error' key on failure
    """
    if not pmids:
        return {"request": {"url": ""}, "data": ""}
        
    # Text abstracts; easier for POC snippet extraction.
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "text",
        **_base_params(),
    }
    timeout = httpx.Timeout(HTTP_TIMEOUT_SECS)
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return {"request": {"url": str(r.url)}, "data": r.text}
    except httpx.TimeoutException:
        return {"request": {"url": url}, "error": "timeout", "data": ""}
    except httpx.HTTPStatusError as e:
        return {"request": {"url": str(e.request.url)}, "error": f"http_error_{e.response.status_code}", "data": ""}
    except httpx.ConnectError as e:
        return {"request": {"url": url}, "error": f"connection_failed: {e}", "data": ""}
    except Exception as e:
        return {"request": {"url": url}, "error": f"ncbi_error: {type(e).__name__}: {e}", "data": ""}
