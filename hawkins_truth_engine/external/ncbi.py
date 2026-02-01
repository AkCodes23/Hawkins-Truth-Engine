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
    rm = PUBMED_RETMAX if retmax is None else retmax
    params = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": str(rm),
        **_base_params(),
    }
    timeout = httpx.Timeout(HTTP_TIMEOUT_SECS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params
        )
        r.raise_for_status()
        return {"request": {"url": str(r.url)}, "data": r.json()}


async def pubmed_esummary(pmids: list[str]) -> dict[str, Any]:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "json",
        **_base_params(),
    }
    timeout = httpx.Timeout(HTTP_TIMEOUT_SECS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params=params
        )
        r.raise_for_status()
        return {"request": {"url": str(r.url)}, "data": r.json()}


async def pubmed_efetch_abstract(pmids: list[str]) -> dict[str, Any]:
    # Text abstracts; easier for POC snippet extraction.
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "text",
        **_base_params(),
    }
    timeout = httpx.Timeout(HTTP_TIMEOUT_SECS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=params
        )
        r.raise_for_status()
        return {"request": {"url": str(r.url)}, "data": r.text}
