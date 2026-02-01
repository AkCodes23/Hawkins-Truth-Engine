from __future__ import annotations

from typing import Any

import httpx

from ..config import GDELT_MAXRECORDS, HTTP_TIMEOUT_SECS


async def gdelt_doc_search(query: str, maxrecords: int | None = None) -> dict[str, Any]:
    # DOC 2.1: https://api.gdeltproject.org/api/v2/doc/doc
    mr = GDELT_MAXRECORDS if maxrecords is None else maxrecords
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": str(mr),
        "sort": "hybridrel",
    }
    timeout = httpx.Timeout(HTTP_TIMEOUT_SECS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
        r.raise_for_status()
        return {"request": {"url": str(r.url)}, "data": r.json()}
