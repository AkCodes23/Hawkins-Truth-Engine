from __future__ import annotations

from typing import Any

import httpx

from ..config import GDELT_MAXRECORDS, HTTP_TIMEOUT_SECS


async def gdelt_doc_search(query: str, maxrecords: int | None = None) -> dict[str, Any]:
    """
    Search GDELT for news articles matching the query.
    
    Args:
        query: Search query string
        maxrecords: Maximum number of records to return
        
    Returns:
        Dict with 'request' and 'data' keys, or 'error' key on failure
    """
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
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
            r.raise_for_status()
            return {"request": {"url": str(r.url)}, "data": r.json()}
    except httpx.TimeoutException:
        return {"request": {"url": "https://api.gdeltproject.org/api/v2/doc/doc"}, "error": "timeout", "data": {"articles": []}}
    except httpx.HTTPStatusError as e:
        return {"request": {"url": str(e.request.url)}, "error": f"http_error_{e.response.status_code}", "data": {"articles": []}}
    except httpx.ConnectError as e:
        return {"request": {"url": "https://api.gdeltproject.org/api/v2/doc/doc"}, "error": f"connection_failed: {e}", "data": {"articles": []}}
    except Exception as e:
        return {"request": {"url": "https://api.gdeltproject.org/api/v2/doc/doc"}, "error": f"gdelt_error: {type(e).__name__}: {e}", "data": {"articles": []}}
