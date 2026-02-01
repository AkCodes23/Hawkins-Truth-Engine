from __future__ import annotations

from typing import Any

import httpx

from ..config import GDELT_MAXRECORDS, HTTP_TIMEOUT_SECS


async def gdelt_doc_search(query: str, maxrecords: int | None = None, retries: int = 2) -> dict[str, Any]:
    """
    Search GDELT for news articles matching the query.
    
    Args:
        query: Search query string
        maxrecords: Maximum number of records to return
        retries: Number of retry attempts (default: 2)
        
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
    
    last_error = None
    for attempt in range(retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                r = await client.get("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
                r.raise_for_status()
                json_data = r.json()
                # Validate response structure
                if isinstance(json_data, dict) and ("articles" in json_data or "data" in json_data):
                    return {"request": {"url": str(r.url)}, "data": json_data}
                else:
                    # Unexpected response format
                    return {"request": {"url": str(r.url)}, "error": "invalid_response_format", "data": {"articles": []}}
        except httpx.TimeoutException:
            last_error = "timeout"
            if attempt < retries:
                continue
            return {"request": {"url": "https://api.gdeltproject.org/api/v2/doc/doc"}, "error": "timeout", "data": {"articles": []}}
        except httpx.HTTPStatusError as e:
            last_error = f"http_error_{e.response.status_code}"
            # Don't retry on client errors (4xx), but retry on server errors (5xx)
            if e.response.status_code < 500 or attempt >= retries:
                return {"request": {"url": str(e.request.url)}, "error": last_error, "data": {"articles": []}}
            continue
        except httpx.ConnectError as e:
            last_error = f"connection_failed: {str(e)}"
            if attempt < retries:
                continue
            return {"request": {"url": "https://api.gdeltproject.org/api/v2/doc/doc"}, "error": last_error, "data": {"articles": []}}
        except Exception as e:
            last_error = f"gdelt_error: {type(e).__name__}: {str(e)}"
            if attempt < retries:
                continue
            return {"request": {"url": "https://api.gdeltproject.org/api/v2/doc/doc"}, "error": last_error, "data": {"articles": []}}
    
    # Fallback if all retries exhausted
    return {"request": {"url": "https://api.gdeltproject.org/api/v2/doc/doc"}, "error": last_error or "unknown_error", "data": {"articles": []}}
