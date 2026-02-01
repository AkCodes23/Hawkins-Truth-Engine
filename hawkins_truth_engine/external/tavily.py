from __future__ import annotations

from typing import Any, Literal

import httpx

from ..config import HTTP_TIMEOUT_SECS, TAVILY_API_KEY, TAVILY_MAX_RESULTS, TAVILY_SEARCH_DEPTH


SearchDepth = Literal["basic", "advanced"]


async def tavily_search(
    query: str,
    *,
    max_results: int | None = None,
    search_depth: SearchDepth | None = None,
) -> dict[str, Any]:
    """Search the public web via Tavily.

    This is intentionally optional: if `TAVILY_API_KEY` is not set, callers should
    skip invoking this provider.

    API docs: https://docs.tavily.com/
    """

    if not TAVILY_API_KEY:
        raise RuntimeError("Tavily API key not configured")

    mr = TAVILY_MAX_RESULTS if max_results is None else max_results
    sd: SearchDepth = (
        TAVILY_SEARCH_DEPTH if search_depth is None else search_depth  # type: ignore[assignment]
    )

    payload = {
        "api_key": TAVILY_API_KEY,
        "query": query,
        "max_results": int(mr),
        "search_depth": sd,
        "include_answer": False,
        "include_raw_content": False,
    }

    timeout = httpx.Timeout(HTTP_TIMEOUT_SECS)
    url = "https://api.tavily.com/search"
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()

        return {
            "request": {
                "endpoint": url,
                "max_results": int(mr),
                "search_depth": sd,
            },
            "data": data,
        }
    except httpx.TimeoutException:
        return {"request": {"endpoint": url}, "error": "timeout", "data": {"results": []}}
    except httpx.HTTPStatusError as e:
        return {"request": {"endpoint": url}, "error": f"http_error_{e.response.status_code}", "data": {"results": []}}
    except httpx.ConnectError as e:
        return {"request": {"endpoint": url}, "error": f"connection_failed: {e}", "data": {"results": []}}
    except Exception as e:
        return {"request": {"endpoint": url}, "error": f"tavily_error: {type(e).__name__}: {e}", "data": {"results": []}}
