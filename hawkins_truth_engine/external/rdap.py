from __future__ import annotations

from typing import Any

import httpx

from ..config import HTTP_TIMEOUT_SECS


async def rdap_domain(domain: str) -> dict[str, Any]:
    """
    Look up domain registration information via RDAP.
    
    Args:
        domain: Domain name to look up (e.g., 'example.com')
        
    Returns:
        Dict with 'request' and 'data' keys, or 'error' key on failure
    """
    timeout = httpx.Timeout(HTTP_TIMEOUT_SECS)
    url = f"https://rdap.org/domain/{domain}"
    
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            r = await client.get(url)
            r.raise_for_status()
            return {"request": {"url": str(r.url)}, "data": r.json()}
    except httpx.TimeoutException:
        return {"request": {"url": url}, "error": "timeout", "data": {}}
    except httpx.HTTPStatusError as e:
        return {"request": {"url": str(e.request.url)}, "error": f"http_error_{e.response.status_code}", "data": {}}
    except httpx.ConnectError as e:
        return {"request": {"url": url}, "error": f"connection_failed: {e}", "data": {}}
    except Exception as e:
        return {"request": {"url": url}, "error": f"rdap_error: {type(e).__name__}: {e}", "data": {}}
