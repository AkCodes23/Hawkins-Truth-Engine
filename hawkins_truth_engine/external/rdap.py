from __future__ import annotations

from typing import Any

import httpx

from ..config import HTTP_TIMEOUT_SECS


async def rdap_domain(domain: str) -> dict[str, Any]:
    timeout = httpx.Timeout(HTTP_TIMEOUT_SECS)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.get(f"https://rdap.org/domain/{domain}")
        r.raise_for_status()
        return {"request": {"url": str(r.url)}, "data": r.json()}
