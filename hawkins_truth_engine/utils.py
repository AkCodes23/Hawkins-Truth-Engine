from __future__ import annotations

import re
from urllib.parse import urlparse

from .schemas import CharSpan


_WS_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = _WS_RE.sub(" ", text).strip()
    return text


def safe_domain(url: str) -> str | None:
    try:
        u = urlparse(url)
        if not u.netloc:
            return None
        return u.netloc.lower()
    except Exception:
        return None


def find_spans(text: str, needle: str, max_spans: int = 5) -> list[CharSpan]:
    if not needle:
        return []
    spans: list[CharSpan] = []
    start = 0
    n = needle.lower()
    t = text.lower()
    while len(spans) < max_spans:
        idx = t.find(n, start)
        if idx < 0:
            break
        spans.append(CharSpan(start=idx, end=idx + len(needle)))
        start = idx + len(needle)
    return spans
