from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from langdetect import detect_langs

from .config import FETCH_MAX_BYTES, HTTP_TIMEOUT_SECS
from .schemas import (
    Attribution,
    CharSpan,
    Document,
    Entity,
    LanguageInfo,
    Sentence,
    Token,
)
from .utils import normalize_text, safe_domain


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _best_effort_language(text: str) -> LanguageInfo:
    dist: list[dict[str, Any]] = []
    top = "unknown"
    try:
        langs = detect_langs(text[:5000])
        for l in langs:
            dist.append({"lang": l.lang, "prob": float(l.prob)})
        if dist:
            top = str(dist[0]["lang"])
    except Exception:
        pass
    return LanguageInfo(top=top, distribution=dist)


def _sentences(text: str) -> list[Sentence]:
    sents: list[Sentence] = []
    if not text:
        return sents
    # Keep approximate spans via incremental search.
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    cursor = 0
    sid = 0
    for p in parts:
        idx = text.find(p, cursor)
        if idx < 0:
            idx = cursor
        span = CharSpan(start=idx, end=min(len(text), idx + len(p)))
        sents.append(Sentence(id=sid, text=p, char_span=span))
        sid += 1
        cursor = span.end
    return sents


def _tokens(text: str) -> list[Token]:
    toks: list[Token] = []
    for m in _WORD_RE.finditer(text):
        toks.append(Token(text=m.group(0), lemma=None, char_span=CharSpan(start=m.start(), end=m.end())))
    return toks


def _entities_best_effort(sentences: list[Sentence]) -> list[Entity]:
    # POC: heuristic entities (Capitalized sequences + simple org suffixes).
    ent_id = 0
    ents: list[Entity] = []
    org_suffix = {"inc", "ltd", "llc", "corp", "company", "co"}
    for s in sentences:
        words = s.text.split()
        i = 0
        while i < len(words):
            w = words[i]
            if len(w) > 1 and w[0].isupper() and w[1:].islower():
                j = i + 1
                while j < len(words):
                    ww = words[j].strip(".,;:!?")
                    if len(ww) > 1 and ww[0].isupper():
                        j += 1
                        continue
                    break
                phrase = " ".join(words[i:j]).strip(".,;:!?")
                if len(phrase.split()) >= 2:
                    t = "PERSON"
                    tail = phrase.split()[-1].lower().strip(".,;:!?")
                    if tail in org_suffix:
                        t = "ORG"
                    # map to doc char span approximately
                    start = s.text.find(phrase)
                    if start >= 0:
                        doc_start = s.char_span.start + start
                        doc_end = doc_start + len(phrase)
                        ents.append(
                            Entity(
                                id=ent_id,
                                text=phrase,
                                type=t,
                                sentence_id=s.id,
                                char_span=CharSpan(start=doc_start, end=doc_end),
                                normalized=None,
                            )
                        )
                        ent_id += 1
                i = j
            else:
                i += 1
    return ents


def _attributions_best_effort(text: str, sentences: list[Sentence]) -> list[Attribution]:
    # POC: detect quoted spans and attribute verbs nearby.
    attrs: list[Attribution] = []
    quote_re = re.compile(r"\"([^\"]{10,300})\"")
    verbs = {"said", "says", "stated", "claim", "claimed", "report", "reported", "according"}
    for s in sentences:
        for m in quote_re.finditer(s.text):
            qstart = s.char_span.start + m.start(0)
            qend = s.char_span.start + m.end(0)
            # pick a verb if present
            verb = "said"
            ctx = s.text[max(0, m.start(0) - 60) : m.start(0)].lower()
            for v in verbs:
                if v in ctx:
                    verb = v
                    break
            attrs.append(Attribution(speaker_entity_id=None, verb=verb, quote_span=CharSpan(start=qstart, end=qend), sentence_id=s.id))
    return attrs


async def fetch_url(url: str) -> dict[str, Any]:
    timeout = httpx.Timeout(HTTP_TIMEOUT_SECS)
    headers = {
        "User-Agent": "HawkinsTruthEnginePOC/0.1 (+https://example.invalid)",
        "Accept": "text/html,application/xhtml+xml",
    }
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        r = await client.get(url)
        content = r.content[:FETCH_MAX_BYTES]
        return {
            "final_url": str(r.url),
            "status_code": r.status_code,
            "headers": dict(r.headers),
            "content": content,
        }


def extract_text_from_html(html_bytes: bytes, url: str | None = None) -> dict[str, Any]:
    html = html_bytes.decode("utf-8", errors="replace")
    extracted = trafilatura.extract(html, include_comments=False, include_tables=False)
    soup = BeautifulSoup(html, "lxml")
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    # Lightweight meta extraction
    author = None
    pub = None
    for meta in soup.find_all("meta"):
        name = (meta.get("name") or meta.get("property") or "").lower()
        if name in {"author"} and meta.get("content"):
            author = meta.get("content").strip()
        if name in {"article:published_time", "pubdate", "publishdate", "date"} and meta.get("content"):
            pub = meta.get("content").strip()
    text = extracted.strip() if extracted else soup.get_text(" ", strip=True)
    text = normalize_text(text)
    return {
        "text": text,
        "title": title,
        "author": author,
        "published_raw": pub,
        "extractor": "trafilatura" if extracted else "bs4_fallback",
    }


def _parse_date_best_effort(s: str | None) -> datetime | None:
    if not s:
        return None
    # POC: best-effort ISO-ish parse
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            continue
    return None


async def build_document(input_type: str, content: str) -> Document:
    retrieved_at = None
    url = None
    domain = None
    title = None
    author = None
    published_at = None
    preprocessing_flags: list[str] = []
    provenance: dict[str, Any] = {}

    if input_type == "url":
        url = content
        domain = safe_domain(url)
        retrieved_at = datetime.now(timezone.utc)
        fetched = await fetch_url(url)
        provenance["fetch"] = {
            "status_code": fetched["status_code"],
            "final_url": fetched["final_url"],
            "retrieved_at": retrieved_at.isoformat(),
        }
        if fetched["status_code"] >= 400:
            preprocessing_flags.append("fetch_error")
        ex = extract_text_from_html(fetched["content"], url=fetched["final_url"])
        display_text = ex["text"]
        title = ex.get("title")
        author = ex.get("author")
        published_at = _parse_date_best_effort(ex.get("published_raw"))
        provenance["extraction"] = {"method": ex.get("extractor"), "title": bool(title), "author": bool(author)}
        if not author:
            preprocessing_flags.append("missing_author")
        if not published_at:
            preprocessing_flags.append("missing_published_at")
    else:
        display_text = normalize_text(content)

    lang = _best_effort_language(display_text)
    sents = _sentences(display_text)
    toks = _tokens(display_text)
    ents = _entities_best_effort(sents)
    attrs = _attributions_best_effort(display_text, sents)

    return Document(
        input_type=input_type,  # type: ignore[arg-type]
        raw_input=content,
        url=url,
        domain=domain,
        retrieved_at=retrieved_at,
        title=title,
        author=author,
        published_at=published_at,
        display_text=display_text,
        language=lang,
        sentences=sents,
        tokens=toks,
        entities=ents,
        attributions=attrs,
        preprocessing_flags=preprocessing_flags,
        preprocessing_provenance=provenance,
    )
