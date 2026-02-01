from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from langdetect import detect_langs, LangDetectException

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
logger = logging.getLogger(__name__)


def _best_effort_language(text: str) -> LanguageInfo:
    dist: list[dict[str, Any]] = []
    top = "unknown"
    detection_failed = False
    
    # Skip detection for very short text
    if len(text.strip()) < 20:
        return LanguageInfo(top="unknown", distribution=dist, detection_failed=True, 
                           failure_reason="Text too short for language detection")
    
    try:
        langs = detect_langs(text[:5000])
        for l in langs:
            dist.append({"lang": l.lang, "prob": float(l.prob)})
        if dist:
            top = str(dist[0]["lang"])
    except LangDetectException as e:
        # Specific exception from langdetect
        detection_failed = True
        logger.warning(f"Language detection failed (LangDetectException): {str(e)}")
    except Exception as e:
        # Generic exception handling
        detection_failed = True
        logger.warning(f"Language detection failed (unexpected error): {type(e).__name__}: {str(e)}")
    
    return LanguageInfo(
        top=top, 
        distribution=dist,
        detection_failed=detection_failed,
        failure_reason="Language detection failed - using default" if detection_failed else None
    )


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
    # Enhanced entity extraction with multi-word names, acronyms, and single-word entities
    ent_id = 0
    ents: list[Entity] = []
    org_suffix = {"inc", "ltd", "llc", "corp", "company", "co", "inc.", "ltd.", "llc.", "corp.", "co."}
    
    for s in sentences:
        words = s.text.split()
        i = 0
        detected_entities = set()  # Track detected spans to avoid duplicates
        
        while i < len(words):
            w = words[i]
            w_clean = w.strip(".,;:!?'\"")
            
            # Skip if already detected
            span_key = (s.id, i)
            if span_key in detected_entities:
                i += 1
                continue
            
            # Pattern 1: Multi-word capitalized sequences ("John Smith", "Microsoft Inc")
            if len(w_clean) > 1 and w_clean[0].isupper() and w_clean[1:].islower():
                j = i + 1
                while j < len(words):
                    ww = words[j].strip(".,;:!?'\"")
                    if len(ww) > 1 and ww[0].isupper():
                        j += 1
                        continue
                    break
                
                phrase = " ".join(words[i:j]).strip(".,;:!?'\"")
                if len(phrase.split()) >= 2:
                    entity_type = "PERSON"
                    tail = phrase.split()[-1].lower().strip(".,;:!?")
                    if tail in org_suffix:
                        entity_type = "ORG"
                    
                    start = s.text.find(phrase)
                    if start >= 0:
                        doc_start = s.char_span.start + start
                        doc_end = doc_start + len(phrase)
                        ents.append(
                            Entity(
                                id=ent_id,
                                text=phrase,
                                type=entity_type,
                                sentence_id=s.id,
                                char_span=CharSpan(start=doc_start, end=doc_end),
                                normalized=None,
                            )
                        )
                        detected_entities.add((s.id, i))
                        ent_id += 1
                i = j
            
            # Pattern 2: Single-word capitalized entity (person/place/org name, not at sentence start)
            elif (len(w_clean) > 2 and w_clean[0].isupper() and w_clean[1:].islower() and 
                  (i > 0 or (i == 0 and len(words) > 1 and words[1][0].islower()))):
                # Avoid false positives like sentence-starting "The"
                if w_clean not in {"The", "A", "An", "This", "That", "These", "Those", "It", "They"}:
                    start = s.text.find(w_clean)
                    if start >= 0:
                        doc_start = s.char_span.start + start
                        doc_end = doc_start + len(w_clean)
                        ents.append(
                            Entity(
                                id=ent_id,
                                text=w_clean,
                                type="PERSON",  # Default to PERSON for single words
                                sentence_id=s.id,
                                char_span=CharSpan(start=doc_start, end=doc_end),
                                normalized=None,
                            )
                        )
                        detected_entities.add((s.id, i))
                        ent_id += 1
            
            # Pattern 3: Acronyms ("WHO", "FBI", "US", etc.) - 2+ uppercase letters
            elif len(w_clean) >= 2 and w_clean.isupper() and w_clean.isalpha() and w_clean not in {"A", "I"}:
                start = s.text.find(w_clean)
                if start >= 0:
                    doc_start = s.char_span.start + start
                    doc_end = doc_start + len(w_clean)
                    ents.append(
                        Entity(
                            id=ent_id,
                            text=w_clean,
                            type="ORG",  # Acronyms usually represent organizations
                            sentence_id=s.id,
                            char_span=CharSpan(start=doc_start, end=doc_end),
                            normalized=None,
                        )
                    )
                    detected_entities.add((s.id, i))
                    ent_id += 1
            
            i += 1
    
    return ents


def _attributions_best_effort(text: str, sentences: list[Sentence]) -> list[Attribution]:
    # Enhanced attribution detection with support for multiple quote types and better verb matching
    attrs: list[Attribution] = []
    
    # Support multiple quote types: ASCII quotes, smart quotes, single quotes
    quote_patterns = [
        r'\"([^\"]{10,500})\"',  # ASCII double quotes
        r'\u201c([^\u201d]{10,500})\u201d',  # Unicode smart quotes "..."
        r"'([^']{10,500})'",  # ASCII single quotes
        r'\u2018([^\u2019]{10,500})\u2019',  # Unicode smart single quotes '...'
        r"['\"]([^'\"]{10,500})['\"]"  # Mixed quotes (less strict)
    ]
    
    # Attribution verbs with variations
    attribution_verbs = {
        "said": ["said", "says"],
        "stated": ["stated", "states", "statement"],
        "claimed": ["claimed", "claims", "claim"],
        "reported": ["reported", "reports", "report"],
        "according": ["according", "according to"],
        "told": ["told", "tells"],
        "explained": ["explained", "explains", "explanation"],
        "noted": ["noted", "notes"],
        "argued": ["argued", "argues", "argument"],
        "announced": ["announced", "announces", "announcement"],
        "declared": ["declared", "declares", "declaration"],
        "mentioned": ["mentioned", "mentions"],
        "indicated": ["indicated", "indicates", "indicated"],
    }
    
    all_verbs = set()
    verb_map = {}  # Map normalized verb to canonical form
    for canonical, variations in attribution_verbs.items():
        for var in variations:
            all_verbs.add(var)
            verb_map[var] = canonical
    
    for s in sentences:
        # Try each quote pattern
        for pattern in quote_patterns:
            try:
                quote_re = re.compile(pattern, re.IGNORECASE | re.UNICODE)
                for m in quote_re.finditer(s.text):
                    try:
                        qstart = s.char_span.start + m.start(0)
                        qend = s.char_span.start + m.end(0)
                        
                        # Extract context before the quote (up to 150 chars for better verb matching)
                        context_start = max(0, m.start(0) - 150)
                        ctx_before = s.text[context_start:m.start(0)].lower()
                        
                        # Extract context after the quote (up to 100 chars)
                        context_end = min(len(s.text), m.end(0) + 100)
                        ctx_after = s.text[m.end(0):context_end].lower()
                        
                        # Find the best matching verb
                        verb = "said"  # default
                        best_distance = float('inf')
                        
                        # Check context before quote (more reliable for attribution)
                        for candidate_verb in all_verbs:
                            pos = ctx_before.rfind(candidate_verb)  # Find closest occurrence
                            if pos >= 0:
                                distance = len(ctx_before) - pos  # Distance from end of context
                                if distance < best_distance:
                                    best_distance = distance
                                    verb = verb_map.get(candidate_verb, candidate_verb)
                        
                        # Check context after quote if no good match before
                        if best_distance == float('inf'):
                            for candidate_verb in all_verbs:
                                pos = ctx_after.find(candidate_verb)
                                if pos >= 0:
                                    distance = pos  # Distance from start of context
                                    if distance < best_distance:
                                        best_distance = distance
                                        verb = verb_map.get(candidate_verb, candidate_verb)
                        
                        attrs.append(
                            Attribution(
                                speaker_entity_id=None,  # TODO: Link to extracted entities
                                verb=verb,
                                quote_span=CharSpan(start=qstart, end=qend),
                                sentence_id=s.id
                            )
                        )
                    except (ValueError, IndexError):
                        # Skip malformed matches
                        continue
            except (re.error, TypeError):
                # Skip if regex pattern fails
                continue
    
    # Remove duplicates (same quote span)
    unique_attrs = {}
    for attr in attrs:
        key = (attr.quote_span.start, attr.quote_span.end, attr.sentence_id)
        if key not in unique_attrs or attr.verb in all_verbs:  # Prefer standard verbs
            unique_attrs[key] = attr
    
    return list(unique_attrs.values())


async def fetch_url(url: str) -> dict[str, Any]:
    """
    Fetch URL content with comprehensive error handling.
    
    Returns a dict containing either fetched content or error information.
    This function never raises exceptions - all errors are returned in the dict.
    """
    timeout = httpx.Timeout(HTTP_TIMEOUT_SECS)
    headers = {
        "User-Agent": "HawkinsTruthEnginePOC/0.1 (+https://example.invalid)",
        "Accept": "text/html,application/xhtml+xml",
    }
    
    # Default error response structure
    error_response = {
        "final_url": url,
        "status_code": 0,
        "headers": {},
        "content": b"",
        "error": None,
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
            r = await client.get(url)
            content = r.content[:FETCH_MAX_BYTES]
            return {
                "final_url": str(r.url),
                "status_code": r.status_code,
                "headers": dict(r.headers),
                "content": content,
                "error": None,
            }
    except httpx.TimeoutException:
        error_response["error"] = "timeout"
        return error_response
    except httpx.ConnectError as e:
        error_response["error"] = f"connection_failed: {str(e)}"
        return error_response
    except httpx.InvalidURL as e:
        error_response["error"] = f"invalid_url: {str(e)}"
        return error_response
    except httpx.TooManyRedirects:
        error_response["error"] = "too_many_redirects"
        return error_response
    except httpx.HTTPStatusError as e:
        error_response["error"] = f"http_error: {e.response.status_code}"
        error_response["status_code"] = e.response.status_code
        return error_response
    except Exception as e:
        error_response["error"] = f"fetch_failed: {type(e).__name__}: {str(e)}"
        return error_response


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
        
        # Handle fetch errors gracefully
        fetch_error = fetched.get("error")
        if fetch_error:
            preprocessing_flags.append("fetch_error")
            preprocessing_flags.append(f"fetch_error_type:{fetch_error.split(':')[0] if ':' in fetch_error else fetch_error}")
            provenance["fetch"] = {
                "status_code": fetched.get("status_code", 0),
                "final_url": fetched.get("final_url", url),
                "retrieved_at": retrieved_at.isoformat(),
                "error": fetch_error,
            }
            # Use empty content - will result in minimal document
            display_text = ""
        else:
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
