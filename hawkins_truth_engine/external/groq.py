"""
Groq LLM API integration for intelligent claim extraction and analysis.

Uses Groq's fast inference API with Llama models for:
- Extracting factual claims from text
- Classifying claim types
- Identifying claim verifiability
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ..config import (
    GROQ_API_KEY,
    GROQ_MAX_TOKENS,
    GROQ_MODEL,
    GROQ_TEMPERATURE,
    HTTP_TIMEOUT_SECS,
)

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


async def groq_chat_completion(
    messages: list[dict[str, str]],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    response_format: dict | None = None,
) -> dict[str, Any]:
    """
    Send a chat completion request to Groq API.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model to use (defaults to config)
        max_tokens: Max tokens in response (defaults to config)
        temperature: Sampling temperature (defaults to config)
        response_format: Optional format spec (e.g., {"type": "json_object"})
        
    Returns:
        Dict with 'data' key containing response, or 'error' key on failure
    """
    if not GROQ_API_KEY:
        return {"error": "groq_not_configured", "data": None}
    
    payload = {
        "model": model or GROQ_MODEL,
        "messages": messages,
        "max_tokens": max_tokens or GROQ_MAX_TOKENS,
        "temperature": temperature if temperature is not None else GROQ_TEMPERATURE,
    }
    
    if response_format:
        payload["response_format"] = response_format
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    
    timeout = httpx.Timeout(HTTP_TIMEOUT_SECS * 2)  # LLM calls may take longer
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(GROQ_API_URL, json=payload, headers=headers)
            r.raise_for_status()
            return {"data": r.json(), "error": None}
    except httpx.TimeoutException:
        logger.warning("Groq API timeout")
        return {"error": "timeout", "data": None}
    except httpx.HTTPStatusError as e:
        error_detail = ""
        try:
            error_detail = e.response.text[:500]
        except Exception:
            pass
        logger.warning(f"Groq API HTTP error {e.response.status_code}: {error_detail}")
        return {"error": f"http_error_{e.response.status_code}", "data": None, "detail": error_detail}
    except httpx.ConnectError as e:
        logger.warning(f"Groq API connection error: {e}")
        return {"error": f"connection_failed", "data": None}
    except Exception as e:
        logger.warning(f"Groq API error: {type(e).__name__}: {e}")
        return {"error": f"groq_error: {type(e).__name__}", "data": None}


def get_completion_text(response: dict) -> str | None:
    """Extract the text content from a Groq API response."""
    if not response or "error" in response and response["error"]:
        return None
    
    data = response.get("data")
    if not data:
        return None
    
    choices = data.get("choices", [])
    if not choices:
        return None
    
    message = choices[0].get("message", {})
    return message.get("content")


async def extract_claims_with_llm(text: str, sentences: list[str]) -> dict[str, Any]:
    """
    Use Groq LLM to extract and classify claims from text.
    
    Args:
        text: Full document text
        sentences: List of sentences from the document
        
    Returns:
        Dict with extracted claims and metadata
    """
    if not GROQ_API_KEY:
        return {"claims": [], "error": "groq_not_configured"}
    
    # Truncate text if too long
    max_text_len = 4000
    if len(text) > max_text_len:
        text = text[:max_text_len] + "..."
    
    system_prompt = """You are a fact-checking assistant that extracts factual claims from text.

For each claim, identify:
1. The exact claim text
2. The claim type: "factual", "speculative", "predictive", or "opinion"
3. Whether it's verifiable (can be fact-checked)
4. Key entities or topics mentioned

Respond in JSON format with this structure:
{
  "claims": [
    {
      "text": "exact claim text",
      "type": "factual|speculative|predictive|opinion",
      "verifiable": true|false,
      "topics": ["topic1", "topic2"],
      "confidence": 0.0-1.0
    }
  ],
  "summary": "brief summary of the text's main claims",
  "risk_indicators": ["list of any misinformation red flags detected"]
}

Focus on extracting FACTUAL CLAIMS that can be verified. Ignore questions, greetings, and purely subjective statements."""

    user_prompt = f"""Extract all factual claims from this text:

---
{text}
---

Respond ONLY with valid JSON."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    response = await groq_chat_completion(
        messages,
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    
    if response.get("error"):
        return {"claims": [], "error": response["error"]}
    
    content = get_completion_text(response)
    if not content:
        return {"claims": [], "error": "empty_response"}
    
    try:
        parsed = json.loads(content)
        return {
            "claims": parsed.get("claims", []),
            "summary": parsed.get("summary", ""),
            "risk_indicators": parsed.get("risk_indicators", []),
            "error": None,
        }
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Groq response as JSON: {e}")
        return {"claims": [], "error": "json_parse_error", "raw": content[:500]}


async def analyze_claim_credibility(claim: str, context: str = "") -> dict[str, Any]:
    """
    Use Groq LLM to analyze a claim's credibility based on linguistic patterns.
    
    Args:
        claim: The claim text to analyze
        context: Optional surrounding context
        
    Returns:
        Dict with credibility analysis
    """
    if not GROQ_API_KEY:
        return {"error": "groq_not_configured"}
    
    system_prompt = """You are a misinformation detection expert. Analyze claims for credibility signals.

Look for:
- Clickbait language patterns
- Conspiracy theory framing
- Anonymous sources ("experts say", "studies show")
- Emotional manipulation
- Absolute claims without evidence
- Medical misinformation patterns

Respond in JSON:
{
  "credibility_score": 0-100,
  "red_flags": ["list of concerning patterns"],
  "reasoning": "brief explanation",
  "claim_category": "health|politics|science|general|unknown"
}"""

    user_prompt = f"""Analyze this claim for credibility:

Claim: {claim}
{f'Context: {context}' if context else ''}

Respond ONLY with valid JSON."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    response = await groq_chat_completion(
        messages,
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=512,
    )
    
    if response.get("error"):
        return {"error": response["error"]}
    
    content = get_completion_text(response)
    if not content:
        return {"error": "empty_response"}
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "json_parse_error"}


def is_groq_available() -> bool:
    """Check if Groq API is configured."""
    return bool(GROQ_API_KEY)
