"""WHOIS lookup fallback when RDAP is unavailable.

This module provides a best-effort WHOIS lookup capability as a fallback
when RDAP queries fail. It attempts to parse WHOIS responses to extract
domain registration age.
"""
from __future__ import annotations

import socket
import logging
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)


def _whois_domain_sync(domain: str) -> dict:
    """Synchronous implementation of WHOIS lookup logic."""
    whois_server = "whois.iana.org"
    port = 43
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        
        logger.debug(f"Querying WHOIS for {domain} via {whois_server}")
        sock.connect((whois_server, port))
        sock.sendall((domain + "\r\n").encode())
        
        response = b""
        while True:
            try:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
            except socket.timeout:
                break
        
        sock.close()
        
        response_text = response.decode("utf-8", errors="ignore").lower()
        
        creation_date = None
        for line in response_text.split("\n"):
            if any(pattern in line for pattern in ["created:", "creation date:", "registered:"]):
                try:
                    date_part = line.split(":", 1)[1].strip()
                    for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d-%b-%Y", "%b %d %Y"]:
                        try:
                            creation_date = datetime.strptime(date_part[:10], fmt[:10])
                            break
                        except ValueError:
                            continue
                    if creation_date:
                        break
                except (IndexError, ValueError):
                    continue
        
        age_days = None
        if creation_date:
            age_days = max(0, (datetime.now() - creation_date).days)
            logger.debug(f"WHOIS lookup for {domain}: created {creation_date}, age ~{age_days} days")
        
        return {
            "request": {"url": domain},
            "data": {
                "creation_date": creation_date.isoformat() if creation_date else None,
                "age_days": age_days,
            },
            "success": creation_date is not None,
        }
        
    except socket.timeout:
        logger.warning(f"WHOIS query timeout for {domain}")
        return {
            "request": {"url": domain},
            "data": {},
            "success": False,
            "error": "WHOIS query timeout",
        }
    except Exception as e:
        logger.warning(f"WHOIS lookup failed for {domain}: {type(e).__name__}: {str(e)}")
        return {
            "request": {"url": domain},
            "data": {},
            "success": False,
            "error": str(e),
        }


async def whois_domain(domain: str) -> dict:
    """Async wrapper for WHOIS lookup using asyncio.to_thread."""
    return await asyncio.to_thread(_whois_domain_sync, domain)
