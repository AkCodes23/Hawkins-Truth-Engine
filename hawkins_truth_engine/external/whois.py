"""WHOIS lookup fallback when RDAP is unavailable.

This module provides a best-effort WHOIS lookup capability as a fallback
when RDAP queries fail. It attempts to parse WHOIS responses to extract
domain registration age.
"""
from __future__ import annotations

import socket
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


async def whois_domain(domain: str) -> dict:
    """Query WHOIS for domain information as RDAP fallback.
    
    Args:
        domain: Domain name to query (e.g., 'example.com')
        
    Returns:
        dict with structure {'request': {'url': domain}, 'data': {...}, 'success': bool}
        
    Note:
        This is a best-effort implementation. WHOIS responses vary by registry
        and may not always be parseable. Returns partial data on success.
    """
    whois_server = "whois.iana.org"
    port = 43
    
    try:
        # WHOIS queries use TCP port 43
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
        
        response_text = response.decode("utf-8", errors="ignore").lower()\n        
        # Try to extract creation date from WHOIS response\n        # Common patterns: "created:", "creation date:", "created on:"\n        creation_date = None\n        for line in response_text.split("\n"):\n            if any(pattern in line for pattern in ["created:", "creation date:", "registered:"]):\n                try:\n                    # Simple extraction - format varies by registry\n                    date_part = line.split(":", 1)[1].strip()\n                    # Try parsing common formats\n                    for fmt in [\"%Y-%m-%d\", \"%Y-%m-%d %H:%M:%S\", \"%d-%b-%Y\", \"%b %d %Y\"]:\n                        try:\n                            creation_date = datetime.strptime(date_part[:10], fmt[:10])\n                            break\n                        except ValueError:\n                            continue\n                    if creation_date:\n                        break\n                except (IndexError, ValueError):\n                    continue\n        \n        age_days = None\n        if creation_date:\n            age_days = max(0, (datetime.now() - creation_date).days)\n            logger.debug(f\"WHOIS lookup for {domain}: created {creation_date}, age ~{age_days} days\")\n        \n        return {\n            \"request\": {\"url\": domain},\n            \"data\": {\n                \"creation_date\": creation_date.isoformat() if creation_date else None,\n                \"age_days\": age_days,\n            },\n            \"success\": creation_date is not None,\n        }\n        \n    except socket.timeout:\n        logger.warning(f\"WHOIS query timeout for {domain}\")\n        return {\n            \"request\": {\"url\": domain},\n            \"data\": {},\n            \"success\": False,\n            \"error\": \"WHOIS query timeout\",\n        }\n    except Exception as e:\n        logger.warning(f\"WHOIS lookup failed for {domain}: {type(e).__name__}: {str(e)}\")\n        return {\n            \"request\": {\"url\": domain},\n            \"data\": {},\n            \"success\": False,\n            \"error\": str(e),\n        }
