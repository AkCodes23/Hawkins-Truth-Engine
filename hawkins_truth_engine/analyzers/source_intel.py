from __future__ import annotations

import logging
from datetime import datetime

from ..external.rdap import rdap_domain
from ..external.whois import whois_domain
from ..schemas import EvidenceItem, SourceIntelOutput

logger = logging.getLogger(__name__)


def _rdap_domain_age_days(rdap_json: dict) -> int | None:
    # RDAP events sometimes include registration or last changed.
    events = rdap_json.get("events") or []
    reg = None
    for e in events:
        if e.get("eventAction") in {"registration", "registered"}:
            reg = e.get("eventDate")
            break
    if not reg:
        return None
    try:
        dt = datetime.fromisoformat(reg.replace("Z", "+00:00"))
        return max(0, (datetime.now(dt.tzinfo) - dt).days)
    except Exception:
        return None


async def analyze_source(doc) -> SourceIntelOutput:
    flags: list[EvidenceItem] = []
    trust = 0.5

    if not doc.domain:
        flags.append(
            EvidenceItem(
                id="no_domain",
                module="source",
                weight=0.20,
                value=1.0,
                severity="medium",
                evidence="No domain available (non-URL input).",
                provenance={},
            )
        )
        return SourceIntelOutput(source_trust_score=trust, source_flags=flags)

    try:
        rdap = await rdap_domain(doc.domain)
        rdap_data = rdap.get("data") or {}
        age_days = _rdap_domain_age_days(rdap_data)
        if age_days is not None and age_days < 90:
            flags.append(
                EvidenceItem(
                    id="young_domain",
                    module="source",
                    weight=0.35,
                    value=min(1.0, (90 - age_days) / 90),
                    severity="high",
                    evidence=f"Young domain (age ~{age_days} days via RDAP).",
                    provenance={"rdap_url": rdap["request"]["url"], "age_days": age_days},
                )
            )
        if age_days is not None and age_days >= 365:
            trust += 0.10
        # RDAP status flags
        statuses = rdap_data.get("status") or []
        if any("clienthold" in s.lower() or "serverhold" in s.lower() for s in statuses):
            flags.append(
                EvidenceItem(
                    id="domain_hold_status",
                    module="source",
                    weight=0.25,
                    value=0.8,
                    severity="high",
                    evidence="Domain has hold status in RDAP (potentially unstable).",
                    provenance={"status": statuses, "rdap_url": rdap["request"]["url"]},
                )
            )
    except Exception as e:
        logger.debug(f"RDAP lookup failed for {doc.domain}: {type(e).__name__}, trying WHOIS fallback")
        # Try WHOIS fallback when RDAP fails
        try:
            whois = await whois_domain(doc.domain)
            if whois.get("success"):
                age_days = whois.get("data", {}).get("age_days")
                if age_days is not None and age_days < 90:
                    flags.append(
                        EvidenceItem(
                            id="young_domain_whois",
                            module="source",
                            weight=0.35,
                            value=min(1.0, (90 - age_days) / 90),
                            severity="high",
                            evidence=f"Young domain (age ~{age_days} days via WHOIS fallback).",
                            provenance={"source": "whois", "age_days": age_days},
                        )
                    )
                if age_days is not None and age_days >= 365:
                    trust += 0.10
            else:
                raise Exception(whois.get("error", "WHOIS lookup failed"))
        except Exception as whois_error:
            logger.warning(f"WHOIS fallback also failed for {doc.domain}: {str(whois_error)}")
            flags.append(
                EvidenceItem(
                    id="rdap_unavailable",
                    module="source",
                    weight=0.20,
                    value=1.0,
                    severity="medium",
                    evidence="RDAP and WHOIS lookups failed; source age/stability unknown.",
                    provenance={"rdap_error": str(e), "whois_error": str(whois_error)},
                )
            )

    if not doc.author:
        flags.append(
            EvidenceItem(
                id="missing_author",
                module="source",
                weight=0.15,
                value=1.0,
                severity="medium",
                evidence="Missing author/byline metadata.",
                provenance={},
            )
        )
        trust -= 0.10

    if not doc.published_at:
        flags.append(
            EvidenceItem(
                id="missing_pubdate",
                module="source",
                weight=0.10,
                value=1.0,
                severity="low",
                evidence="Missing publication date metadata.",
                provenance={},
            )
        )
        trust -= 0.05

    # Convert flags into trust adjustment (explainable in aggregation; here just provide a bounded trust score).
    penalty = 0.0
    for f in flags:
        penalty += f.weight * (f.value if f.value is not None else 0.5)
    trust = max(0.0, min(1.0, trust - 0.6 * penalty))
    return SourceIntelOutput(source_trust_score=trust, source_flags=flags)
