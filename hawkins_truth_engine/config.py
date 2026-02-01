from __future__ import annotations

import os


def env_str(name: str, default: str = "") -> str:
    val = os.getenv(name)
    return default if val is None else val


def env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


HTTP_TIMEOUT_SECS = env_int("HTE_HTTP_TIMEOUT_SECS", 20)

NCBI_TOOL = env_str("HTE_NCBI_TOOL", "hawkins_truth_engine_poc")
NCBI_EMAIL = env_str("HTE_NCBI_EMAIL", "")
NCBI_API_KEY = env_str("HTE_NCBI_API_KEY", "")

GDELT_MAXRECORDS = env_int("HTE_GDELT_MAXRECORDS", 25)

PUBMED_RETMAX = env_int("HTE_PUBMED_RETMAX", 10)
PUBMED_MAX_ABSTRACTS = env_int("HTE_PUBMED_MAX_ABSTRACTS", 3)

FETCH_MAX_BYTES = env_int("HTE_FETCH_MAX_BYTES", 2_000_000)
