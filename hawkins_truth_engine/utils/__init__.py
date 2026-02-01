"""
Utility modules for the Hawkins Truth Engine.

This package contains utility functions for source quality assessment,
text processing, and other helper functions.
"""

# Import from source_quality module
from .source_quality import (
    assess_citation_quality,
    assess_domain_quality,
    assess_journal_quality,
    assess_source_authority,
    calculate_evidence_strength,
    calculate_source_diversity,
    detect_contradictions,
    extract_domain_from_url,
    analyze_temporal_relevance,
)

# Import functions from parent utils.py module using importlib to avoid circular import
import importlib.util
from pathlib import Path

_parent_utils_path = Path(__file__).parent.parent / "utils.py"
if _parent_utils_path.exists():
    spec = importlib.util.spec_from_file_location("hawkins_truth_engine._utils", _parent_utils_path)
    if spec and spec.loader:
        _utils_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_utils_module)
        # Re-export functions from utils.py
        find_spans = _utils_module.find_spans
        normalize_text = _utils_module.normalize_text
        safe_domain = _utils_module.safe_domain
    else:
        # Fallback: define stubs if import fails
        def find_spans(*args, **kwargs):
            raise ImportError("find_spans not available")
        def normalize_text(*args, **kwargs):
            raise ImportError("normalize_text not available")
        def safe_domain(*args, **kwargs):
            raise ImportError("safe_domain not available")
else:
    # Fallback if utils.py doesn't exist
    def find_spans(*args, **kwargs):
        raise ImportError("find_spans not available")
    def normalize_text(*args, **kwargs):
        raise ImportError("normalize_text not available")
    def safe_domain(*args, **kwargs):
        raise ImportError("safe_domain not available")

__all__ = [
    # Functions from utils.py
    "find_spans",
    "normalize_text",
    "safe_domain",
    # Functions from source_quality.py
    "assess_citation_quality",
    "assess_domain_quality",
    "assess_journal_quality",
    "assess_source_authority",
    "calculate_evidence_strength",
    "calculate_source_diversity",
    "detect_contradictions",
    "extract_domain_from_url",
    "analyze_temporal_relevance",
]
