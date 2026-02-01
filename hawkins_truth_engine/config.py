from __future__ import annotations

import os
from pathlib import Path

# Load .env file if it exists (for local development)
try:
    from dotenv import load_dotenv
    # Look for .env in project root
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # python-dotenv not installed, skip loading .env


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


def env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


# ============================================================================
# HTTP & External API Settings
# ============================================================================
HTTP_TIMEOUT_SECS = env_int("HTE_HTTP_TIMEOUT_SECS", 20)
FETCH_MAX_BYTES = env_int("HTE_FETCH_MAX_BYTES", 2_000_000)

# NCBI/PubMed settings
NCBI_TOOL = env_str("HTE_NCBI_TOOL", "hawkins_truth_engine_poc")
NCBI_EMAIL = env_str("HTE_NCBI_EMAIL", "")
NCBI_API_KEY = env_str("HTE_NCBI_API_KEY", "")
PUBMED_RETMAX = env_int("HTE_PUBMED_RETMAX", 10)
PUBMED_MAX_ABSTRACTS = env_int("HTE_PUBMED_MAX_ABSTRACTS", 3)

# GDELT settings
GDELT_MAXRECORDS = env_int("HTE_GDELT_MAXRECORDS", 25)

# Optional web search corroboration (Tavily)
TAVILY_API_KEY = env_str("HTE_TAVILY_API_KEY", "")
TAVILY_MAX_RESULTS = env_int("HTE_TAVILY_MAX_RESULTS", 5)
TAVILY_SEARCH_DEPTH = env_str("HTE_TAVILY_SEARCH_DEPTH", "basic")

# Groq LLM settings (for intelligent claim extraction/analysis)
GROQ_API_KEY = env_str("HTE_GROQ_API_KEY", "")
GROQ_MODEL = env_str("HTE_GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MAX_TOKENS = env_int("HTE_GROQ_MAX_TOKENS", 2048)
GROQ_TEMPERATURE = env_float("HTE_GROQ_TEMPERATURE", 0.1)

# ============================================================================
# Reasoning Engine Thresholds
# ============================================================================
# Rule trigger thresholds
REASONING_LOW_TRUST_THRESHOLD = env_float("HTE_LOW_TRUST_THRESHOLD", 0.35)
REASONING_HIGH_TRUST_THRESHOLD = env_float("HTE_HIGH_TRUST_THRESHOLD", 0.75)
REASONING_HIGH_LINGUISTIC_RISK = env_float("HTE_HIGH_LINGUISTIC_RISK", 0.65)
REASONING_LOW_RISK_THRESHOLD = env_float("HTE_LOW_RISK_THRESHOLD", 0.45)

# Scoring weights
REASONING_LINGUISTIC_WEIGHT = env_float("HTE_LINGUISTIC_WEIGHT", 0.40)
REASONING_STATISTICAL_WEIGHT = env_float("HTE_STATISTICAL_WEIGHT", 0.30)
REASONING_SOURCE_WEIGHT = env_float("HTE_SOURCE_WEIGHT", 0.30)

# Gate multipliers
REASONING_LOW_TRUST_MULTIPLIER = env_float("HTE_LOW_TRUST_MULTIPLIER", 1.25)
REASONING_HIGH_TRUST_MULTIPLIER = env_float("HTE_HIGH_TRUST_MULTIPLIER", 0.85)

# Rule override thresholds
REASONING_MIN_FAKE_RISK = env_float("HTE_MIN_FAKE_RISK", 0.80)
REASONING_MAX_REAL_RISK = env_float("HTE_MAX_REAL_RISK", 0.35)

# Verdict score thresholds
VERDICT_LIKELY_REAL_THRESHOLD = env_int("HTE_LIKELY_REAL_THRESHOLD", 70)
VERDICT_SUSPICIOUS_THRESHOLD = env_int("HTE_SUSPICIOUS_THRESHOLD", 40)

# ============================================================================
# Analyzer Thresholds
# ============================================================================
# Linguistic analyzer
LING_CLICKBAIT_PUNCT_THRESHOLD = env_float("HTE_CLICKBAIT_PUNCT_THRESHOLD", 0.25)
LING_CLICKBAIT_CAPS_THRESHOLD = env_float("HTE_CLICKBAIT_CAPS_THRESHOLD", 0.15)

# Statistical analyzer
STAT_LOW_DIVERSITY_THRESHOLD = env_float("HTE_LOW_DIVERSITY_THRESHOLD", 0.22)
STAT_HIGH_REPETITION_THRESHOLD = env_float("HTE_HIGH_REPETITION_THRESHOLD", 0.12)
STAT_UNIFORM_SENTENCE_THRESHOLD = env_float("HTE_UNIFORM_SENTENCE_THRESHOLD", 0.35)
STAT_LOW_ENTROPY_THRESHOLD = env_float("HTE_LOW_ENTROPY_THRESHOLD", 0.72)

# Source intelligence
SOURCE_YOUNG_DOMAIN_DAYS = env_int("HTE_YOUNG_DOMAIN_DAYS", 90)
SOURCE_OLD_DOMAIN_DAYS = env_int("HTE_OLD_DOMAIN_DAYS", 365)

# ============================================================================
# Graph Construction Settings
# ============================================================================
GRAPH_SIMILARITY_THRESHOLD = env_float("HTE_GRAPH_SIMILARITY_THRESHOLD", 0.35)
GRAPH_MENTION_CONFIDENCE_THRESHOLD = env_float("HTE_MENTION_CONFIDENCE_THRESHOLD", 0.25)

# ============================================================================
# Input Validation Limits
# ============================================================================
MAX_CONTENT_LENGTH = env_int("HTE_MAX_CONTENT_LENGTH", 10_000_000)  # 10MB max
MAX_URL_LENGTH = env_int("HTE_MAX_URL_LENGTH", 2048)
MIN_CONTENT_LENGTH = env_int("HTE_MIN_CONTENT_LENGTH", 10)  # Minimum text length
GRAPH_QUICK_SIM_FILTER = env_float("HTE_GRAPH_QUICK_SIM_FILTER", 0.1)
GRAPH_MIN_SIMILARITY = env_float("HTE_GRAPH_MIN_SIMILARITY", 0.15)
GRAPH_SUPPORTED_STRENGTH = env_float("HTE_GRAPH_SUPPORTED_STRENGTH", 0.85)
GRAPH_UNSUPPORTED_STRENGTH = env_float("HTE_GRAPH_UNSUPPORTED_STRENGTH", 0.15)
GRAPH_CONTESTED_STRENGTH = env_float("HTE_GRAPH_CONTESTED_STRENGTH", 0.5)
GRAPH_UNVERIFIABLE_STRENGTH = env_float("HTE_GRAPH_UNVERIFIABLE_STRENGTH", 0.25)
GRAPH_BASE_QUALITY_SCORE = env_float("HTE_GRAPH_BASE_QUALITY_SCORE", 0.5)
GRAPH_HIGH_QUALITY_BOOST = env_float("HTE_GRAPH_HIGH_QUALITY_BOOST", 0.3)
GRAPH_LOW_QUALITY_PENALTY = env_float("HTE_GRAPH_LOW_QUALITY_PENALTY", 0.2)
GRAPH_VERIFIED_SOURCE_BOOST = env_float("HTE_GRAPH_VERIFIED_SOURCE_BOOST", 0.2)
GRAPH_QUESTIONABLE_SOURCE_PENALTY = env_float("HTE_GRAPH_QUESTIONABLE_SOURCE_PENALTY", 0.2)

# ============================================================================
# Reasoning Engine - Claim Adjustment Thresholds
# ============================================================================
# Risk adjustments based on claim support
REASONING_SUPPORTED_CLAIMS_ADJUSTMENT = env_float("HTE_SUPPORTED_CLAIMS_ADJUSTMENT", 0.20)
REASONING_UNVERIFIABLE_CLAIMS_PENALTY = env_float("HTE_UNVERIFIABLE_CLAIMS_PENALTY", 0.10)
REASONING_UNSUPPORTED_CLAIMS_PENALTY = env_float("HTE_UNSUPPORTED_CLAIMS_PENALTY", 0.15)

# Cross-reference and source quality thresholds
REASONING_CROSS_PROVIDER_BOOST = env_float("HTE_CROSS_PROVIDER_BOOST", 0.15)
REASONING_REPUTABLE_SOURCE_BOOST = env_float("HTE_REPUTABLE_SOURCE_BOOST", 0.10)
REASONING_DIVERSITY_BOOST = env_float("HTE_DIVERSITY_BOOST", 0.08)
REASONING_QUESTIONABLE_SOURCE_PENALTY = env_float("HTE_QUESTIONABLE_SOURCE_PENALTY", 0.12)

# Ambiguous case and multi-risk thresholds
REASONING_AMBIGUOUS_MIN_RISK = env_float("HTE_AMBIGUOUS_MIN_RISK", 0.45)
REASONING_MULTIRISK_MIN_RISK = env_float("HTE_MULTIRISK_MIN_RISK", 0.65)
REASONING_CLAIM_AGREEMENT_THRESHOLD = env_float("HTE_CLAIM_AGREEMENT_THRESHOLD", 0.75)

# Confidence adjustments
REASONING_CONFIDENCE_BASE_SCORE = env_float("HTE_CONFIDENCE_BASE_SCORE", 0.35)
REASONING_CONFIDENCE_BASE_COVERAGE = env_float("HTE_CONFIDENCE_BASE_COVERAGE", 0.6)
REASONING_CONFIDENCE_MIN = env_float("HTE_CONFIDENCE_MIN", 0.05)
REASONING_CONFIDENCE_MAX = env_float("HTE_CONFIDENCE_MAX", 0.95)

# ============================================================================
# Document Preprocessing Settings
# ============================================================================
CLAIM_MIN_SENTENCE_LENGTH = env_int("HTE_CLAIM_MIN_SENTENCE_LENGTH", 25)
CLAIM_MAX_CANDIDATES = env_int("HTE_CLAIM_MAX_CANDIDATES", 12)
CLAIM_SNIPPET_MIN_WORDS = env_int("HTE_CLAIM_SNIPPET_MIN_WORDS", 4)
CLAIM_SNIPPET_RELEVANCE_THRESHOLD = env_float("HTE_CLAIM_SNIPPET_RELEVANCE_THRESHOLD", 0.25)

# ============================================================================
# Calibration Settings
# ============================================================================
CALIB_CV_FOLDS = env_int("HTE_CALIB_CV_FOLDS", 3)
CALIB_DEFAULT_TRAIN_SIZE = env_float("HTE_CALIB_DEFAULT_TRAIN_SIZE", 0.8)
CALIB_DEFAULT_VALIDATION_SIZE = env_float("HTE_CALIB_DEFAULT_VALIDATION_SIZE", 0.2)
CALIB_DEFAULT_RANDOM_SEED = env_int("HTE_CALIB_DEFAULT_RANDOM_SEED", 42)
CALIB_CLASS_IMBALANCE_THRESHOLD = env_float("HTE_CALIB_CLASS_IMBALANCE_THRESHOLD", 0.05)
CALIB_MIN_SAMPLES = env_int("HTE_CALIB_MIN_SAMPLES", 20)
CALIB_RECOMMENDED_SAMPLES = env_int("HTE_CALIB_RECOMMENDED_SAMPLES", 100)
CALIB_RELIABILITY_BINS = env_int("HTE_CALIB_RELIABILITY_BINS", 10)
CALIB_FLOAT_TOLERANCE = env_float("HTE_CALIB_FLOAT_TOLERANCE", 0.01)

# ============================================================================
# Configuration Validation
# ============================================================================
def validate_config():
    """Validate configuration values are in acceptable ranges."""
    errors = []
    warnings = []
    
    # Validate probability/threshold values are in [0, 1]
    prob_vars = [
        ("REASONING_LOW_TRUST_THRESHOLD", REASONING_LOW_TRUST_THRESHOLD),
        ("REASONING_HIGH_TRUST_THRESHOLD", REASONING_HIGH_TRUST_THRESHOLD),
        ("REASONING_HIGH_LINGUISTIC_RISK", REASONING_HIGH_LINGUISTIC_RISK),
        ("REASONING_LOW_RISK_THRESHOLD", REASONING_LOW_RISK_THRESHOLD),
        ("REASONING_LINGUISTIC_WEIGHT", REASONING_LINGUISTIC_WEIGHT),
        ("REASONING_STATISTICAL_WEIGHT", REASONING_STATISTICAL_WEIGHT),
        ("REASONING_SOURCE_WEIGHT", REASONING_SOURCE_WEIGHT),
        ("REASONING_MIN_FAKE_RISK", REASONING_MIN_FAKE_RISK),
        ("REASONING_MAX_REAL_RISK", REASONING_MAX_REAL_RISK),
    ]
    
    for name, value in prob_vars:
        if not (0.0 <= value <= 1.0):
            errors.append(f"{name}={value} must be in range [0.0, 1.0]")
    
    # Validate weights sum to 1.0 (with tolerance)
    weight_sum = REASONING_LINGUISTIC_WEIGHT + REASONING_STATISTICAL_WEIGHT + REASONING_SOURCE_WEIGHT
    if not (0.99 <= weight_sum <= 1.01):
        errors.append(
            f"REASONING weights must sum to 1.0, got {weight_sum:.3f} "
            f"(LINGUISTIC={REASONING_LINGUISTIC_WEIGHT}, STATISTICAL={REASONING_STATISTICAL_WEIGHT}, SOURCE={REASONING_SOURCE_WEIGHT})"
        )
    
    # Validate threshold ordering
    if REASONING_LOW_TRUST_THRESHOLD >= REASONING_HIGH_TRUST_THRESHOLD:
        errors.append(
            f"REASONING_LOW_TRUST_THRESHOLD ({REASONING_LOW_TRUST_THRESHOLD}) "
            f"must be < REASONING_HIGH_TRUST_THRESHOLD ({REASONING_HIGH_TRUST_THRESHOLD})"
        )
    
    # Validate calibration settings
    calib_split_sum = CALIB_DEFAULT_TRAIN_SIZE + CALIB_DEFAULT_VALIDATION_SIZE
    if not (0.99 <= calib_split_sum <= 1.01):
        errors.append(
            f"Calibration train+validation sizes must sum to 1.0, got {calib_split_sum:.3f}"
        )
    
    # Check for required API keys (warnings only)
    if not NCBI_EMAIL:
        warnings.append("NCBI_EMAIL not configured - PubMed queries may be rate-limited")
    
    if not TAVILY_API_KEY:
        warnings.append("TAVILY_API_KEY not configured - Tavily search unavailable")
    
    # Raise if there are errors
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(error_msg)
    
    # Log warnings
    if warnings:
        import logging
        logger = logging.getLogger(__name__)
        for warning in warnings:
            logger.warning(f"Config warning: {warning}")
    
    return {"valid": True, "errors": [], "warnings": warnings}

# Run validation on module import
try:
    validate_config()
except ValueError as e:
    # Re-raise with module context
    raise ValueError(f"hawkins_truth_engine.config validation failed: {e}") from e
