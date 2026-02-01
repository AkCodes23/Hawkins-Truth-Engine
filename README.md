# Hawkins Truth Engine (POC)

> **An Explainable, Evidence-First Credibility Reasoning System for Misinformation Detection**

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Design Philosophy](#design-philosophy)
3. [Innovation & Uniqueness](#innovation--uniqueness)
4. [System Architecture](#system-architecture)
5. [End-to-End Pipeline](#end-to-end-pipeline)
6. [Module Reference](#module-reference)
7. [Evidence Item Structure](#evidence-item-structure)
8. [Deterministic Rule System](#deterministic-rule-system)
9. [Verdict & World Label Mapping](#verdict--world-label-mapping)
10. [API Response Structure](#api-response-structure)
11. [External Evidence Providers](#external-evidence-providers)
12. [Configuration & Environment Variables](#configuration--environment-variables)
13. [Security Considerations](#security-considerations)
14. [Limitations & Extensions](#limitations--extensions)
15. [Quickstart Guide](#quickstart-guide)
16. [FAQ](#faq)

---

## Problem Statement

### The Challenge: Misinformation in the Digital Age

In the fictional town of Hawkins, rumors and strange stories spread as fast as creatures from the Upside Down. With misinformation creeping into every corner of the community, citizens struggle to distinguish reality from distorted tales. Fake reports about supernatural events, missing people, or secret experiments cause panic and confusion.

**This mirrors a real-world crisis**: the proliferation of fake news, health misinformation, and manipulated content across social media platforms and news outlets. Traditional approaches to this problem often rely on:

- **Black-box classifiers** that provide a binary "fake/real" label with no explanation
- **Human fact-checkers** who cannot scale to the volume of content
- **Keyword blocklists** that are easily circumvented

### Our Solution: The Hawkins Truth Engine

The Hawkins Truth Engine is an **intelligent credibility assessment system** that analyzes news articles, social media posts, and URLs to determine whether content belongs to the "Real World" or the "Upside Down."

**Key Differentiator**: The system doesn't just classify—it **explains why** a story appears suspicious using:

- **Linguistic evidence** (writing patterns, clickbait indicators, conspiracy language)
- **Statistical evidence** (lexical diversity, entropy, repetition patterns)
- **Source-based evidence** (domain age, registration status, authorship)
- **Corroboration evidence** (cross-referencing with scientific literature and news archives)

### Problem Domain Relevance

| Real-World Problem | How Hawkins Truth Engine Addresses It |
|-------------------|--------------------------------------|
| Health misinformation spreading during pandemics | Medical topic detection + PubMed corroboration + harm potential flagging |
| Clickbait articles with sensationalized claims | Linguistic pattern analysis (punctuation, caps, urgency lexicon) |
| Anonymous websites spreading conspiracy theories | Source intelligence via RDAP (domain age, registration status) |
| Claims presented without attribution or sources | Claim extraction + attribution detection + support labeling |
| Inability to explain why content is flagged | Full evidence ledger with provenance, reasoning path, and evidence IDs |

---

## Design Philosophy

### Evidence-First, Not Label-First

The Hawkins Truth Engine fundamentally differs from traditional fake news detectors:

```
Traditional Approach:           Hawkins Truth Engine:
─────────────────────           ─────────────────────
Input → Black Box → Label       Input → Evidence Extraction → Aggregation → Explained Verdict
                                         ↓                        ↓
                                  Evidence Ledger          Reasoning Path
                                  (traceable items)        (rule triggers)
```

### Core Principles

1. **Transparency Over Accuracy Claims**: We do not claim to determine absolute truth. We provide an interpretable assessment with explicit uncertainty flags.

2. **Evidence Traceability**: Every signal is recorded as an `EvidenceItem` with:
   - Unique identifier (`id`)
   - Source module (`module`)
   - Severity classification (`severity`)
   - Pointers back to source text (`char_spans`, `sentence_ids`)
   - Provenance metadata (API URLs, query parameters)

3. **Deterministic Reasoning**: No hidden neural weights. Every decision is traceable through explicit rules in `reasoning.py`.

4. **Conservative Claim Labeling**: "Unsupported" means "no backing found"—**not** "false." We never treat absence of evidence as evidence of falsity.

5. **Uncertainty Acknowledgment**: External service failures, missing metadata, and ambiguous signals are explicitly flagged, not hidden.

### What This POC Is (And Is Not)

| This POC IS | This POC IS NOT |
|-------------|-----------------|
| An explainable evidence ledger | A binary fake-news classifier |
| A credibility aid for human review | An end-to-end black box model |
| Conservative about claim labeling | A guarantee of truth or accuracy |
| Transparent about limitations | Medical advice or expert judgment |
| A triage tool for reviewers | An automated content moderation system |

---

## Innovation & Uniqueness

### Differentiators from Existing Solutions

| Aspect | Existing Solutions | Hawkins Truth Engine |
|--------|-------------------|---------------------|
| **Output** | Binary label (fake/real) | Tri-verdict + confidence + reasoning path |
| **Explainability** | None or post-hoc LIME/SHAP | Built-in evidence ledger with provenance |
| **Multi-signal fusion** | Single model (usually NLP-only) | 4 independent analyzers + deterministic fusion |
| **Claim handling** | Ignored or sentence-level classification | Explicit claim extraction + external corroboration |
| **Medical content** | No special handling | PubMed integration + harm potential flagging |
| **Source assessment** | Blocklist-based | Real-time RDAP domain intelligence |
| **Uncertainty** | Hidden in model weights | Explicit flags + confidence dampening |

### Novel Technical Contributions

1. **Evidence Item Architecture**: Structured data model ensuring every signal carries `id`, `module`, `weight`, `severity`, `value`, `evidence`, `pointers`, and `provenance`.

2. **Hybrid Corroboration Pipeline**: Combines scientific literature (PubMed) with news archives (GDELT) for multi-source claim verification.

3. **Deterministic Reasoning Layer**: Explicit rules that can be audited, modified, and extended without retraining any model.

4. **World Label Abstraction**: Maps technical verdicts to thematic labels ("Real World" / "Upside Down") for engaging presentation.

5. **Preprocessing Provenance Chain**: Full traceability from raw input through extraction to evidence generation.

### Appropriate Technology Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| Web Framework | FastAPI | Async I/O for external API calls, automatic OpenAPI docs |
| Data Validation | Pydantic v2 | Type-safe schemas, JSON serialization, validation |
| HTTP Client | httpx | Async support, timeout handling, streaming |
| Text Extraction | trafilatura + BeautifulSoup | Robust HTML-to-text with fallback |
| Language Detection | langdetect | Lightweight, no external dependencies |
| Statistical Analysis | numpy + scikit-learn | Industry-standard numerical computing |
| String Matching | rapidfuzz | Fast fuzzy matching for claim comparison |

---

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                     (HTML UI at / or POST /analyze)                         │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INGEST LAYER                                       │
│                        hawkins_truth_engine/ingest.py                        │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │ URL Fetch   │→│ HTML Extract  │→│ NLP Preproc │→│ Document Builder │  │
│  │ (httpx)     │  │ (trafilatura) │  │ (sentences, │  │ (tokens,entities)│  │
│  └─────────────┘  └──────────────┘  │  tokens)    │  └──────────────────┘  │
│                                      └─────────────┘                         │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ Document
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MULTI-SIGNAL ANALYZERS                                 │
│                    hawkins_truth_engine/analyzers/*                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │   Linguistic    │  │   Statistical   │  │ Source Intel    │             │
│  │   linguistic.py │  │  statistical.py │  │ source_intel.py │             │
│  │                 │  │                 │  │                 │             │
│  │ • Clickbait     │  │ • Lexical div.  │  │ • RDAP lookup   │             │
│  │ • Conspiracy    │  │ • Repetition    │  │ • Domain age    │             │
│  │ • Urgency       │  │ • Entropy       │  │ • Author check  │             │
│  │ • Authority     │  │ • Burstiness    │  │ • Publish date  │             │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘             │
│           │                    │                    │                       │
│           ▼                    ▼                    ▼                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Claims & Corroboration                            │   │
│  │                    hawkins_truth_engine/analyzers/claims.py          │   │
│  │                                                                      │   │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │   │
│  │  │ Claim        │    │ PubMed       │    │ GDELT        │          │   │
│  │  │ Extraction   │ →  │ Corroboration│    │ News Search  │          │   │
│  │  └──────────────┘    └──────────────┘    └──────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ Evidence Items + Module Outputs
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REASONING & AGGREGATION                                   │
│                  hawkins_truth_engine/reasoning.py                           │
│                                                                              │
│   Signals → Base Risk Calculation → Source Trust Gate → Claim Adjustments   │
│                                          ↓                                   │
│                              Rule Evaluation (R1, R2, R3)                    │
│                                          ↓                                   │
│                    credibility_score + verdict + world_label                 │
│                                          ↓                                   │
│                              reasoning_path (evidence_ids)                   │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ AggregationOutput
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXPLANATION GENERATION                                  │
│                    hawkins_truth_engine/explain.py                           │
│                                                                              │
│   verdict_text + evidence_bullets + assumptions + blind_spots + spans       │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │ VerdictExplanation
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API RESPONSE                                         │
│                      AnalysisResponse                                        │
│                                                                              │
│  { document, linguistic, statistical, source_intel, claims,                 │
│    aggregation, explanation }                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## End-to-End Pipeline

### Stage 1: Input & Preprocessing

**File**: `hawkins_truth_engine/ingest.py`

The pipeline accepts three input types:
- `raw_text`: Direct text content
- `url`: Web page URL (fetched and extracted)
- `social_post`: Social media content (treated as raw text)

**Processing Steps**:

```
Input → URL Fetch (if needed) → HTML Extraction → Normalization →
        Sentence Splitting → Tokenization → Entity Recognition →
        Attribution Detection → Language Detection → Document
```

### Stage 2: Multi-Signal Analysis

Four independent analyzers process the Document in parallel:

| Analyzer | File | Output | Risk Metric |
|----------|------|--------|-------------|
| Linguistic | `analyzers/linguistic.py` | `LinguisticOutput` | `linguistic_risk_score` (0-1) |
| Statistical | `analyzers/statistical.py` | `StatisticalOutput` | `statistical_risk_score` (0-1) |
| Source Intel | `analyzers/source_intel.py` | `SourceIntelOutput` | `source_trust_score` (0-1) |
| Claims | `analyzers/claims.py` | `ClaimsOutput` | Claim counts + support labels |

### Stage 3: Evidence Aggregation & Deterministic Reasoning

**File**: `hawkins_truth_engine/reasoning.py`

Combines signals using explicit, auditable rules:

```python
# Simplified scoring logic
base_risk = 0.55 * linguistic_risk + 0.45 * statistical_risk
# Apply source trust gate
if source_trust < 0.35:
    risk *= 1.25
elif source_trust > 0.75:
    risk *= 0.85
# Apply claim support adjustments
# Apply rule overrides
credibility_score = 100 * (1 - final_risk)
```

### Stage 4: Explainable Verdict

**File**: `hawkins_truth_engine/explain.py`

Generates human-readable output:
- Verdict headline with confidence
- Top evidence bullets (ranked by severity)
- Claim summary
- Stated assumptions
- Acknowledged blind spots
- Highlighted text spans for UI

### Stage 5: API/UI Delivery

**File**: `hawkins_truth_engine/app.py`

- **GET /**: Minimal HTML interface with input form and results display
- **POST /analyze**: JSON API returning full `AnalysisResponse`
- **GET /docs**: Auto-generated OpenAPI documentation

---

## Module Reference

### Core Processing Pipeline

The Hawkins Truth Engine processes content through a systematic pipeline where each module has a specific role in building evidence for credibility assessment.

---

### 🔄 `hawkins_truth_engine/ingest.py` — Document Builder & Preprocessor

**What it does**: Converts raw input (text, URLs, social posts) into a structured `Document` that all analyzers can work with.

**Why it's essential**: Without proper preprocessing, analyzers would work with inconsistent, messy data. This module standardizes everything into a clean format with metadata like sentences, tokens, entities, and attributions.

**How it works**:
1. **URL Fetching**: Downloads web pages with size limits and timeout protection
2. **Text Extraction**: Uses trafilatura (with BeautifulSoup fallback) to extract clean text from HTML
3. **Content Parsing**: Breaks text into sentences and tokens with precise character positions
4. **Entity Recognition**: Identifies people and organizations using capitalization patterns
5. **Attribution Detection**: Finds quotes and who said them using regex patterns
6. **Language Detection**: Determines the primary language of the content

**Key Functions**:

| Function | What It Does | Why It Matters |
|----------|-------------|----------------|
| `fetch_url(url)` | Downloads web pages safely with size/timeout limits | Prevents system abuse and hanging requests |
| `extract_text_from_html(html, url)` | Extracts clean text + metadata (title, author, date) | Gets structured content instead of raw HTML |
| `_sentences(text)` | Splits text into sentences with character positions | Enables precise highlighting and reference tracking |
| `_tokens(text)` | Breaks text into words with positions | Allows statistical analysis and pattern detection |
| `_entities_best_effort(sentences, tokens)` | Finds people/organizations in text | Helps detect anonymous vs. attributed claims |
| `_attributions_best_effort(sentences, entities)` | Matches quotes to speakers | Critical for assessing claim attribution |

**Output**: A structured `Document` object containing all extracted features that other modules need.

---

### 📝 `hawkins_truth_engine/analyzers/linguistic.py` — Writing Pattern Detector

**What it does**: Analyzes how content is written to detect patterns commonly found in misinformation.

**Why it's essential**: Misinformation often uses specific language patterns (clickbait, conspiracy framing, urgency) that legitimate news avoids. These patterns are strong credibility signals.

**How it works**: Scans text for suspicious writing patterns and assigns risk scores based on frequency and severity.

**Detected Patterns & Why They Matter**:

| Pattern | How It's Detected | Why It Indicates Risk |
|---------|------------------|----------------------|
| **Clickbait Punctuation** | High `!` and `?` frequency (>25% of sentences) | Legitimate news uses measured tone; excessive punctuation suggests sensationalism |
| **Clickbait Caps** | High ALL-CAPS token ratio | Professional writing avoids shouting; caps suggest emotional manipulation |
| **Clickbait Phrases** | "you won't believe", "shocking", "miracle", "secret", "exposed" | These phrases are designed to bypass critical thinking |
| **Conspiracy Phrases** | "they don't want you to know", "mainstream media", "cover-up", "deep state" | Indicates conspiratorial framing that undermines institutional trust |
| **Urgency Lexicon** | "urgent", "now", "immediately", "warning", "alert", "breaking" | Creates false time pressure to prevent fact-checking |
| **Certainty Without Hedging** | Absolute claims without qualifying language | Legitimate sources use hedging ("may", "suggests") for uncertain claims |
| **Anonymous Authority** | "experts say", "scientists say" without names | Vague attribution is a red flag; credible sources name their experts |

**Output**: Risk score (0-1) plus specific evidence items with text highlighting for each detected pattern.

---

### 📊 `hawkins_truth_engine/analyzers/statistical.py` — Text Structure Analyzer

**What it does**: Examines the mathematical properties of text to detect artificial or manipulated content.

**Why it's essential**: Human writing has natural statistical patterns. Content that deviates significantly from these patterns may be generated, templated, or manipulated.

**How it works**: Calculates statistical measures of text structure and flags unusual patterns.

**Detected Anomalies & Their Significance**:

| Pattern | Threshold | What It Reveals |
|---------|-----------|----------------|
| **Low Lexical Diversity** | Vocabulary richness < 0.22 (for >200 tokens) | Repetitive content suggests keyword stuffing or template generation |
| **High Repetition Ratio** | Top 5 tokens > 12% of text (for >120 tokens) | Excessive repetition indicates SEO manipulation or bot-generated content |
| **Uniform Sentence Length** | Coefficient of variation < 0.35 (for ≥8 sentences) | Natural writing varies sentence length; uniformity suggests artificial generation |
| **Low Token Entropy** | Normalized entropy < 0.72 | Irregular word distribution may indicate manipulation or poor translation |

**Why these patterns matter**: Legitimate journalism has natural variation in vocabulary, sentence structure, and word usage. Statistical anomalies often indicate content designed to manipulate search engines or readers rather than inform them.

**Output**: Risk score (0-1) plus evidence items explaining which statistical thresholds were exceeded.

---

### 🔍 `hawkins_truth_engine/analyzers/source_intel.py` — Source Credibility Assessor

**What it does**: Investigates the source of content to assess its trustworthiness based on domain history and metadata.

**Why it's essential**: The source of information is often as important as the content itself. New domains, missing authorship, and suspicious registration patterns are strong credibility indicators.

**How it works**: Queries external services to gather intelligence about domains and analyzes document metadata.

**Intelligence Checks & Their Importance**:

| Check | Data Source | Why It Matters |
|-------|-------------|----------------|
| **Domain Age** | RDAP registry lookup | New domains (<90 days) are often used for misinformation campaigns; established domains (>1 year) have more credibility |
| **Domain Hold Status** | RDAP status field | Domains on "hold" may be suspended for policy violations or disputes |
| **Missing Author** | Document metadata | Anonymous content lacks accountability; credible sources identify their writers |
| **Missing Publication Date** | Document metadata | Undated content prevents verification and context assessment |
| **RDAP Service Availability** | External service status | When domain intelligence is unavailable, confidence is reduced |

**Trust Score Calculation**:
- Starts at neutral (0.5)
- Young domains: -0.20 penalty
- Old domains: +0.10 bonus  
- Missing author: -0.10 penalty
- Missing date: -0.05 penalty

**Output**: Trust score (0-1), evidence flags, and uncertainty indicators when external services fail.

---

### 🎯 `hawkins_truth_engine/analyzers/claims.py` — Claim Extraction & Verification

**What it does**: Identifies factual claims in content and searches external databases to verify them.

**Why it's essential**: The core of misinformation detection is checking whether claims can be supported by credible sources. This module does the heavy lifting of fact-checking.

**How it works**:
1. **Claim Extraction**: Identifies declarative sentences that make factual assertions
2. **Claim Classification**: Categorizes claims by type (factual, speculative, predictive, opinion)
3. **Medical Detection**: Flags health-related content for special handling
4. **External Verification**: Searches PubMed (medical) and GDELT (news) for supporting evidence
5. **Support Labeling**: Classifies each claim's level of external support

**Claim Processing Pipeline**:
```
Document → Extract Claims (up to 12) → Classify Type → Detect Medical Topics → 
Search External Sources → Analyze Relevance → Assign Support Labels
```

**Support Classification & Meaning**:

| Label | Criteria | What It Means |
|-------|----------|---------------|
| **Supported** | ≥2 external citations with relevant snippets | Strong evidence backing the claim |
| **Unsupported** | Strong claim without attribution + no external backing | Red flag: confident claim with no support |
| **Unverifiable** | No relevant search results or unclear matches | Cannot confirm or deny with available sources |
| **Contested** | Conflicting evidence found in search results | Claim has both supporting and opposing evidence |

**Medical Topic Handling**:
- Detects 28+ medical trigger terms ("cure", "vaccine", "covid", "cancer", etc.)
- Uses PubMed for scientific literature verification
- Flags strong medical claims without attribution as high-risk
- Special handling because medical misinformation can cause physical harm

**External Data Sources**:
- **PubMed (NCBI)**: Biomedical literature for health claims
- **GDELT**: Global news archive for general factual claims

**Output**: Claim counts by support level, medical topic flags, and detailed claim items with verification results.

---

### ⚖️ `hawkins_truth_engine/reasoning.py` — Decision Engine

**What it does**: Combines evidence from all analyzers using explicit, auditable rules to produce a final credibility verdict.

**Why it's essential**: Raw analyzer outputs need to be intelligently combined. This module implements the decision logic that determines whether content is credible, suspicious, or likely fake.

**How it works**: Uses a deterministic rule system (not machine learning) where every decision can be traced and explained.

**Decision Process**:
```
Analyzer Outputs → Extract Signals → Calculate Base Risk → Apply Source Trust Gate → 
Adjust for Claims → Apply Rule Overrides → Generate Final Score → Assign Verdict
```

**Core Rules**:

| Rule | Trigger Conditions | Effect | Rationale |
|------|-------------------|--------|-----------|
| **R1: High-Risk Combination** | Low source trust + high linguistic risk + no claim support | Force toward "Likely Fake" | Multiple red flags together indicate high misinformation probability |
| **R2: Medical Harm Potential** | Medical topic + strong unsupported claims | Add uncertainty flag | Medical misinformation requires extra caution due to harm potential |
| **R3: High-Trust Override** | High source trust + low risk signals | Force toward "Likely Real" | Trusted sources with clean signals get credibility boost |

**Scoring Algorithm**:
1. **Base Risk**: 55% linguistic + 45% statistical risk
2. **Source Gate**: Amplify risk for untrusted sources, dampen for trusted ones
3. **Claim Adjustments**: Reduce risk for supported claims, increase for unsupported
4. **Rule Overrides**: Apply explicit rules that can override base calculations
5. **Final Score**: Convert to 0-100 credibility score

**Output**: Credibility score, verdict (Likely Real/Suspicious/Likely Fake), world label, confidence estimate, and complete reasoning path.

---

### 💬 `hawkins_truth_engine/explain.py` — Human-Readable Explanation Generator

**What it does**: Converts technical analysis results into clear, understandable explanations for human reviewers.

**Why it's essential**: Technical scores and evidence items aren't useful without clear explanations. This module makes the system's reasoning transparent and actionable.

**How it works**: Ranks evidence by importance, generates bullet points, and creates explanatory text with acknowledged limitations.

**Generated Components**:

| Component | Purpose | Example |
|-----------|---------|---------|
| **Verdict Text** | Clear summary with confidence | "World: Upside Down \| Verdict: Suspicious (67% confidence)" |
| **Evidence Bullets** | Top 6 most important findings | "High severity: Conspiracy phrase detected: 'big pharma'" |
| **Assumptions** | What the system assumes to be true | "Online evidence providers returned representative results" |
| **Blind Spots** | Acknowledged limitations | "No guaranteed real-time fact verification" |
| **Highlighted Spans** | Text positions for UI highlighting | Character positions of detected patterns |

**Evidence Ranking**: Combines severity (high/medium/low), weight (importance), and value (strength) to prioritize the most significant findings.

**Output**: Complete explanation package ready for display to human reviewers.

---

### 🌐 `hawkins_truth_engine/app.py` — Web Interface & API

**What it does**: Provides both a web interface and REST API for accessing the truth engine functionality.

**Why it's essential**: Makes the system accessible to users through a simple web form and to developers through a programmatic API.

**Features**:
- **Web UI**: Simple form for entering content and viewing results
- **REST API**: JSON endpoint for programmatic access
- **Auto-documentation**: OpenAPI/Swagger docs at `/docs`
- **Real-time Analysis**: Processes content and returns results immediately

**Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Web interface with input form and results display |
| `/analyze` | POST | Main analysis API accepting JSON requests |
| `/docs` | GET | Interactive API documentation |

**Input Types Supported**:
- `raw_text`: Direct text content
- `url`: Web page URL (automatically fetched)
- `social_post`: Social media content

---

### 📋 `hawkins_truth_engine/schemas.py` — Data Structure Definitions

**What it does**: Defines all data structures used throughout the system using Pydantic for type safety and validation.

**Why it's essential**: Ensures data consistency across modules and provides automatic validation, serialization, and documentation.

**Key Model Categories**:

| Category | Purpose | Key Models |
|----------|---------|------------|
| **Core Types** | Basic enums and primitives | `InputType`, `Verdict`, `WorldLabel` |
| **Document Structure** | Parsed content representation | `Document`, `Sentence`, `Token`, `Entity` |
| **Evidence System** | Traceability and provenance | `EvidenceItem`, `Pointer`, `CharSpan` |
| **Analyzer Outputs** | Module-specific results | `LinguisticOutput`, `StatisticalOutput`, etc. |
| **API Interface** | Request/response formats | `AnalyzeRequest`, `AnalysisResponse` |

---

### ⚙️ `hawkins_truth_engine/config.py` — Configuration Management

**What it does**: Manages all configurable settings through environment variables with sensible defaults.

**Why it's essential**: Allows customization of timeouts, API limits, and external service parameters without code changes.

**Key Configuration Areas**:

| Setting Type | Purpose | Examples |
|--------------|---------|----------|
| **HTTP Settings** | Request timeouts and size limits | `HTE_HTTP_TIMEOUT_SECS`, `HTE_FETCH_MAX_BYTES` |
| **External APIs** | Service credentials and limits | `HTE_NCBI_EMAIL`, `HTE_GDELT_MAXRECORDS` |
| **Analysis Tuning** | Search result limits | `HTE_PUBMED_RETMAX`, `HTE_PUBMED_MAX_ABSTRACTS` |

---

### 🔌 External Service Integrations

#### `hawkins_truth_engine/external/rdap.py` — Domain Intelligence

**What it does**: Queries RDAP (Registration Data Access Protocol) for domain registration information.

**Why it's needed**: Domain age and status are strong credibility indicators. New domains are often used for misinformation campaigns.

**Key Function**: `rdap_domain(domain)` → Returns registration date, status, and other metadata

#### `hawkins_truth_engine/external/ncbi.py` — Medical Literature Search

**What it does**: Searches PubMed database for biomedical literature to verify health-related claims.

**Why it's needed**: Medical misinformation is particularly dangerous. PubMed provides authoritative scientific sources for verification.

**Key Functions**:
- `pubmed_esearch(term)` → Search for relevant articles
- `pubmed_esummary(pmids)` → Get article metadata  
- `pubmed_efetch_abstract(pmids)` → Fetch full abstracts

#### `hawkins_truth_engine/external/gdelt.py` — News Archive Search

**What it does**: Searches GDELT (Global Database of Events, Language, and Tone) for news coverage of claims.

**Why it's needed**: Cross-referencing claims against established news sources helps verify factual assertions.

**Key Function**: `gdelt_doc_search(query)` → Returns relevant news articles

---

### 🔄 How All Modules Work Together

1. **Input Processing** (`ingest.py`): Converts raw input into structured document
2. **Parallel Analysis**: Four analyzers examine different aspects simultaneously
   - `linguistic.py`: Writing patterns
   - `statistical.py`: Text structure  
   - `source_intel.py`: Source credibility
   - `claims.py`: Fact verification
3. **Decision Making** (`reasoning.py`): Combines evidence using explicit rules
4. **Explanation** (`explain.py`): Generates human-readable results
5. **Delivery** (`app.py`): Serves results via web interface or API

Each module is designed to be independent, testable, and explainable, contributing specific evidence to the overall credibility assessment.

---

## Evidence Item Structure

Every signal in the system is captured as an `EvidenceItem` ensuring full traceability:

```python
class EvidenceItem(BaseModel):
    id: str              # Unique identifier (e.g., "clickbait_punct", "young_domain")
    module: str          # Source module (e.g., "linguistic", "source_intel")
    weight: float        # Relative importance (0.0 - 1.0)
    severity: str        # "low" | "medium" | "high"
    value: float         # Numeric measurement (0.0 - 1.0)
    evidence: str        # Human-readable description
    pointers: Pointer    # Links to source text
    provenance: dict     # Metadata (API URLs, query parameters)
```

**Pointer Structure** (linking back to source):
```python
class Pointer(BaseModel):
    char_spans: list[CharSpan]  # Character positions in original text
    sentence_ids: list[int]     # Referenced sentence indices
    entity_ids: list[int]       # Referenced entity indices
```

**Example Evidence Item**:
```json
{
  "id": "young_domain",
  "module": "source_intel",
  "weight": 0.7,
  "severity": "high",
  "value": 0.8,
  "evidence": "Domain registered 45 days ago (< 90 day threshold)",
  "pointers": {"char_spans": [], "sentence_ids": [], "entity_ids": []},
  "provenance": {"rdap_url": "https://rdap.org/domain/example.com"}
}
```

---

## Deterministic Rule System

### Rule Definitions

The reasoning engine applies three explicit rules:

#### Rule R1: `R_LOW_SOURCE_HIGH_LING_LOW_SUPPORT`

**Trigger Conditions**:
- `source_trust_score < 0.35` AND
- `linguistic_risk_score > 0.65` AND
- `supported_count == 0` AND
- `unsupported_count + unverifiable_count >= 2`

**Effect**: Direction → `toward_fake`, minimum risk = 0.80

**Rationale**: Untrusted source + high linguistic risk + no claim support = strong fake signal

---

#### Rule R2: `R_MED_STRONG_CLAIM_NO_SUPPORT`

**Trigger Conditions**:
- `is_medical_topic == True` AND
- Strong claim detected without attribution AND
- `supported_count == 0`

**Effect**: Adds `high_harm_potential_medical` uncertainty flag

**Rationale**: Medical misinformation requires extra caution

---

#### Rule R3: `R_HIGH_SOURCE_LOW_RISK`

**Trigger Conditions**:
- `source_trust_score > 0.75` AND
- `linguistic_risk_score < 0.45` AND
- `statistical_risk_score < 0.45`

**Effect**: Direction → `toward_real`, maximum risk = 0.35

**Rationale**: Trusted source + low risk signals = credibility boost

---

### Scoring Algorithm

```python
# Step 1: Base risk from linguistic + statistical
base_risk = min(1.0, 0.55 * linguistic_risk + 0.45 * statistical_risk)

# Step 2: Source trust gate
if source_trust < 0.35:
    risk *= 1.25  # Amplify risk for untrusted sources
elif source_trust > 0.75:
    risk *= 0.85  # Dampen risk for trusted sources

# Step 3: Claim support adjustments
if supported_count >= 2:
    risk -= 0.20
if unverifiable_count >= 3:
    risk += 0.10
if unsupported_count >= 2:
    risk += 0.15

# Step 4: Rule overrides
if R1_triggered or R2_triggered:
    risk = max(risk, 0.80)
if R3_triggered:
    risk = min(risk, 0.35)

# Step 5: Final score
credibility_score = round(100 * (1 - risk))
```

### Confidence Calculation (Heuristic)

```python
agreement = 1 - abs(linguistic_risk - statistical_risk)
coverage = 0.6 + 0.2 * (unverifiable_count == 0) + 0.1 * (supported_count >= 1)
confidence = 0.35 + 0.35 * agreement + 0.30 * coverage

if uncertainty_flags:
    confidence = min(confidence, 0.75)  # Dampen if uncertain
```

**Important**: Confidence is explicitly marked as **uncalibrated** and serves as a heuristic indicator only.

---

## Verdict & World Label Mapping

### Tri-Verdict System

| Credibility Score | Verdict | Interpretation |
|-------------------|---------|----------------|
| ≥ 70 | `Likely Real` | Evidence suggests credible content |
| 40 - 69 | `Suspicious` | Mixed signals, requires human review |
| < 40 | `Likely Fake` | Multiple risk indicators detected |

### Binary World Label Mapping

The thematic "Real World" / "Upside Down" labels are mapped as follows:

| Verdict | WorldLabel |
|---------|------------|
| `Likely Real` | `Real World` |
| `Suspicious` | `Upside Down` |
| `Likely Fake` | `Upside Down` |

**Implementation** (from `reasoning.py`):
```python
world_label = WorldLabel.REAL_WORLD if verdict == Verdict.LIKELY_REAL else WorldLabel.UPSIDE_DOWN
```

**Rationale**: Binary classification is useful for quick triage while the tri-verdict provides nuance for detailed review.

---

## API Response Structure

### `AnalysisResponse` Schema

```python
class AnalysisResponse(BaseModel):
    document: Document              # Preprocessed input with all extracted features
    linguistic: LinguisticOutput    # Linguistic analysis results + evidence
    statistical: StatisticalOutput  # Statistical analysis results + evidence
    source_intel: SourceIntelOutput # Source credibility assessment + evidence
    claims: ClaimsOutput            # Claim extraction + corroboration results
    aggregation: AggregationOutput  # Final scores, verdict, world_label, reasoning_path
    explanation: VerdictExplanation # Human-readable explanation
```

### Key Fields in `AggregationOutput`

```json
{
  "credibility_score": 42,
  "verdict": "Suspicious",
  "world_label": "Upside Down",
  "confidence": 0.58,
  "uncertainty_flags": ["ncbi_unavailable"],
  "reasoning_path": [
    {
      "rule_id": "R_LOW_SOURCE_HIGH_LING_LOW_SUPPORT",
      "triggered": false,
      "conditions": "source_trust=0.45, linguistic_risk=0.62, supported=0, unsupported+unverifiable=1",
      "evidence_ids": ["clickbait_punct", "conspiracy_phrases", "claim:C1"]
    }
  ]
}
```

### Evidence ID Conventions

- Module evidence: Uses `EvidenceItem.id` (e.g., `young_domain`, `clickbait_punct`)
- Claim references: Formatted as `claim:C1`, `claim:C2`, etc.

---

## External Evidence Providers

### GDELT DOC API

**Purpose**: News article corroboration

**Endpoint**: `https://api.gdeltproject.org/api/v2/doc/doc`

**Parameters**:
- `query`: Search term (claim text)
- `mode`: `artlist`
- `format`: `json`
- `maxrecords`: Configurable (default: 25)
- `sort`: `hybridrel`

**Failure Handling**: Adds `gdelt_unavailable` uncertainty flag

---

### RDAP.org

**Purpose**: Domain registration metadata

**Endpoint**: `https://rdap.org/domain/{domain}`

**Data Extracted**:
- `events[].eventDate` where `eventAction == "registration"` → domain age
- `status` array → hold status detection

**Failure Handling**: Adds `rdap_unavailable` uncertainty flag

---

### NCBI E-utilities (PubMed)

**Purpose**: Biomedical literature corroboration for medical topics

**Endpoints**:
- `esearch.fcgi` — Search for PMIDs
- `esummary.fcgi` — Fetch article metadata
- `efetch.fcgi` — Fetch abstracts

**Base URL**: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`

**Parameters**:
- `tool`: Application identifier
- `email`: Contact email (recommended)
- `api_key`: For higher rate limits (optional)
- `retmax`: Max results per query

**Failure Handling**: Adds `ncbi_unavailable` uncertainty flag

---

### Tavily Search (Optional)

**Purpose**: General web corroboration when GDELT is sparse or for non-news sources

**Endpoint**: `https://api.tavily.com/search`

**Configuration**:
- `HTE_TAVILY_API_KEY` (optional) — if not set, Tavily is skipped
- `HTE_TAVILY_MAX_RESULTS` (optional) — max results per query
- `HTE_TAVILY_SEARCH_DEPTH` (optional) — `basic` or `advanced`

**Failure Handling**: Adds `tavily_unavailable` uncertainty flag

---

## Configuration & Environment Variables

### Required/Optional Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HTE_HTTP_TIMEOUT_SECS` | No | 20 | HTTP request timeout in seconds |
| `HTE_FETCH_MAX_BYTES` | No | 2000000 | Maximum bytes to download from URLs |
| `HTE_GDELT_MAXRECORDS` | No | 25 | Maximum GDELT search results |
| `HTE_NCBI_TOOL` | No | hawkins_truth_engine_poc | NCBI tool identifier |
| `HTE_NCBI_EMAIL` | Recommended | (empty) | Contact email for NCBI |
| `HTE_NCBI_API_KEY` | No | (empty) | NCBI API key for higher rate limits |
| `HTE_PUBMED_RETMAX` | No | 10 | Maximum PubMed search results |
| `HTE_PUBMED_MAX_ABSTRACTS` | No | 3 | Maximum abstracts to fetch per claim |
| `HTE_TAVILY_API_KEY` | No | (empty) | Tavily API key for optional web search corroboration |
| `HTE_TAVILY_MAX_RESULTS` | No | 5 | Maximum Tavily results per query |
| `HTE_TAVILY_SEARCH_DEPTH` | No | basic | Tavily search depth (`basic` or `advanced`) |

### Example `.env` File

```bash
HTE_HTTP_TIMEOUT_SECS=30
HTE_FETCH_MAX_BYTES=5000000
HTE_GDELT_MAXRECORDS=50
HTE_NCBI_EMAIL=your.email@example.com
HTE_NCBI_API_KEY=your_api_key_here
HTE_PUBMED_RETMAX=20
HTE_PUBMED_MAX_ABSTRACTS=5
```

---

## Security Considerations

### No Hard-Coded Secrets

- All credentials configured via environment variables
- `.env` file included in `.gitignore`
- No API keys or passwords in source code

### Input Validation

- All inputs validated via Pydantic models
- URL fetching bounded by `FETCH_MAX_BYTES`
- Request timeouts prevent hanging

### External Service Safety

- All external calls use HTTPS
- Failures recorded as uncertainty flags (not exceptions)
- No sensitive data sent to external services

### Content Safety

- System does not store user content
- No persistent database of analyzed content
- Results returned directly to requester

---

## Limitations & Extensions

### Current Limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Confidence not calibrated | Cannot interpret as probability | Clearly marked as heuristic |
| English-first claim extraction | Reduced accuracy for other languages | Language detection flags non-English |
| Evidence absence ≠ falsity | May mark valid claims as unverifiable | Conservative labeling, explicit documentation |
| External services may be incomplete | Missing corroboration | Uncertainty flags, confidence dampening |
| No image/video analysis | Cannot detect visual manipulation | Future extension opportunity |
| Heuristic NER | May miss or misclassify entities | Best-effort, not relied upon for scoring |

### Open-Ended Extensions

| Extension | Description | Difficulty |
|-----------|-------------|------------|
| **Confidence Calibration** | Train on labeled dataset to calibrate confidence scores | Medium |
| **Corpus Indexing** | Build local index of known credible/non-credible sources | Medium |
| **Optional NLP Extras** | spaCy NER, sentence-transformers for semantic similarity | Low |
| **Image Analysis** | Reverse image search, metadata extraction, manipulation detection | High |
| **Multi-language Support** | Extend claim extraction and corroboration to other languages | Medium |
| **Real-time Monitoring** | Stream processing for social media monitoring | High |
| **User Feedback Loop** | Allow reviewers to correct verdicts for model improvement | Medium |

---

## Quickstart Guide

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd hawkins-truth-engine

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# Unix/macOS:
source .venv/bin/activate

# 4. Install dependencies
python -m pip install -U pip
python -m pip install -e .
```

### Running the Application

```bash
# Option 1: Module execution
python -m hawkins_truth_engine.app

# Option 2: Console script (after install)
hawkins-truth-engine
```

### Accessing the Interface

- **Web UI**: http://127.0.0.1:8000
- **API Documentation**: http://127.0.0.1:8000/docs
- **OpenAPI Schema**: http://127.0.0.1:8000/openapi.json

### Example API Request

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "raw_text",
    "content": "BREAKING: Scientists discover miracle cure that Big Pharma does not want you to know about! This 100% natural remedy cures all diseases with no side effects. Doctors hate this one weird trick!"
  }'
```

### Example Response (Abbreviated)

```json
{
  "aggregation": {
    "credibility_score": 18,
    "verdict": "Likely Fake",
    "world_label": "Upside Down",
    "confidence": 0.72,
    "uncertainty_flags": [],
    "reasoning_path": [...]
  },
  "explanation": {
    "verdict_text": "World: Upside Down | Verdict: Likely Fake (72% confidence)",
    "evidence_bullets": [
      "High severity: Conspiracy phrase detected: 'big pharma'",
      "High severity: Medical claim without attribution: 'cures all diseases'",
      "Medium severity: Clickbait phrase detected: 'miracle'",
      "Medium severity: Urgency lexicon detected: 'breaking'"
    ]
  }
}
```

---

## FAQ

### General Questions

**Q: Does the Hawkins Truth Engine guarantee truth or prove something is fake?**

A: **No.** The system reports evidence-backed signals and produces an interpretable verdict with uncertainty flags. It is designed as a **credibility aid for human reviewers**, not an oracle of truth. It never treats missing evidence as proof of falsity.

---

**Q: Why "Real World" vs "Upside Down"?**

A: These are thematic labels inspired by the problem prompt (Hawkins/Stranger Things theme). They provide a binary classification for quick triage:
- **Real World** = `Likely Real` verdict (credibility score ≥ 70)
- **Upside Down** = `Suspicious` or `Likely Fake` verdict (credibility score < 70)

The underlying tri-verdict system provides more nuance for detailed review.

---

**Q: What does "unsupported" mean for a claim?**

A: A conservative internal label indicating:
1. The claim is strongly framed (uses certainty language)
2. No attribution to a named source was detected
3. No supporting evidence was returned from configured evidence providers (PubMed, GDELT)

**Important**: "Unsupported" does **NOT** mean "false." It means we couldn't find backing—the claim may still be true but poorly attributed or too recent for our sources.

---

### Technical Questions

**Q: What happens when external services (GDELT, PubMed, RDAP) fail?**

A: The system continues processing with available evidence:
1. The failure is recorded as an `uncertainty_flag` (e.g., `ncbi_unavailable`)
2. The confidence heuristic is dampened (capped at 0.75)
3. The reasoning path shows which evidence sources were unavailable
4. The verdict is still produced based on available signals

---

**Q: Why not use a machine learning classifier?**

A: Deliberate design choice for **explainability** and **transparency**:
- ML classifiers are black boxes—users can't understand why content was flagged
- Our deterministic rules can be audited, modified, and explained
- Evidence items with provenance allow verification of each signal
- No training data bias or distribution shift concerns

---

**Q: How is the confidence score calculated?**

A: The confidence is a **heuristic** (not calibrated probability) based on:
1. **Agreement** between linguistic and statistical risk scores (35% weight)
2. **Coverage** of evidence sources (30% weight)
3. Base confidence (35%)

If uncertainty flags are present, confidence is capped at 0.75. The score is explicitly marked as uncalibrated in all outputs.

---

**Q: Can I add custom rules to the reasoning engine?**

A: Yes. The rule system in `reasoning.py` is designed for extension:
1. Define new rule conditions in the `_evaluate_rules()` function
2. Add the rule to the `ReasoningStep` output
3. Integrate the rule effect into the scoring logic

---

**Q: Why are some claims marked "unverifiable"?**

A: Claims are labeled "unverifiable" when:
- The search query returned no results from evidence providers
- The results were ambiguous or didn't clearly match the claim
- The claim type is speculative or predictive (inherently unverifiable)

This is the most conservative label—it indicates uncertainty, not judgment.

---

### Usage Questions

**Q: What input types are supported?**

A: Three input types:
- `raw_text`: Direct text content (articles, posts, claims)
- `url`: Web page URL (automatically fetched and extracted)
- `social_post`: Social media content (treated as raw text with appropriate context)

---

**Q: How do I interpret the credibility score?**

A: The score ranges from 0-100:
- **70-100**: `Likely Real` — Multiple positive signals, few risk indicators
- **40-69**: `Suspicious` — Mixed signals, warrants human review
- **0-39**: `Likely Fake` — Multiple risk indicators detected

Higher scores indicate higher assessed credibility under this POC's rule set.

---

**Q: Is this suitable for production use?**

A: This is a **Proof of Concept (POC)** designed for:
- Educational demonstration of explainable credibility assessment
- Triage and prioritization of content for human review
- Research and experimentation with misinformation detection

For production deployment, consider:
- Confidence calibration on representative data
- Rate limiting and caching for external services
- Enhanced error handling and monitoring
- Legal review for content moderation use cases

---

## Evaluation Rubric Alignment

For academic or competition evaluation, this project maps to common criteria:

### Clarity of Problem Statement
- See [Problem Statement](#problem-statement) section
- Clear articulation of the misinformation challenge
- Relevance to real-world digital literacy issues

### Innovation & Uniqueness
- See [Innovation & Uniqueness](#innovation--uniqueness) section
- Evidence-first approach (not black-box classification)
- Multi-signal fusion with deterministic reasoning
- Full traceability through evidence ledger

### Technical Explanation
- See [System Architecture](#system-architecture) and [Module Reference](#module-reference)
- Complete pipeline documentation
- Every file and function documented
- Appropriate technology stack justified

### Presentation & Demo Readiness
- Web UI at `/` for live demonstration
- OpenAPI docs at `/docs` for API exploration
- Clear visual feedback (world label, confidence, evidence bullets)

---

## License

MIT License — see [LICENSE](LICENSE) file.

---

## Repository Structure

```
hawkins-truth-engine/
├── README.md                           # This file
├── LICENSE                             # MIT License
├── pyproject.toml                      # Project metadata and dependencies
├── .gitignore                          # Git ignore patterns
└── hawkins_truth_engine/
    ├── __init__.py                     # Package version
    ├── app.py                          # FastAPI application + UI
    ├── config.py                       # Environment variable configuration
    ├── schemas.py                      # Pydantic models (27 classes)
    ├── ingest.py                       # Document preprocessing pipeline
    ├── utils.py                        # Text processing utilities
    ├── reasoning.py                    # Deterministic rule engine
    ├── explain.py                      # Explanation generator
    ├── analyzers/
    │   ├── __init__.py
    │   ├── linguistic.py               # Linguistic pattern analyzer
    │   ├── statistical.py              # Statistical pattern analyzer
    │   ├── source_intel.py             # Source credibility analyzer
    │   └── claims.py                   # Claim extraction & corroboration
    └── external/
        ├── __init__.py
        ├── rdap.py                     # RDAP domain lookup client
        ├── ncbi.py                     # PubMed E-utilities client
        └── gdelt.py                    # GDELT DOC API client
```

---

*Built for the Hawkins community—because the truth is out there, and it shouldn't require supernatural powers to find it.*
