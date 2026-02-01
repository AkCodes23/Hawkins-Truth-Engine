# Design Document

## Overview

This design document specifies the architecture for expanding the Hawkins Truth Engine with three core components: Claim Graph (Stage 3), Evidence Graph (Stage 5), and Confidence Calibration (Stage 8). The expansion builds upon the existing Python FastAPI application while maintaining backward compatibility and the deterministic reasoning approach.

The design introduces graph-based relationship modeling to capture complex interactions between sources, claims, and entities, along with machine learning-based confidence calibration to provide probabilistically meaningful confidence scores.

## Architecture

### High-Level Architecture

The expanded system maintains the existing pipeline while adding three new processing stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXISTING PIPELINE                                  │
│  Ingest → Multi-Signal Analysis → Reasoning & Aggregation → Explanation     │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NEW GRAPH LAYER                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │   Claim Graph   │  │  Evidence Graph │  │   Confidence Calibration   │  │
│  │   (Stage 3)     │  │   (Stage 5)     │  │       (Stage 8)             │  │
│  │                 │  │                 │  │                             │  │
│  │ • Source nodes  │  │ • SUPPORTS      │  │ • Platt scaling             │  │
│  │ • Claim nodes   │  │ • CONTRADICTS   │  │ • Isotonic regression       │  │
│  │ • Entity nodes  │  │ • RELATES_TO    │  │ • Calibrated probabilities  │  │
│  │ • Relationships │  │ • Edge weights  │  │ • Model persistence         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Integration Points

The new components integrate with existing modules at specific points:

1. **claims.py** → Populates Claim Graph with claim and entity nodes
2. **source_intel.py** → Creates source nodes and attribution edges
3. **reasoning.py** → Updates Evidence Graph and applies calibration
4. **explain.py** → Incorporates graph insights into explanations

## Components and Interfaces

### Graph Data Structures

#### GraphNode Schema
```python
class GraphNode(BaseModel):
    id: str                           # Unique identifier (e.g., "claim:C1", "source:S1", "entity:E1")
    type: Literal["source", "claim", "entity"]
    text: str                         # Display text for the node
    metadata: dict[str, Any]          # Type-specific metadata
    confidence: float | None = None   # Optional confidence score
    created_at: datetime              # Timestamp for provenance
    updated_at: datetime              # Last modification timestamp
```

#### GraphEdge Schema
```python
class GraphEdge(BaseModel):
    id: str                           # Unique edge identifier
    source_id: str                    # Source node ID
    target_id: str                    # Target node ID
    relationship_type: Literal["MENTIONS", "ATTRIBUTED_TO", "FROM_SOURCE", 
                              "SUPPORTS", "CONTRADICTS", "RELATES_TO"]
    weight: float = Field(ge=0.0, le=1.0)  # Relationship strength
    provenance: dict[str, Any]        # Evidence for this relationship
    created_at: datetime              # Timestamp for provenance
```

#### ClaimGraph Schema
```python
class ClaimGraph(BaseModel):
    nodes: dict[str, GraphNode]       # Node ID → Node mapping
    edges: dict[str, GraphEdge]       # Edge ID → Edge mapping
    metadata: dict[str, Any]          # Graph-level metadata
    created_at: datetime              # Graph creation timestamp
```

#### EvidenceGraph Schema
```python
class EvidenceGraph(BaseModel):
    claim_nodes: dict[str, str]       # Claim ID → Node ID mapping
    edges: dict[str, GraphEdge]       # Edge ID → Edge mapping
    similarity_threshold: float       # Threshold for RELATES_TO edges
    metadata: dict[str, Any]          # Graph-level metadata
    created_at: datetime              # Graph creation timestamp
```

### Claim Graph Module

#### File: `hawkins_truth_engine/graph/claim_graph.py`

**Core Functions:**

```python
def build_claim_graph(document: Document, claims_output: ClaimsOutput, 
                     source_output: SourceIntelOutput) -> ClaimGraph:
    """Constructs claim graph from analysis outputs."""
    
def create_source_nodes(document: Document, source_output: SourceIntelOutput) -> list[GraphNode]:
    """Creates source nodes from document metadata and source intelligence."""
    
def create_claim_nodes(claims_output: ClaimsOutput) -> list[GraphNode]:
    """Creates claim nodes from extracted claims."""
    
def create_entity_nodes(document: Document) -> list[GraphNode]:
    """Creates entity nodes from document entities."""
    
def create_relationship_edges(nodes: dict[str, GraphNode], 
                            document: Document, 
                            claims_output: ClaimsOutput) -> list[GraphEdge]:
    """Creates edges representing relationships between nodes."""
```

**Node Creation Logic:**

- **Source Nodes**: Created from document metadata (domain, author) and source intelligence
- **Claim Nodes**: One node per ClaimItem with claim text and support status
- **Entity Nodes**: Created from document entities with normalization
- **Relationship Edges**: 
  - MENTIONS: Claim → Entity (based on text overlap)
  - ATTRIBUTED_TO: Claim → Source (from attribution detection)
  - FROM_SOURCE: Source → Claim (document provenance)

### Evidence Graph Module

#### File: `hawkins_truth_engine/graph/evidence_graph.py`

**Core Functions:**

```python
def build_evidence_graph(claims_output: ClaimsOutput, 
                        external_corroboration: dict) -> EvidenceGraph:
    """Constructs evidence graph from claims and corroboration data."""
    
def calculate_claim_similarity(claim1: ClaimItem, claim2: ClaimItem) -> float:
    """Calculates semantic similarity between two claims."""
    
def determine_evidence_relationship(claim1: ClaimItem, claim2: ClaimItem,
                                  similarity: float,
                                  corroboration: dict) -> str | None:
    """Determines relationship type between claims based on evidence."""
    
def create_evidence_edges(claims: list[ClaimItem], 
                         corroboration: dict) -> list[GraphEdge]:
    """Creates evidence relationship edges between claims."""
```

**Relationship Determination Logic:**

- **SUPPORTS**: Claims with similar text and consistent external corroboration
- **CONTRADICTS**: Claims with similar topics but conflicting support status
- **RELATES_TO**: Claims with moderate similarity but unclear evidence relationship

**Edge Weight Calculation:**
```python
weight = (similarity_score * 0.4 + 
          evidence_strength * 0.4 + 
          corroboration_confidence * 0.2)
```

### Confidence Calibration Module

#### File: `hawkins_truth_engine/calibration/model.py`

**Core Classes:**

```python
class CalibrationDataPoint(BaseModel):
    features: dict[str, float]        # Input features (linguistic_risk, etc.)
    heuristic_confidence: float       # Original confidence score
    true_label: bool                  # Ground truth label
    verdict: str                      # Original verdict
    metadata: dict[str, Any]          # Additional context

class ConfidenceCalibrator:
    """Handles confidence calibration using Platt scaling or isotonic regression."""
    
    def __init__(self, method: Literal["platt", "isotonic"] = "platt"):
        self.method = method
        self.calibrator = None
        self.is_fitted = False
    
    def fit(self, calibration_data: list[CalibrationDataPoint]) -> None:
        """Trains calibration model on labeled data."""
    
    def predict_proba(self, heuristic_confidence: float) -> float:
        """Converts heuristic confidence to calibrated probability."""
    
    def evaluate(self, test_data: list[CalibrationDataPoint]) -> dict[str, float]:
        """Evaluates calibration quality using reliability metrics."""
```

**Implementation Details:**

- **Platt Scaling**: Uses scikit-learn's `CalibratedClassifierCV` with sigmoid method
- **Isotonic Regression**: Uses scikit-learn's isotonic regression for non-parametric calibration
- **Model Persistence**: Saves trained models using joblib for reuse
- **Fallback Behavior**: Returns heuristic confidence when calibration unavailable

## Data Models

### Extended Schemas

The existing schemas are extended with new graph-related fields:

```python
# Extended AggregationOutput
class AggregationOutput(BaseModel):
    # ... existing fields ...
    claim_graph: ClaimGraph | None = None
    evidence_graph: EvidenceGraph | None = None
    calibrated_confidence: float | None = None
    calibration_method: str | None = None

# Extended AnalysisResponse  
class AnalysisResponse(BaseModel):
    # ... existing fields ...
    graphs: GraphData | None = None

class GraphData(BaseModel):
    claim_graph: ClaimGraph
    evidence_graph: EvidenceGraph
    graph_metrics: dict[str, Any]
```

### Graph Query Interface

```python
class GraphQueryInterface:
    """Provides query methods for graph structures."""
    
    def find_claims_by_source(self, source_id: str) -> list[GraphNode]:
        """Returns all claims attributed to a source."""
    
    def find_entities_in_claim(self, claim_id: str) -> list[GraphNode]:
        """Returns all entities mentioned in a claim."""
    
    def find_supporting_claims(self, claim_id: str) -> list[GraphNode]:
        """Returns claims that support the given claim."""
    
    def find_contradicting_claims(self, claim_id: str) -> list[GraphNode]:
        """Returns claims that contradict the given claim."""
    
    def calculate_node_centrality(self, graph: ClaimGraph) -> dict[str, float]:
        """Calculates centrality metrics for nodes."""
    
    def export_graph(self, graph: ClaimGraph | EvidenceGraph, 
                    format: Literal["json", "graphml", "dot"]) -> str:
        """Exports graph in specified format."""
```

## Error Handling

### Graceful Degradation Strategy

The system implements graceful degradation to ensure core functionality remains available:

1. **Graph Construction Failures**: Log errors and continue with core analysis
2. **Calibration Model Unavailable**: Fall back to heuristic confidence with warning
3. **External Service Timeouts**: Mark uncertainty flags and proceed
4. **Memory Constraints**: Implement graph size limits and sampling

### Error Recovery Mechanisms

```python
class GraphOperationError(Exception):
    """Custom exception for graph operation failures."""
    pass

def safe_graph_operation(operation: Callable, fallback_value: Any = None) -> Any:
    """Wrapper for safe graph operations with error handling."""
    try:
        return operation()
    except Exception as e:
        logger.warning(f"Graph operation failed: {e}")
        return fallback_value
```

## Testing Strategy

### Dual Testing Approach

The system employs both unit testing and property-based testing for comprehensive coverage:

**Unit Testing Focus:**
- Specific graph construction scenarios
- Calibration model training and prediction
- Integration points with existing modules
- Error handling and edge cases

**Property-Based Testing Focus:**
- Universal properties that hold across all inputs
- Graph invariants and consistency checks
- Calibration model correctness properties
- Performance characteristics under load

**Property-Based Testing Configuration:**
- Minimum 100 iterations per property test
- Uses Hypothesis library for Python property testing
- Each property test references its design document property
- Tag format: **Feature: hawkins-truth-engine-expansion, Property {number}: {property_text}**

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Graph Construction Completeness
*For any* document with identified sources, claims, and entities, the Claim_Graph should contain corresponding nodes with unique IDs and all required metadata fields (text, type, timestamps).
**Validates: Requirements 1.1, 1.5, 1.6**

### Property 2: Graph Relationship Consistency  
*For any* claim-entity mention relationship, claim-source attribution, or source-claim containment, the Claim_Graph should contain the corresponding edges (MENTIONS, ATTRIBUTED_TO, FROM_SOURCE) with proper provenance metadata.
**Validates: Requirements 1.2, 1.3, 1.4, 1.7**

### Property 3: Evidence Graph Relationship Determination
*For any* pair of claims with evidence relationships, the Evidence_Graph should create the appropriate edge type (SUPPORTS, CONTRADICTS, RELATES_TO) based on evidence strength and similarity scores.
**Validates: Requirements 2.1, 2.2, 2.3, 2.4**

### Property 4: Graph Schema Validation
*For any* graph structure (GraphNode, GraphEdge, ClaimGraph, EvidenceGraph), the object should conform to its Pydantic schema and support JSON serialization round-trips.
**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.7**

### Property 5: Graph ID Uniqueness
*For any* graph of the same type, all node IDs and edge IDs should be globally unique within that graph instance.
**Validates: Requirements 3.6**

### Property 6: Integration Behavior Consistency
*For any* document processing pipeline execution, when claims are extracted, sources identified, or evidence processed, the corresponding graph structures should be automatically populated without breaking existing functionality.
**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.7**

### Property 7: Calibration Model Functionality
*For any* calibration method (Platt scaling or isotonic regression), when training data is provided, the system should fit a model that maps heuristic confidence scores to calibrated probabilities and gracefully fallback when unavailable.
**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.7**

### Property 8: Calibration Quality Validation
*For any* trained calibration model, the system should calculate reliability metrics (Brier scores) and validate model quality before deployment.
**Validates: Requirements 5.6**

### Property 9: Calibration Data Management
*For any* calibration dataset, the system should validate data completeness, support multiple formats (JSON, CSV), and enable model persistence with versioning.
**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7**

### Property 10: Graph Construction Algorithm Correctness
*For any* document with claims and entities, the graph construction algorithms should correctly determine relationships using text overlap, semantic similarity, and attribution detection.
**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.7**

### Property 11: Graph Query Functionality
*For any* graph structure, the query interface should support finding related nodes (claims by source, entities by claim, supporting/contradicting claims) and calculating graph metrics accurately.
**Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7**

### Property 12: Concurrent Operation Safety
*For any* concurrent graph operations, the system should maintain data consistency without race conditions and support query caching.
**Validates: Requirements 9.3, 9.4, 9.5, 9.7**

### Property 13: Error Resilience and Graceful Degradation
*For any* graph operation failure, calibration model corruption, or timeout condition, the system should continue core analysis, provide appropriate fallbacks, and maintain system stability.
**Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7**

### Property 14: Evidence Graph Bidirectional Queries
*For any* two claims in the Evidence_Graph, relationship queries should work in both directions and maintain provenance tracking for all determinations.
**Validates: Requirements 2.5, 2.6**

### Property 15: Backward Compatibility Preservation
*For any* existing ClaimItem structure or API response format, the expanded system should maintain compatibility while optionally adding graph data.
**Validates: Requirements 2.7**

## Testing Strategy

### Dual Testing Approach

The system employs both unit testing and property-based testing for comprehensive coverage:

**Unit Testing Focus:**
- Specific graph construction scenarios with known inputs and outputs
- Calibration model training with sample datasets
- Integration points between new and existing modules
- Error handling for specific failure modes
- API endpoint functionality with mock data

**Property-Based Testing Focus:**
- Universal properties that hold across all document types and sizes
- Graph invariants (uniqueness, consistency, completeness)
- Calibration model correctness across different data distributions
- Performance characteristics under varying loads
- Concurrent operation safety with randomized access patterns

**Property-Based Testing Configuration:**
- Uses Hypothesis library for Python property testing
- Minimum 100 iterations per property test due to randomization
- Each property test references its design document property
- Tag format: **Feature: hawkins-truth-engine-expansion, Property {number}: {property_text}**
- Custom generators for documents, claims, and graph structures
- Shrinking enabled to find minimal failing examples

**Testing Libraries and Tools:**
- **pytest**: Primary testing framework
- **Hypothesis**: Property-based testing library
- **pytest-mock**: Mocking external dependencies
- **pytest-asyncio**: Testing async operations
- **memory-profiler**: Memory usage validation
- **scikit-learn**: Calibration model testing utilities

**Performance Testing:**
- Graph construction time limits (2x baseline processing time)
- Memory usage constraints (50MB per document)
- Concurrent operation throughput
- API response time maintenance
- Cache hit rate optimization

**Integration Testing:**
- End-to-end pipeline with graph features enabled/disabled
- External service integration (PubMed, GDELT, RDAP)
- Database persistence and model loading
- API compatibility with existing clients
- Error propagation and recovery scenarios