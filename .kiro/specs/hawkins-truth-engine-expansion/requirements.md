# Requirements Document

## Introduction

This document specifies the requirements for expanding the Hawkins Truth Engine with three core architectural components: Claim Graph (Stage 3), Evidence Graph (Stage 5), and Confidence Calibration (Stage 8). These additions will enhance the existing evidence-first credibility reasoning system by introducing graph-based relationship modeling and calibrated confidence estimates.

The expansion builds upon the existing Python FastAPI application (~1,400 lines across 15+ modules) while maintaining API compatibility and the deterministic reasoning approach.

## Glossary

- **Claim_Graph**: Graph structure representing relationships between sources, claims, and entities
- **Evidence_Graph**: Graph structure representing evidential relationships between claims
- **Confidence_Calibrator**: Machine learning component that maps heuristic confidence scores to calibrated probabilities
- **Graph_Node**: Vertex in a graph representing a Source, Claim, or Entity
- **Graph_Edge**: Directed connection between nodes with relationship type and weight
- **Platt_Scaling**: Sigmoid-based calibration method for converting scores to probabilities
- **Isotonic_Regression**: Non-parametric calibration method preserving monotonicity
- **Calibration_Dataset**: Labeled data used to train confidence calibration models

## Requirements

### Requirement 1: Claim Graph Implementation

**User Story:** As a credibility analyst, I want to visualize relationships between sources, claims, and entities, so that I can understand the provenance and interconnections of information.

#### Acceptance Criteria

1. WHEN the system processes a document, THE Claim_Graph SHALL create nodes for each identified source, claim, and entity
2. WHEN a claim mentions an entity, THE Claim_Graph SHALL create a MENTIONS edge between the claim node and entity node
3. WHEN a claim is attributed to a source, THE Claim_Graph SHALL create an ATTRIBUTED_TO edge between the claim node and source node
4. WHEN a source contains multiple claims, THE Claim_Graph SHALL create FROM_SOURCE edges between the source node and each claim node
5. THE Claim_Graph SHALL assign unique identifiers to all nodes and edges
6. THE Claim_Graph SHALL store node metadata including text, type, and confidence scores
7. THE Claim_Graph SHALL store edge metadata including relationship strength and provenance information

### Requirement 2: Evidence Graph Implementation

**User Story:** As a fact-checker, I want to see how claims support or contradict each other, so that I can assess the consistency and reliability of information.

#### Acceptance Criteria

1. WHEN two claims have supporting evidence relationships, THE Evidence_Graph SHALL create a SUPPORTS edge between them
2. WHEN two claims have contradictory evidence relationships, THE Evidence_Graph SHALL create a CONTRADICTS edge between them
3. WHEN two claims have topical relationships without clear support/contradiction, THE Evidence_Graph SHALL create a RELATES_TO edge between them
4. THE Evidence_Graph SHALL calculate edge weights based on evidence strength and similarity scores
5. THE Evidence_Graph SHALL support bidirectional relationship queries between any two claims
6. THE Evidence_Graph SHALL maintain provenance tracking for all relationship determinations
7. THE Evidence_Graph SHALL integrate with existing ClaimItem structures without breaking changes

### Requirement 3: Graph Data Models

**User Story:** As a system architect, I want standardized graph data structures, so that graph operations are consistent and extensible.

#### Acceptance Criteria

1. THE System SHALL define a GraphNode schema with id, type, text, metadata, and timestamps
2. THE System SHALL define a GraphEdge schema with source_id, target_id, relationship_type, weight, and provenance
3. THE System SHALL define a ClaimGraph schema containing nodes and edges collections
4. THE System SHALL define an EvidenceGraph schema containing claim relationships
5. THE System SHALL validate all graph structures using Pydantic schemas
6. THE System SHALL ensure graph node IDs are globally unique within each graph type
7. THE System SHALL support JSON serialization for all graph structures

### Requirement 4: Graph Integration Points

**User Story:** As a developer, I want graph functionality integrated with existing modules, so that the system maintains backward compatibility while adding new capabilities.

#### Acceptance Criteria

1. WHEN claims.py extracts claims, THE System SHALL automatically populate the Claim_Graph with claim nodes
2. WHEN source_intel.py identifies sources, THE System SHALL create source nodes in the Claim_Graph
3. WHEN reasoning.py processes evidence, THE System SHALL update the Evidence_Graph with relationship edges
4. WHEN explain.py generates explanations, THE System SHALL include graph-based insights in the output
5. THE System SHALL maintain existing API response formats while adding optional graph data
6. THE System SHALL ensure graph operations do not impact existing performance benchmarks
7. THE System SHALL provide graph data through new optional API endpoints

### Requirement 5: Confidence Calibration Implementation

**User Story:** As a credibility analyst, I want calibrated confidence scores that represent actual probabilities, so that I can make better-informed decisions about content reliability.

#### Acceptance Criteria

1. THE Confidence_Calibrator SHALL implement Platt scaling for sigmoid-based calibration
2. THE Confidence_Calibrator SHALL implement isotonic regression for non-parametric calibration
3. WHEN training data is available, THE System SHALL fit calibration models to map heuristic scores to probabilities
4. WHEN making predictions, THE System SHALL apply the trained calibration model to raw confidence scores
5. THE System SHALL support both calibration methods with configurable selection
6. THE System SHALL validate calibration quality using reliability diagrams and Brier scores
7. THE System SHALL gracefully fallback to heuristic confidence when calibration models are unavailable

### Requirement 6: Calibration Data Management

**User Story:** As a machine learning engineer, I want to manage calibration datasets, so that confidence models can be trained and updated with new data.

#### Acceptance Criteria

1. THE System SHALL define a CalibrationDataPoint schema with features, heuristic_confidence, and true_label
2. THE System SHALL support loading calibration datasets from JSON and CSV formats
3. THE System SHALL validate calibration data for completeness and consistency
4. THE System SHALL split calibration data into training and validation sets
5. THE System SHALL persist trained calibration models to disk for reuse
6. THE System SHALL support model versioning and rollback capabilities
7. THE System SHALL log calibration training metrics and model performance

### Requirement 7: Graph Construction Algorithms

**User Story:** As a system designer, I want efficient graph construction algorithms, so that graph building scales with document complexity.

#### Acceptance Criteria

1. THE System SHALL extract entities from claims using existing entity recognition
2. THE System SHALL determine claim-entity relationships using text overlap and semantic similarity
3. THE System SHALL identify claim-source attributions using existing attribution detection
4. THE System SHALL calculate evidence relationships using claim similarity and external corroboration results
5. THE System SHALL assign relationship weights based on confidence scores and evidence strength
6. THE System SHALL optimize graph construction for documents with up to 50 claims and 100 entities
7. THE System SHALL provide progress indicators for long-running graph construction operations

### Requirement 8: Graph Query and Analysis

**User Story:** As a researcher, I want to query and analyze graph structures, so that I can extract insights about information networks.

#### Acceptance Criteria

1. THE System SHALL support finding all claims attributed to a specific source
2. THE System SHALL support finding all entities mentioned in a specific claim
3. THE System SHALL support finding all claims that support or contradict a given claim
4. THE System SHALL calculate graph metrics including node centrality and edge density
5. THE System SHALL support subgraph extraction based on node or edge filters
6. THE System SHALL provide graph traversal methods for relationship exploration
7. THE System SHALL export graph data in standard formats (GraphML, JSON, DOT)

### Requirement 9: Performance and Scalability

**User Story:** As a system administrator, I want the expanded system to maintain performance, so that response times remain acceptable with the new graph features.

#### Acceptance Criteria

1. THE System SHALL complete graph construction within 2x the current document processing time
2. THE System SHALL limit memory usage for graph structures to 50MB per document
3. THE System SHALL support concurrent graph operations without race conditions
4. THE System SHALL cache frequently accessed graph queries for improved performance
5. THE System SHALL provide configuration options to disable graph features for performance-critical deployments
6. THE System SHALL maintain existing API response times when graph features are disabled
7. THE System SHALL log performance metrics for graph operations

### Requirement 10: Error Handling and Robustness

**User Story:** As a system operator, I want robust error handling for graph operations, so that failures in graph processing don't break the core credibility analysis.

#### Acceptance Criteria

1. WHEN graph construction fails, THE System SHALL continue with core analysis and log the error
2. WHEN calibration models are corrupted, THE System SHALL fallback to heuristic confidence with appropriate warnings
3. WHEN graph queries timeout, THE System SHALL return partial results with timeout indicators
4. THE System SHALL validate graph integrity and report inconsistencies
5. THE System SHALL handle missing or malformed graph data gracefully
6. THE System SHALL provide detailed error messages for graph operation failures
7. THE System SHALL maintain system stability even when graph components encounter exceptions