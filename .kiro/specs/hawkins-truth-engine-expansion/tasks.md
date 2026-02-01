# Implementation Plan: Hawkins Truth Engine Expansion

## Overview

This implementation plan expands the Hawkins Truth Engine with three core architectural components: Claim Graph (Stage 3), Evidence Graph (Stage 5), and Confidence Calibration (Stage 8). The implementation follows a 5-phase approach building incrementally on the existing ~1,400 lines of Python code while maintaining API compatibility.

## Tasks

- [ ] 1. Set up graph infrastructure and core schemas
  - [x] 1.1 Create graph module directory structure
    - Create `hawkins_truth_engine/graph/` directory
    - Create `hawkins_truth_engine/calibration/` directory
    - Add `__init__.py` files for proper Python packaging
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 1.2 Implement core graph data schemas
    - Add GraphNode, GraphEdge, ClaimGraph, EvidenceGraph schemas to schemas.py
    - Implement Pydantic validation with proper field constraints
    - Add JSON serialization support and custom validators
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.7_

  - [ ]* 1.3 Write property test for graph schema validation
    - **Property 4: Graph Schema Validation**
    - **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.7**

- [ ] 2. Implement Claim Graph construction
  - [x] 2.1 Create claim_graph.py module
    - Implement build_claim_graph() function
    - Create node creation functions (sources, claims, entities)
    - Implement relationship edge creation logic
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [x] 2.2 Implement graph construction algorithms
    - Add entity-claim relationship detection using text overlap
    - Implement claim-source attribution using existing attribution detection
    - Add source-claim containment relationship creation
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [ ]* 2.3 Write property test for graph construction completeness
    - **Property 1: Graph Construction Completeness**
    - **Validates: Requirements 1.1, 1.5, 1.6**

  - [ ]* 2.4 Write property test for graph relationship consistency
    - **Property 2: Graph Relationship Consistency**
    - **Validates: Requirements 1.2, 1.3, 1.4, 1.7**

  - [ ]* 2.5 Write property test for graph ID uniqueness
    - **Property 5: Graph ID Uniqueness**
    - **Validates: Requirements 3.6**

- [ ] 3. Implement Evidence Graph construction
  - [x] 3.1 Create evidence_graph.py module
    - Implement build_evidence_graph() function
    - Add claim similarity calculation using text analysis
    - Implement evidence relationship determination logic
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 3.2 Implement evidence relationship algorithms
    - Add SUPPORTS edge creation for claims with consistent corroboration
    - Add CONTRADICTS edge creation for conflicting evidence
    - Add RELATES_TO edge creation for topically similar claims
    - Calculate edge weights based on similarity and evidence strength
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ]* 3.3 Write property test for evidence graph relationship determination
    - **Property 3: Evidence Graph Relationship Determination**
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4**

  - [ ]* 3.4 Write property test for evidence graph bidirectional queries
    - **Property 14: Evidence Graph Bidirectional Queries**
    - **Validates: Requirements 2.5, 2.6**

- [ ] 4. Checkpoint - Ensure graph construction tests pass
  - Ensure all graph construction tests pass, ask the user if questions arise.

- [ ] 5. Implement Confidence Calibration system
  - [x] 5.1 Create calibration/model.py module
    - Implement CalibrationDataPoint schema
    - Create ConfidenceCalibrator class with Platt scaling support
    - Add isotonic regression calibration method
    - Implement model persistence and loading functionality
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.7, 6.1, 6.5_

  - [x] 5.2 Implement calibration data management
    - Add support for loading datasets from JSON and CSV formats
    - Implement data validation and train/validation splitting
    - Add model versioning and rollback capabilities
    - Implement training metrics logging
    - _Requirements: 6.2, 6.3, 6.4, 6.6, 6.7_

  - [ ]* 5.3 Write property test for calibration model functionality
    - **Property 7: Calibration Model Functionality**
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.7**

  - [ ]* 5.4 Write property test for calibration quality validation
    - **Property 8: Calibration Quality Validation**
    - **Validates: Requirements 5.6**

  - [ ]* 5.5 Write property test for calibration data management
    - **Property 9: Calibration Data Management**
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7**

- [ ] 6. Implement graph query and analysis interface
  - [ ] 6.1 Create graph query interface
    - Implement GraphQueryInterface class
    - Add methods for finding claims by source, entities by claim
    - Add methods for finding supporting/contradicting claims
    - Implement graph metrics calculation (centrality, density)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [ ] 6.2 Add graph export functionality
    - Implement export methods for JSON, GraphML, and DOT formats
    - Add subgraph extraction based on node/edge filters
    - Implement graph traversal methods for relationship exploration
    - _Requirements: 8.7, 8.5, 8.6_

  - [ ]* 6.3 Write property test for graph query functionality
    - **Property 11: Graph Query Functionality**
    - **Validates: Requirements 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7**

- [ ] 7. Integrate graph components with existing modules
  - [ ] 7.1 Modify claims.py for graph integration
    - Add automatic Claim_Graph population when claims are extracted
    - Ensure ClaimItem compatibility is maintained
    - Add graph data to ClaimsOutput schema
    - _Requirements: 4.1, 2.7_

  - [ ] 7.2 Modify source_intel.py for graph integration
    - Add automatic source node creation when sources are identified
    - Integrate with existing source trust scoring
    - _Requirements: 4.2_

  - [ ] 7.3 Modify reasoning.py for graph and calibration integration
    - Add Evidence_Graph updates during evidence processing
    - Integrate confidence calibration with existing heuristic confidence
    - Update AggregationOutput schema with graph and calibration data
    - _Requirements: 4.3, 5.4_

  - [ ] 7.4 Modify explain.py for graph insights
    - Add graph-based insights to explanation generation
    - Include relationship information in evidence bullets
    - _Requirements: 4.4_

  - [ ]* 7.5 Write property test for integration behavior consistency
    - **Property 6: Integration Behavior Consistency**
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.7**

  - [ ]* 7.6 Write property test for backward compatibility preservation
    - **Property 15: Backward Compatibility Preservation**
    - **Validates: Requirements 2.7**

- [ ] 8. Checkpoint - Ensure integration tests pass
  - Ensure all integration tests pass, ask the user if questions arise.

- [ ] 9. Add API endpoints and response extensions
  - [ ] 9.1 Extend existing API responses
    - Add optional graph data to AnalysisResponse schema
    - Ensure existing API response formats are maintained
    - Add configuration options to enable/disable graph features
    - _Requirements: 4.5, 9.5_

  - [ ] 9.2 Create new graph-specific API endpoints
    - Add /graphs endpoint for retrieving graph data
    - Add /calibration endpoint for calibration model management
    - Implement proper error handling and validation
    - _Requirements: 4.7_

  - [ ]* 9.3 Write unit tests for API endpoints
    - Test new endpoints with various input scenarios
    - Test backward compatibility of existing endpoints
    - Test error handling and validation

- [ ] 10. Implement performance optimizations and error handling
  - [ ] 10.1 Add performance optimizations
    - Implement graph construction time limits and memory constraints
    - Add query caching for frequently accessed graph operations
    - Add progress indicators for long-running operations
    - _Requirements: 9.4, 7.7_

  - [ ] 10.2 Implement comprehensive error handling
    - Add graceful degradation when graph construction fails
    - Implement fallback behavior for corrupted calibration models
    - Add timeout handling for graph queries with partial results
    - Add graph integrity validation and inconsistency reporting
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7_

  - [ ]* 10.3 Write property test for concurrent operation safety
    - **Property 12: Concurrent Operation Safety**
    - **Validates: Requirements 9.3, 9.4, 9.5, 9.7**

  - [ ]* 10.4 Write property test for error resilience and graceful degradation
    - **Property 13: Error Resilience and Graceful Degradation**
    - **Validates: Requirements 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7**

- [ ] 11. Final integration and validation
  - [ ] 11.1 Perform end-to-end testing
    - Test complete pipeline with graph features enabled
    - Test pipeline with graph features disabled for performance comparison
    - Validate memory usage and performance benchmarks
    - _Requirements: 9.1, 9.2, 9.6_

  - [ ] 11.2 Update configuration and documentation
    - Add environment variables for graph and calibration configuration
    - Update API documentation with new endpoints and response formats
    - Add usage examples for graph functionality
    - _Requirements: 9.5_

  - [ ]* 11.3 Write property test for graph construction algorithm correctness
    - **Property 10: Graph Construction Algorithm Correctness**
    - **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.7**

- [ ] 12. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate universal correctness properties from the design document
- Unit tests validate specific examples and integration scenarios
- The implementation maintains backward compatibility while adding new graph capabilities
- Performance requirements are validated through benchmarking rather than unit tests
- Error handling ensures the system remains stable even when graph components fail