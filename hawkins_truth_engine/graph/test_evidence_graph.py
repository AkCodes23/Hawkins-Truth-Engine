"""
Unit tests for evidence graph construction module.
"""

from datetime import datetime
from hawkins_truth_engine.schemas import (
    ClaimsOutput, ClaimItem, Pointer
)
from hawkins_truth_engine.graph.evidence_graph import (
    calculate_claim_similarity, determine_evidence_relationship,
    create_evidence_edges, build_evidence_graph
)


def test_calculate_claim_similarity():
    """Test claim similarity calculation."""
    # Test identical claims
    claim1 = ClaimItem(
        id="claim1", text="Climate change is real", type="factual", support="supported"
    )
    claim2 = ClaimItem(
        id="claim2", text="Climate change is real", type="factual", support="supported"
    )
    
    # Same ID should return 1.0
    similarity = calculate_claim_similarity(claim1, claim1)
    assert similarity == 1.0
    
    # Different IDs but same text should have high similarity
    similarity = calculate_claim_similarity(claim1, claim2)
    assert similarity > 0.8
    
    # Test similar claims
    claim3 = ClaimItem(
        id="claim3", text="Global warming is happening", type="factual", support="supported"
    )
    similarity = calculate_claim_similarity(claim1, claim3)
    print(f"Similarity between '{claim1.text}' and '{claim3.text}': {similarity}")
    assert similarity > 0.1  # Should have some similarity
    
    # Test dissimilar claims
    claim4 = ClaimItem(
        id="claim4", text="The economy is growing", type="factual", support="supported"
    )
    similarity = calculate_claim_similarity(claim1, claim4)
    assert similarity < 0.3  # Should have low similarity
    
    # Test normalization improvements
    claim5 = ClaimItem(
        id="claim5", text="Climate change isn't real", type="factual", support="unsupported"
    )
    claim6 = ClaimItem(
        id="claim6", text="Climate change is not real", type="factual", support="unsupported"
    )
    similarity = calculate_claim_similarity(claim5, claim6)
    assert similarity > 0.8  # Should handle contractions
    
    print("✓ Claim similarity calculation tests passed")


def test_determine_evidence_relationship():
    """Test evidence relationship determination."""
    # Test SUPPORTS relationship
    claim1 = ClaimItem(
        id="claim1", text="Climate change is real", type="factual", support="supported"
    )
    claim2 = ClaimItem(
        id="claim2", text="Global warming is happening", type="factual", support="supported"
    )
    
    corroboration = {
        "claim1": {"confidence": 0.8, "supported": True},
        "claim2": {"confidence": 0.8, "supported": True}
    }
    
    relationship = determine_evidence_relationship(claim1, claim2, 0.7, corroboration)
    assert relationship == "SUPPORTS"
    
    # Test CONTRADICTS relationship
    claim3 = ClaimItem(
        id="claim3", text="Climate change is not real", type="factual", support="unsupported"
    )
    
    corroboration_contradicts = {
        "claim1": {"confidence": 0.8, "supported": True},
        "claim3": {"confidence": 0.8, "supported": False}
    }
    
    relationship = determine_evidence_relationship(claim1, claim3, 0.6, corroboration_contradicts)
    assert relationship == "CONTRADICTS"
    
    # Test RELATES_TO relationship
    claim4 = ClaimItem(
        id="claim4", text="Environmental issues are important", type="opinion_presented_as_fact", support="unverifiable"
    )
    claim5 = ClaimItem(
        id="claim5", text="Pollution affects the environment", type="factual", support="unverifiable"
    )
    
    corroboration_relates = {
        "claim4": {"confidence": 0.4, "supported": None},
        "claim5": {"confidence": 0.4, "supported": None}
    }
    
    relationship = determine_evidence_relationship(claim4, claim5, 0.5, corroboration_relates)
    assert relationship == "RELATES_TO"
    
    # Test no relationship (low similarity)
    relationship = determine_evidence_relationship(claim1, claim4, 0.1, {})
    assert relationship is None
    
    print("✓ Evidence relationship determination tests passed")


def test_create_evidence_edges():
    """Test evidence edge creation."""
    claims = [
        ClaimItem(
            id="claim1", text="Climate change is real", type="factual", support="supported",
            quality_flags=["high_quality"], citations=[{"source": "research"}]
        ),
        ClaimItem(
            id="claim2", text="Global warming is happening", type="factual", support="supported",
            quality_flags=[], citations=[]
        ),
        ClaimItem(
            id="claim3", text="Climate change is not real", type="factual", support="unsupported",
            quality_flags=["low_quality"], citations=[]
        )
    ]
    
    corroboration = {
        "claim1": {"confidence": 0.9, "supported": True},
        "claim2": {"confidence": 0.7, "supported": True},
        "claim3": {"confidence": 0.8, "supported": False}
    }
    
    edges = create_evidence_edges(claims, corroboration)
    
    # Should create edges between related claims
    assert len(edges) > 0
    
    # Check edge properties
    for edge in edges:
        assert edge.source_id.startswith("claim:")
        assert edge.target_id.startswith("claim:")
        assert edge.relationship_type in ["SUPPORTS", "CONTRADICTS", "RELATES_TO"]
        assert 0.0 <= edge.weight <= 1.0
        assert "similarity_score" in edge.provenance
        assert "evidence_strength" in edge.provenance
        assert "corroboration_confidence" in edge.provenance
    
    # Check for expected relationships
    edge_types = [edge.relationship_type for edge in edges]
    assert "SUPPORTS" in edge_types or "CONTRADICTS" in edge_types
    
    print("✓ Evidence edge creation tests passed")


def test_build_evidence_graph():
    """Test complete evidence graph construction."""
    claims = [
        ClaimItem(
            id="claim1", text="Climate change is real", type="factual", support="supported"
        ),
        ClaimItem(
            id="claim2", text="Global warming is happening", type="factual", support="supported"
        ),
        ClaimItem(
            id="claim3", text="The economy is growing", type="factual", support="supported"
        )
    ]
    
    claims_output = ClaimsOutput(
        claims={"factual": 3},
        claim_items=claims
    )
    
    corroboration = {
        "claim1": {"confidence": 0.8, "supported": True},
        "claim2": {"confidence": 0.7, "supported": True},
        "claim3": {"confidence": 0.6, "supported": True}
    }
    
    evidence_graph = build_evidence_graph(claims_output, corroboration)
    
    # Check graph structure
    assert len(evidence_graph.claim_nodes) == 3
    assert all(node_id.startswith("claim:") for node_id in evidence_graph.claim_nodes.values())
    
    # Check metadata
    assert "num_claims" in evidence_graph.metadata
    assert "num_edges" in evidence_graph.metadata
    assert "construction_time" in evidence_graph.metadata
    assert "relationship_distribution" in evidence_graph.metadata
    assert "builder_version" in evidence_graph.metadata
    
    # Check similarity threshold (updated for enhanced algorithm)
    assert evidence_graph.similarity_threshold == 0.35
    
    # Check edge validation
    for edge in evidence_graph.edges.values():
        assert edge.relationship_type in ["SUPPORTS", "CONTRADICTS", "RELATES_TO"]
    
    print("✓ Evidence graph construction tests passed")


def test_edge_weight_calculation():
    """Test edge weight calculation formula."""
    claim1 = ClaimItem(
        id="claim1", text="Climate change is real", type="factual", support="supported",
        quality_flags=["high_quality"], citations=[{"source": "research"}]
    )
    claim2 = ClaimItem(
        id="claim2", text="Global warming is happening", type="factual", support="supported",
        quality_flags=[], citations=[]
    )
    
    corroboration = {
        "claim1": {"confidence": 0.9, "supported": True},
        "claim2": {"confidence": 0.7, "supported": True}
    }
    
    edges = create_evidence_edges([claim1, claim2], corroboration)
    
    if edges:
        edge = edges[0]
        
        # Check that weight follows the design formula
        similarity_score = edge.provenance["similarity_score"]
        evidence_strength = edge.provenance["evidence_strength"]
        corroboration_confidence = edge.provenance["corroboration_confidence"]
        
        expected_weight = (
            similarity_score * 0.4 + 
            evidence_strength * 0.4 + 
            corroboration_confidence * 0.2
        )
        
        assert abs(edge.weight - expected_weight) < 0.01  # Allow small floating point differences
    
    print("✓ Edge weight calculation tests passed")


def test_bidirectional_relationships():
    """Test that relationships work bidirectionally."""
    claim1 = ClaimItem(
        id="claim1", text="Climate change is real", type="factual", support="supported"
    )
    claim2 = ClaimItem(
        id="claim2", text="Global warming is happening", type="factual", support="supported"
    )
    
    corroboration = {
        "claim1": {"confidence": 0.8, "supported": True},
        "claim2": {"confidence": 0.8, "supported": True}
    }
    
    # Test relationship in both directions
    rel1 = determine_evidence_relationship(claim1, claim2, 0.7, corroboration)
    rel2 = determine_evidence_relationship(claim2, claim1, 0.7, corroboration)
    
    # Should be the same relationship type
    assert rel1 == rel2
    
    print("✓ Bidirectional relationship tests passed")


def test_enhanced_relationship_algorithms():
    """Test enhanced relationship determination algorithms."""
    # Test enhanced SUPPORTS with quality flags
    claim1 = ClaimItem(
        id="claim1", text="Climate change is real", type="factual", support="supported",
        quality_flags=["high_quality", "verified_source"], citations=[{"source": "research"}]
    )
    claim2 = ClaimItem(
        id="claim2", text="Global warming is happening", type="factual", support="supported",
        quality_flags=["peer_reviewed"], citations=[{"source": "study1"}, {"source": "study2"}]
    )
    
    corroboration = {
        "claim1": {"confidence": 0.9, "supported": True},
        "claim2": {"confidence": 0.8, "supported": True}
    }
    
    relationship = determine_evidence_relationship(claim1, claim2, 0.6, corroboration)
    assert relationship == "SUPPORTS"
    
    # Test enhanced CONTRADICTS with semantic indicators
    claim3 = ClaimItem(
        id="claim3", text="Temperature will increase significantly", type="factual", support="supported",
        quality_flags=["high_quality"], citations=[{"source": "study"}]
    )
    claim4 = ClaimItem(
        id="claim4", text="Temperature will decrease significantly", type="factual", support="unsupported",
        quality_flags=["high_quality"], citations=[{"source": "study"}]
    )
    
    corroboration_contradicts = {
        "claim3": {"confidence": 0.8, "supported": True},
        "claim4": {"confidence": 0.8, "supported": False}
    }
    
    # Use higher similarity since they share many words
    relationship = determine_evidence_relationship(claim3, claim4, 0.7, corroboration_contradicts)
    print(f"Relationship between '{claim3.text}' and '{claim4.text}': {relationship}")
    assert relationship == "CONTRADICTS"
    
    # Test enhanced RELATES_TO with domain keywords
    claim5 = ClaimItem(
        id="claim5", text="Climate policy affects economic growth", type="factual", support="contested"
    )
    claim6 = ClaimItem(
        id="claim6", text="Environmental regulations impact business", type="factual", support="unverifiable"
    )
    
    relationship = determine_evidence_relationship(claim5, claim6, 0.4, {})
    assert relationship == "RELATES_TO"
    
    print("✓ Enhanced relationship algorithms tests passed")


def test_performance_optimizations():
    """Test performance optimizations in edge creation."""
    # Create a larger set of claims to test optimization
    claims = []
    for i in range(10):
        claims.append(ClaimItem(
            id=f"claim{i}", 
            text=f"Test claim {i} about climate change and environment",
            type="factual", 
            support="supported" if i % 2 == 0 else "unsupported"
        ))
    
    corroboration = {f"claim{i}": {"confidence": 0.6, "supported": i % 2 == 0} for i in range(10)}
    
    import time
    start_time = time.time()
    edges = create_evidence_edges(claims, corroboration)
    end_time = time.time()
    
    # Should complete quickly even with optimizations
    assert end_time - start_time < 5.0  # Should be much faster, but allowing generous margin
    
    # Should still create meaningful edges
    assert len(edges) > 0
    
    # Check that provenance includes optimization info
    if edges:
        assert "quick_similarity_prefilter" in edges[0].provenance
        assert "relationship_determination_method" in edges[0].provenance
        assert edges[0].provenance["relationship_determination_method"] == "enhanced_evidence_analysis_v2"
    
    print("✓ Performance optimization tests passed")


def test_enhanced_edge_weights():
    """Test enhanced edge weight calculation."""
    claim1 = ClaimItem(
        id="claim1", text="Climate change is real", type="factual", support="supported",
        quality_flags=["high_quality", "peer_reviewed"], 
        citations=[{"source": "research", "type": "academic"}, {"source": "study", "type": "government"}]
    )
    claim2 = ClaimItem(
        id="claim2", text="Global warming is happening", type="factual", support="supported",
        quality_flags=["verified_source"], citations=[{"source": "report"}]
    )
    
    corroboration = {
        "claim1": {"confidence": 0.95, "supported": True},
        "claim2": {"confidence": 0.85, "supported": True}
    }
    
    edges = create_evidence_edges([claim1, claim2], corroboration)
    
    if edges:
        edge = edges[0]
        
        # Enhanced weights should be higher for high-quality claims
        assert edge.weight > 0.6  # Should be relatively high
        
        # Check enhanced provenance information
        assert "claim1_quality" in edge.provenance
        assert "claim2_quality" in edge.provenance
        assert "claim1_citations" in edge.provenance
        assert "claim2_citations" in edge.provenance
        
        # Quality flags should be recorded
        assert edge.provenance["claim1_quality"] == ["high_quality", "peer_reviewed"]
        assert edge.provenance["claim2_quality"] == ["verified_source"]
        
        # Citation counts should be recorded
        assert edge.provenance["claim1_citations"] == 2
        assert edge.provenance["claim2_citations"] == 1
    
    print("✓ Enhanced edge weight tests passed")


def test_enhanced_metadata():
    """Test enhanced metadata in evidence graph."""
    claims = [
        ClaimItem(
            id="claim1", text="Climate change is real", type="factual", support="supported",
            quality_flags=["high_quality"], citations=[{"source": "research"}]
        ),
        ClaimItem(
            id="claim2", text="Global warming is happening", type="factual", support="supported"
        ),
        ClaimItem(
            id="claim3", text="The economy is growing", type="factual", support="supported"
        )
    ]
    
    claims_output = ClaimsOutput(
        claims={"factual": 3},
        claim_items=claims
    )
    
    corroboration = {
        "claim1": {"confidence": 0.8, "supported": True},
        "claim2": {"confidence": 0.7, "supported": True},
        "claim3": {"confidence": 0.6, "supported": True}
    }
    
    evidence_graph = build_evidence_graph(claims_output, corroboration)
    
    # Check enhanced metadata
    assert "quality_metrics" in evidence_graph.metadata
    assert "performance_metrics" in evidence_graph.metadata
    assert "algorithm_enhancements" in evidence_graph.metadata
    assert evidence_graph.metadata["builder_version"] == "2.0_enhanced"
    
    # Check quality metrics structure
    quality_metrics = evidence_graph.metadata["quality_metrics"]
    assert "high_confidence_edges" in quality_metrics
    assert "avg_similarity_score" in quality_metrics
    assert "quality_distribution" in quality_metrics
    
    # Check performance metrics structure
    performance_metrics = evidence_graph.metadata["performance_metrics"]
    assert "edges_per_second" in performance_metrics
    assert "edge_creation_rate" in performance_metrics
    assert "memory_efficiency_score" in performance_metrics
    
    print("✓ Enhanced metadata tests passed")
def run_all_tests():
    """Run all evidence graph tests."""
    print("Running evidence graph construction tests...")
    
    test_calculate_claim_similarity()
    test_determine_evidence_relationship()
    test_create_evidence_edges()
    test_build_evidence_graph()
    test_edge_weight_calculation()
    test_bidirectional_relationships()
    test_enhanced_relationship_algorithms()
    test_performance_optimizations()
    test_enhanced_edge_weights()
    test_enhanced_metadata()
    
    print("\n✅ All evidence graph tests passed!")


if __name__ == "__main__":
    run_all_tests()