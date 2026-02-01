"""
Unit tests for enhanced claim graph construction algorithms.
"""

from datetime import datetime
from hawkins_truth_engine.schemas import (
    Document, Entity, Attribution, CharSpan, ClaimsOutput, ClaimItem, 
    SourceIntelOutput, EvidenceItem, Pointer, LanguageInfo
)
from hawkins_truth_engine.graph.claim_graph import (
    _calculate_text_overlap, _detect_claim_entity_mentions, 
    _detect_claim_source_attribution, build_claim_graph
)


def test_enhanced_text_overlap():
    """Test enhanced text overlap calculation."""
    # Test basic overlap
    assert _calculate_text_overlap("climate change", "climate") > 0.0
    
    # Test normalization improvements
    assert _calculate_text_overlap("don't", "do not") > 0.0
    assert _calculate_text_overlap("it's", "it is") > 0.0
    
    # Test stop word filtering
    overlap1 = _calculate_text_overlap("the climate change", "climate change")
    overlap2 = _calculate_text_overlap("climate change", "climate change")
    assert overlap1 == overlap2  # Stop words should be filtered
    
    print("✓ Enhanced text overlap tests passed")


def test_enhanced_entity_mention_detection():
    """Test enhanced entity-claim mention detection."""
    # Test direct containment
    assert _detect_claim_entity_mentions("Dr. Smith said something", "Dr. Smith") == 1.0
    
    # Test person name matching (last name)
    confidence = _detect_claim_entity_mentions("Smith argued that...", "John Smith", "PERSON")
    assert confidence > 0.25  # Enhanced detection for person names
    
    # Test basic word overlap with position boosting
    confidence1 = _detect_claim_entity_mentions("Climate change is serious", "climate", "MISC")
    confidence2 = _detect_claim_entity_mentions("The issue of climate is serious", "climate", "MISC")
    assert confidence1 > 0.0 and confidence2 > 0.0  # Both should detect mentions
    
    # Test enhanced normalization
    confidence = _detect_claim_entity_mentions("The company's profits", "company", "ORG")
    assert confidence > 0.0  # Should handle contractions
    
    print("✓ Enhanced entity mention detection tests passed")


def test_enhanced_attribution_detection():
    """Test enhanced claim-source attribution detection."""
    # Create test document
    document = Document(
        input_type="raw_text",
        raw_input="Test text",
        display_text="Dr. Smith said climate change is real",
        language=LanguageInfo(top="en"),
        sentences=[], tokens=[],
        entities=[
            Entity(id=1, text="Dr. Smith", type="PERSON", sentence_id=0, 
                   char_span=CharSpan(start=0, end=9))
        ],
        attributions=[
            Attribution(speaker_entity_id=1, verb="said", 
                       quote_span=CharSpan(start=15, end=37), sentence_id=0)
        ]
    )
    
    # Create test claim
    claim_item = ClaimItem(
        id="test_claim",
        text="climate change is real",
        type="factual",
        support="supported",
        pointers=Pointer(char_spans=[CharSpan(start=15, end=37)]),
        citations=[{"source": "research"}]
    )
    
    # Test attribution detection
    attributions = _detect_claim_source_attribution(claim_item, document)
    
    # Should find entity attribution
    entity_attributions = [attr for attr in attributions if "entity:" in attr[0]]
    assert len(entity_attributions) > 0
    assert any(conf > 0.5 for _, conf in entity_attributions)
    
    print("✓ Enhanced attribution detection tests passed")


def test_performance_optimization():
    """Test performance optimization features."""
    # Create test data with multiple entities and claims
    entities = [
        Entity(id=i, text=f"Entity {i}", type="MISC", sentence_id=0,
               char_span=CharSpan(start=i*10, end=i*10+8))
        for i in range(10)
    ]
    
    document = Document(
        input_type="raw_text",
        raw_input="Test text",
        display_text="Test document with multiple entities",
        language=LanguageInfo(top="en"),
        sentences=[], tokens=[], entities=entities, attributions=[]
    )
    
    claims = [
        ClaimItem(id=f"claim_{i}", text=f"Claim {i} about Entity {i}", 
                 type="factual", support="supported")
        for i in range(5)
    ]
    
    claims_output = ClaimsOutput(claims={}, claim_items=claims)
    source_output = SourceIntelOutput(source_trust_score=0.8, source_flags=[])
    
    # Build graph and check performance metadata
    graph = build_claim_graph(document, claims_output, source_output)
    
    assert "performance" in graph.metadata
    assert "total_construction_time" in graph.metadata["performance"]
    assert "complexity_score" in graph.metadata
    
    print("✓ Performance optimization tests passed")


def run_all_tests():
    """Run all enhancement tests."""
    print("Running enhanced claim graph construction tests...")
    
    test_enhanced_text_overlap()
    test_enhanced_entity_mention_detection()
    test_enhanced_attribution_detection()
    test_performance_optimization()
    
    print("\n✅ All enhanced claim graph tests passed!")


if __name__ == "__main__":
    run_all_tests()