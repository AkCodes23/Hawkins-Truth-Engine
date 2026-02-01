"""Test edge cases in analysis pipeline."""
import pytest
from hawkins_truth_engine.schemas import (
    Document, LanguageInfo, Sentence, Token, CharSpan, Entity, Attribution, ClaimsOutput, ClaimItem, Pointer,
    LinguisticOutput, StatisticalOutput, SourceIntelOutput, EvidenceItem, AggregationOutput
)
from hawkins_truth_engine.reasoning import aggregate


@pytest.fixture
def minimal_linguistic():
    """Minimal linguistic output."""
    return LinguisticOutput(
        linguistic_risk_score=0.5,
        signals=[],
        highlighted_phrases=[]
    )


@pytest.fixture
def minimal_statistical():
    """Minimal statistical output."""
    return StatisticalOutput(
        statistical_risk_score=0.5,
        evidence=[]
    )


@pytest.fixture
def minimal_source():
    """Minimal source output."""
    return SourceIntelOutput(
        source_trust_score=0.5,
        source_flags=[]
    )


class TestEmptyCases:
    """Test edge cases with empty or minimal data."""
    
    def test_no_claims_extracted(self, minimal_linguistic, minimal_statistical, minimal_source):
        """Test aggregation when no claims are extracted."""
        claims = ClaimsOutput(
            claims={},
            claim_items=[],
            medical_topic_detected=False
        )
        
        result = aggregate(minimal_linguistic, minimal_statistical, minimal_source, claims)
        
        assert result.credibility_score == 50
        assert result.verdict == "Suspicious"
        assert result.confidence <= 0.15
        assert "no_claims_extracted" in result.uncertainty_flags
    
    def test_single_claim_unverifiable(self, minimal_linguistic, minimal_statistical, minimal_source):
        """Test with single unverifiable claim."""
        claims = ClaimsOutput(
            claims={"supported": 0, "unsupported": 0, "unverifiable": 1},
            claim_items=[
                ClaimItem(
                    id="C1",
                    text="Test claim",
                    type="factual",
                    support="unverifiable",
                    reasons=[],
                    pointers=Pointer(),
                    citations=[],
                    query_trace=[],
                    quality_flags=[],
                    uncertainty_flags=[]
                )
            ]
        )
        
        result = aggregate(minimal_linguistic, minimal_statistical, minimal_source, claims)
        
        assert result.credibility_score >= 0
        assert result.credibility_score <= 100
        assert result.confidence >= 0.05
        assert result.confidence <= 0.95


class TestAmbiguousCases:
    """Test ambiguous scenarios with mixed signals."""
    
    def test_mixed_claim_support(self, minimal_linguistic, minimal_statistical, minimal_source):
        """Test ambiguous case with mixed support statuses."""
        claims = ClaimsOutput(
            claims={"supported": 1, "unsupported": 1, "unverifiable": 1},
            claim_items=[
                ClaimItem(id="C1", text="Claim 1", type="factual", support="supported", pointers=Pointer()),
                ClaimItem(id="C2", text="Claim 2", type="factual", support="unsupported", pointers=Pointer()),
                ClaimItem(id="C3", text="Claim 3", type="factual", support="unverifiable", pointers=Pointer()),
            ]
        )
        
        result = aggregate(minimal_linguistic, minimal_statistical, minimal_source, claims)
        
        assert "mixed_claim_support" in result.uncertainty_flags
        assert result.confidence <= 0.65
    
    def test_high_multi_risk_scenario(self, minimal_source):
        """Test high risk across multiple dimensions."""
        from hawkins_truth_engine import config
        
        linguistic = LinguisticOutput(
            linguistic_risk_score=0.8,
            signals=[],
            highlighted_phrases=[]
        )
        statistical = StatisticalOutput(
            statistical_risk_score=0.8,
            evidence=[]
        )
        claims = ClaimsOutput(
            claims={"supported": 0, "unsupported": 2, "unverifiable": 0},
            claim_items=[
                ClaimItem(id="C1", text="Claim 1", type="factual", support="unsupported", pointers=Pointer()),
                ClaimItem(id="C2", text="Claim 2", type="factual", support="unsupported", pointers=Pointer()),
            ]
        )
        
        result = aggregate(linguistic, statistical, minimal_source, claims)
        
        assert "high_multi_signal_risk" in result.uncertainty_flags
        assert result.credibility_score <= 50


class TestConfidenceCalculation:
    """Test confidence score calculations."""
    
    def test_confidence_range_validation(self, minimal_linguistic, minimal_statistical, minimal_source):
        """Test that confidence is always in valid range [0, 1]."""
        claims = ClaimsOutput(
            claims={"supported": 5, "unsupported": 0, "unverifiable": 0},
            claim_items=[
                ClaimItem(id=f"C{i}", text=f"Claim {i}", type="factual", support="supported", pointers=Pointer())
                for i in range(5)
            ]
        )
        
        result = aggregate(minimal_linguistic, minimal_statistical, minimal_source, claims)
        
        assert 0.0 <= result.confidence <= 1.0
    
    def test_credibility_score_range_validation(self, minimal_linguistic, minimal_statistical, minimal_source):
        """Test that credibility score is always in valid range [0, 100]."""
        claims = ClaimsOutput(
            claims={"supported": 1, "unsupported": 0, "unverifiable": 0},
            claim_items=[
                ClaimItem(id="C1", text="Test claim", type="factual", support="supported", pointers=Pointer())
            ]
        )
        
        result = aggregate(minimal_linguistic, minimal_statistical, minimal_source, claims)
        
        assert 0 <= result.credibility_score <= 100


class TestClaimAgreement:
    """Test claim agreement detection."""
    
    def test_majority_supported_claims(self, minimal_linguistic, minimal_statistical, minimal_source):
        """Test when majority of claims are supported."""
        claims = ClaimsOutput(
            claims={"supported": 3, "unsupported": 0, "unverifiable": 1},
            claim_items=[
                ClaimItem(id="C1", text="Claim 1", type="factual", support="supported", pointers=Pointer()),
                ClaimItem(id="C2", text="Claim 2", type="factual", support="supported", pointers=Pointer()),
                ClaimItem(id="C3", text="Claim 3", type="factual", support="supported", pointers=Pointer()),
                ClaimItem(id="C4", text="Claim 4", type="factual", support="unverifiable", pointers=Pointer()),
            ]
        )
        
        result = aggregate(minimal_linguistic, minimal_statistical, minimal_source, claims)
        
        # High agreement should increase confidence
        assert result.confidence >= 0.5


class TestReasoningRules:
    """Test reasoning rule triggering."""
    
    def test_low_trust_high_linguistic_low_support_rule(self, minimal_source):
        """Test R1: Low trust + high linguistic risk + no support triggers."""
        linguistic = LinguisticOutput(
            linguistic_risk_score=0.75,
            signals=[],
            highlighted_phrases=[]
        )
        statistical = StatisticalOutput(
            statistical_risk_score=0.3,
            evidence=[]
        )
        source = SourceIntelOutput(
            source_trust_score=0.2,
            source_flags=[]
        )
        claims = ClaimsOutput(
            claims={"supported": 0, "unsupported": 2, "unverifiable": 1},
            claim_items=[
                ClaimItem(id="C1", text="Claim 1", type="factual", support="unsupported", pointers=Pointer()),
                ClaimItem(id="C2", text="Claim 2", type="factual", support="unsupported", pointers=Pointer()),
                ClaimItem(id="C3", text="Claim 3", type="factual", support="unverifiable", pointers=Pointer()),
            ]
        )
        
        result = aggregate(linguistic, statistical, source, claims)
        
        # Rule should trigger and push toward fake
        assert result.verdict in ("Likely Fake", "Suspicious")
        assert result.credibility_score < 50
