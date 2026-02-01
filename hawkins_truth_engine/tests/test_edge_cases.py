"""
Edge case tests for critical issue fixes in the Hawkins Truth Engine.

Tests cover:
1. Input validation (empty content, oversized content, invalid URLs)
2. URL fetch error handling (timeouts, network errors)
3. Empty claims handling in reasoning
4. Graph construction edge cases
5. API fallback behavior
6. Type consistency (confidence score clamping)
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from hawkins_truth_engine.validators import (
    validate_analyze_request,
    validate_content_length,
    validate_url_format,
    ValidationError,
)
from hawkins_truth_engine.reasoning import aggregate
from hawkins_truth_engine.schemas import (
    ClaimsOutput,
    ClaimItem,
    LinguisticOutput,
    StatisticalOutput,
    SourceIntelOutput,
    EvidenceItem,
    AggregationOutput,
)
from hawkins_truth_engine.graph.evidence_graph import build_evidence_graph
from hawkins_truth_engine.graph.claim_graph import build_claim_graph


# =============================================================================
# Issue 1: Input Validation Tests
# =============================================================================

class TestInputValidation:
    """Tests for input validation (Issue 1)."""
    
    def test_empty_content_rejected(self):
        """Empty content should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_content_length("", "raw_text")
        assert "too short" in str(exc_info.value).lower()
    
    def test_whitespace_only_content_rejected(self):
        """Whitespace-only content should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_content_length("   \n\t  ", "raw_text")
        assert "too short" in str(exc_info.value).lower()
    
    def test_oversized_content_rejected(self):
        """Content exceeding maximum size should raise ValidationError."""
        # Create content larger than 10MB limit
        large_content = "x" * (10_000_001)
        with pytest.raises(ValidationError) as exc_info:
            validate_content_length(large_content, "raw_text")
        assert "exceeds maximum length" in str(exc_info.value).lower()
    
    def test_valid_content_accepted(self):
        """Valid content should not raise any errors."""
        # Should not raise
        validate_content_length("This is valid content for testing.", "raw_text")
    
    def test_invalid_url_format_rejected(self):
        """Invalid URL format should raise ValidationError."""
        with pytest.raises(ValidationError):
            validate_url_format("not-a-valid-url")
    
    def test_url_without_scheme_rejected(self):
        """URL without scheme should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_url_format("www.example.com/page")
        assert "scheme" in str(exc_info.value).lower()
    
    def test_url_with_invalid_scheme_rejected(self):
        """URL with invalid scheme should raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_url_format("ftp://example.com/file")
        assert "not supported" in str(exc_info.value).lower()
    
    def test_valid_url_accepted(self):
        """Valid URLs should not raise any errors."""
        # Should not raise
        validate_url_format("https://www.example.com/page")
        validate_url_format("http://example.com")


# =============================================================================
# Issue 2: URL Fetch Error Handling Tests
# =============================================================================

class TestUrlFetchErrorHandling:
    """Tests for URL fetch error handling (Issue 2)."""
    
    @pytest.mark.asyncio
    async def test_fetch_url_timeout_returns_error(self):
        """fetch_url should return error dict on timeout, not crash."""
        from hawkins_truth_engine.ingest import fetch_url
        import httpx
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.side_effect = httpx.TimeoutException("Timeout")
            mock_client.return_value.__aenter__.return_value = mock_instance
            
            result = await fetch_url("https://example.com")
            
            assert result["error"] == "timeout"
            assert result["status_code"] == 0
    
    @pytest.mark.asyncio
    async def test_fetch_url_connection_error_returns_error(self):
        """fetch_url should return error dict on connection failure."""
        from hawkins_truth_engine.ingest import fetch_url
        import httpx
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.side_effect = httpx.ConnectError("DNS failure")
            mock_client.return_value.__aenter__.return_value = mock_instance
            
            result = await fetch_url("https://nonexistent-domain-12345.invalid")
            
            assert "connection_failed" in result["error"]
    
    @pytest.mark.asyncio
    async def test_fetch_url_invalid_url_returns_error(self):
        """fetch_url should return error dict on invalid URL."""
        from hawkins_truth_engine.ingest import fetch_url
        import httpx
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_instance.get.side_effect = httpx.InvalidURL("Invalid URL")
            mock_client.return_value.__aenter__.return_value = mock_instance
            
            result = await fetch_url("not-a-url")
            
            assert "invalid_url" in result["error"]


# =============================================================================
# Issue 3: Empty Claims Handling Tests
# =============================================================================

class TestEmptyClaimsHandling:
    """Tests for empty claims handling in reasoning (Issue 3)."""
    
    def _create_mock_linguistic(self):
        return LinguisticOutput(
            linguistic_risk_score=0.3,
            signals=[],
            highlighted_phrases=[]
        )
    
    def _create_mock_statistical(self):
        return StatisticalOutput(
            statistical_risk_score=0.3,
            evidence=[]
        )
    
    def _create_mock_source(self):
        return SourceIntelOutput(
            source_trust_score=0.7,
            source_flags=[]
        )
    
    def test_empty_claims_returns_valid_output(self):
        """aggregate() should return valid output when no claims are extracted."""
        linguistic = self._create_mock_linguistic()
        statistical = self._create_mock_statistical()
        source = self._create_mock_source()
        claims = ClaimsOutput(claims={}, claim_items=[])
        
        result = aggregate(linguistic, statistical, source, claims)
        
        # Should return valid AggregationOutput
        assert isinstance(result, AggregationOutput)
        assert result.credibility_score == 50  # Neutral
        assert result.confidence == 0.1  # Very low
        assert "no_claims_extracted" in result.uncertainty_flags
    
    def test_empty_claims_does_not_crash(self):
        """aggregate() should not crash with empty claims."""
        linguistic = self._create_mock_linguistic()
        statistical = self._create_mock_statistical()
        source = self._create_mock_source()
        claims = ClaimsOutput(claims={}, claim_items=[])
        
        # Should not raise any exception
        try:
            result = aggregate(linguistic, statistical, source, claims)
            assert result is not None
        except Exception as e:
            pytest.fail(f"aggregate() crashed with empty claims: {e}")


# =============================================================================
# Issue 4: Graph Construction Edge Cases Tests
# =============================================================================

class TestGraphConstructionEdgeCases:
    """Tests for graph construction edge cases (Issue 4)."""
    
    def test_evidence_graph_with_none_input(self):
        """build_evidence_graph should handle None claims_output."""
        result = build_evidence_graph(None, {})
        
        assert result is not None
        assert len(result.claim_nodes) == 0
        assert "error" in result.metadata
    
    def test_evidence_graph_with_none_corroboration(self):
        """build_evidence_graph should handle None corroboration."""
        claims = ClaimsOutput(claims={}, claim_items=[])
        
        # Should not crash
        result = build_evidence_graph(claims, None)
        
        assert result is not None
    
    def test_evidence_graph_with_single_claim(self):
        """build_evidence_graph should handle exactly 1 claim."""
        claim = ClaimItem(
            id="C1", text="Test claim", type="factual", support="supported"
        )
        claims = ClaimsOutput(claims={"supported": 1}, claim_items=[claim])
        
        result = build_evidence_graph(claims, {})
        
        # Should return valid graph with 1 claim
        assert len(result.claim_nodes) == 1
        # No edges possible with single claim
        assert len(result.edges) == 0
    
    def test_evidence_graph_with_zero_claims(self):
        """build_evidence_graph should handle 0 claims."""
        claims = ClaimsOutput(claims={}, claim_items=[])
        
        result = build_evidence_graph(claims, {})
        
        assert result is not None
        assert len(result.claim_nodes) == 0


# =============================================================================
# Issue 5: API Fallback Tests  
# =============================================================================

class TestApiFallback:
    """Tests for API fallback behavior (Issue 5)."""
    
    @pytest.mark.asyncio
    async def test_all_apis_failed_marks_unverifiable(self):
        """When all APIs fail, claims should be marked unverifiable."""
        from hawkins_truth_engine.analyzers.claims import analyze_claims
        from hawkins_truth_engine.schemas import Document, LanguageInfo, Sentence, CharSpan
        
        # Create minimal document
        doc = MagicMock()
        doc.display_text = "This is a test claim that vaccines cause autism."
        doc.sentences = [
            Sentence(id=0, text="This is a test claim that vaccines cause autism.", 
                    char_span=CharSpan(start=0, end=48))
        ]
        doc.attributions = []
        
        # Mock all external APIs to fail
        with patch('hawkins_truth_engine.analyzers.claims._pubmed_evidence_for_claim') as mock_pubmed, \
             patch('hawkins_truth_engine.analyzers.claims._gdelt_evidence_for_claim') as mock_gdelt, \
             patch('hawkins_truth_engine.analyzers.claims._tavily_evidence_for_claim') as mock_tavily:
            
            mock_pubmed.return_value = {
                "citations": [], "query_trace": [], 
                "quality_flags": [], "uncertainty_flags": ["ncbi_unavailable"]
            }
            mock_gdelt.return_value = {
                "neighbors": [], "query_trace": [], 
                "uncertainty_flags": ["gdelt_unavailable"]
            }
            mock_tavily.return_value = {
                "neighbors": [], "query_trace": [], 
                "uncertainty_flags": ["tavily_unavailable"]
            }
            
            result = await analyze_claims(doc)
            
            # All claims should be unverifiable
            for claim in result.claim_items:
                assert claim.support == "unverifiable" or "all_external_apis_unavailable" in claim.uncertainty_flags


# =============================================================================
# Issue 6: Type Consistency Tests
# =============================================================================

class TestTypeConsistency:
    """Tests for type consistency in reasoning (Issue 6)."""
    
    def _create_mock_inputs(self):
        linguistic = LinguisticOutput(
            linguistic_risk_score=0.5,
            signals=[],
            highlighted_phrases=[]
        )
        statistical = StatisticalOutput(
            statistical_risk_score=0.5,
            evidence=[]
        )
        source = SourceIntelOutput(
            source_trust_score=0.5,
            source_flags=[]
        )
        claim = ClaimItem(
            id="C1", text="Test claim", type="factual", support="supported"
        )
        claims = ClaimsOutput(
            claims={"supported": 1}, 
            claim_items=[claim]
        )
        return linguistic, statistical, source, claims
    
    def test_confidence_in_valid_range(self):
        """Confidence score should always be between 0.0 and 1.0."""
        linguistic, statistical, source, claims = self._create_mock_inputs()
        
        result = aggregate(linguistic, statistical, source, claims)
        
        assert 0.0 <= result.confidence <= 1.0
    
    def test_credibility_score_in_valid_range(self):
        """Credibility score should always be between 0 and 100."""
        linguistic, statistical, source, claims = self._create_mock_inputs()
        
        result = aggregate(linguistic, statistical, source, claims)
        
        assert 0 <= result.credibility_score <= 100
    
    def test_extreme_risk_values_produce_valid_output(self):
        """Extreme input values should still produce valid output."""
        # Test with extreme high risk
        linguistic = LinguisticOutput(
            linguistic_risk_score=1.0,
            signals=[],
            highlighted_phrases=[]
        )
        statistical = StatisticalOutput(
            statistical_risk_score=1.0,
            evidence=[]
        )
        source = SourceIntelOutput(
            source_trust_score=0.0,
            source_flags=[]
        )
        claim = ClaimItem(
            id="C1", text="Test claim", type="factual", support="unsupported"
        )
        claims = ClaimsOutput(
            claims={"unsupported": 1}, 
            claim_items=[claim]
        )
        
        result = aggregate(linguistic, statistical, source, claims)
        
        # Should still be in valid ranges
        assert 0.0 <= result.confidence <= 1.0
        assert 0 <= result.credibility_score <= 100


# =============================================================================
# Run all tests
# =============================================================================

def run_all_tests():
    """Run all edge case tests."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_all_tests()
