"""
End-to-End Tests for Hawkins Truth Engine API.

Tests the complete API functionality including:
- Health and status endpoints
- Main analysis endpoint with various input types
- Input validation and error handling
- Graph export and metrics endpoints
- Calibration endpoints

Uses subprocess to spawn a test server.

Markers:
- @pytest.mark.slow: Tests that involve full analysis with external APIs (may take 2+ minutes)
"""

import pytest
import subprocess
import time
import socket
import requests
import os
import sys
from contextlib import contextmanager


# =============================================================================
# Pre-flight Import Check
# =============================================================================

def test_app_can_be_imported():
    """Verify that the application module can be imported without errors.

    This test runs first to catch any module-level import errors before
    attempting to spawn subprocess servers, which would hide the actual
    import error message.
    """
    try:
        from hawkins_truth_engine import app
        assert hasattr(app, 'app'), "FastAPI app instance not found"
    except Exception as e:
        import traceback
        pytest.fail(f"Failed to import hawkins_truth_engine.app: {type(e).__name__}: {e}\n{traceback.format_exc()}")


# =============================================================================
# Test Server Management
# =============================================================================

def find_free_port():
    """Find an available port for the test server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


@contextmanager
def server_context(port: int, timeout: int = 30):
    """Context manager that starts and stops the test server.

    Args:
        port: Port to run the server on
        timeout: Max seconds to wait for server startup (default: 30s for slower machines)
    """
    # Start the server process
    env = os.environ.copy()
    # Set shorter timeouts for external APIs during testing
    env["HTE_HTTP_TIMEOUT_SECS"] = "15"

    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Ensure Python path includes project root
    python_path = env.get("PYTHONPATH", "")
    if python_path:
        env["PYTHONPATH"] = f"{project_root}{os.pathsep}{python_path}"
    else:
        env["PYTHONPATH"] = project_root

    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "hawkins_truth_engine.app:app",
         "--host", "127.0.0.1",
         "--port", str(port),
         "--log-level", "info"],  # Use info level to capture more details
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=project_root
    )

    # Wait for server to be ready
    base_url = f"http://127.0.0.1:{port}"
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/health", timeout=1)
            if response.status_code == 200:
                server_ready = True
                break
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(0.3)

    if not server_ready:
        # Capture any output from the server for debugging
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=2)
            stdout_text = stdout.decode('utf-8', errors='replace') if stdout else ""
            stderr_text = stderr.decode('utf-8', errors='replace') if stderr else ""
            error_details = ""
            if stdout_text.strip():
                error_details += f"\nServer stdout:\n{stdout_text[:2000]}"
            if stderr_text.strip():
                error_details += f"\nServer stderr:\n{stderr_text[:2000]}"
            if not error_details:
                error_details = "\nNo output captured from server."
            pytest.fail(f"Server failed to start within {timeout}s on port {port}{error_details}")
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            pytest.fail(f"Server failed to start within {timeout}s and couldn't capture output")

    try:
        yield base_url
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def server():
    """Start a test server for the module's tests."""
    port = find_free_port()
    with server_context(port) as base_url:
        yield base_url


@pytest.fixture
def sample_raw_text():
    """Sample raw text content for analysis."""
    return """
    Scientists at Harvard University have discovered a new treatment for diabetes.
    The research, published in the Journal of Medicine, shows promising results
    in clinical trials with over 500 patients. Dr. Sarah Johnson, the lead
    researcher, stated that the treatment could be available within two years.
    """


@pytest.fixture
def sample_suspicious_text():
    """Sample text with suspicious patterns."""
    return """
    BREAKING NEWS!!! You won't BELIEVE what they're hiding from you!!!
    The government doesn't want you to know this SECRET cure that Big Pharma
    has been suppressing for years! Share this before it gets deleted!
    This miracle treatment cures EVERYTHING - cancer, diabetes, and more!
    WAKE UP SHEEPLE! They're putting chemicals in the water to control us!
    """


@pytest.fixture
def sample_short_text():
    """Sample minimal text for edge case testing."""
    return "This is a short test statement about climate change."


@pytest.fixture
def sample_medical_claim():
    """Sample text with medical claims for corroboration testing."""
    return """
    Recent studies have shown that vitamin D supplementation may reduce the risk
    of respiratory infections. Research published in the British Medical Journal
    found that participants who took vitamin D had 12% fewer infections compared
    to the control group. The World Health Organization recommends adequate
    vitamin D intake for immune health.
    """


# =============================================================================
# Health and Status Endpoint Tests (Fast)
# =============================================================================


class TestHealthEndpoints:
    """Tests for health check and status endpoints."""

    def test_health_check_returns_ok(self, server):
        """Test that health check endpoint returns healthy status."""
        response = requests.get(f"{server}/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "1.0"

    def test_status_endpoint_returns_api_info(self, server):
        """Test that status endpoint returns API information."""
        response = requests.get(f"{server}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "operational"
        assert "version" in data

    def test_test_apis_endpoint_lists_endpoints(self, server):
        """Test that test-apis endpoint lists available endpoints."""
        response = requests.get(f"{server}/test-apis")

        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data
        # Endpoints is a dict mapping names to descriptions
        assert isinstance(data["endpoints"], dict)
        assert "analyze" in data["endpoints"]
        assert "health" in data["endpoints"]


# =============================================================================
# Input Validation Tests (Fast - no external API calls)
# =============================================================================


class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_analyze_rejects_empty_content(self, server):
        """Test that empty content is rejected."""
        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "raw_text",
                "content": ""
            }
        )

        assert response.status_code == 422

    def test_analyze_rejects_too_short_content(self, server):
        """Test that content below minimum length is rejected."""
        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "raw_text",
                "content": "Hi"  # Too short
            }
        )

        assert response.status_code == 422

    def test_analyze_rejects_invalid_input_type(self, server):
        """Test that invalid input type is rejected."""
        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "invalid_type",
                "content": "Some valid content for testing"
            }
        )

        assert response.status_code == 422

    def test_analyze_rejects_missing_content(self, server):
        """Test that missing content field is rejected."""
        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "raw_text"
            }
        )

        assert response.status_code == 422

    def test_analyze_rejects_missing_input_type(self, server):
        """Test that missing input_type field is rejected."""
        response = requests.post(
            f"{server}/analyze",
            json={
                "content": "Some content to analyze"
            }
        )

        assert response.status_code == 422

    def test_analyze_url_type_validates_url_format(self, server):
        """Test that URL input type validates URL format."""
        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "url",
                "content": "not-a-valid-url"
            }
        )

        # Should reject invalid URL
        assert response.status_code == 422

    def test_analyze_url_type_rejects_missing_scheme(self, server):
        """Test that URL without http/https scheme is rejected."""
        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "url",
                "content": "www.example.com"
            }
        )

        assert response.status_code == 422

    def test_analyze_url_type_rejects_non_http_scheme(self, server):
        """Test that non-http URLs are rejected."""
        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "url",
                "content": "ftp://example.com/file.txt"
            }
        )

        assert response.status_code == 422


# =============================================================================
# URL Input Tests (SLOW - require network)
# =============================================================================


@pytest.mark.slow
class TestUrlInput:
    """Tests for URL input type analysis."""

    def test_analyze_valid_url_returns_response(self, server):
        """Test analysis of a valid URL returns proper response structure."""
        # Use a well-known, stable URL for testing
        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "url",
                "content": "https://www.example.com",
                "include_graphs": False
            },
            timeout=300
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "document" in data
        assert data["document"]["input_type"] == "url"
        assert data["document"]["url"] == "https://www.example.com"
        assert "domain" in data["document"]
        assert data["document"]["domain"] == "example.com"

        # Should have aggregation even if content is minimal
        assert "aggregation" in data
        assert "verdict" in data["aggregation"]

    def test_analyze_url_handles_unreachable_host(self, server):
        """Test that unreachable URLs are handled gracefully."""
        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "url",
                "content": "https://this-domain-definitely-does-not-exist-12345.com",
                "include_graphs": False
            },
            timeout=60
        )

        # Should return 200 with graceful degradation, not 500
        assert response.status_code == 200
        data = response.json()

        # Should have preprocessing flags indicating fetch error
        assert "document" in data
        assert "preprocessing_flags" in data["document"]
        assert any("fetch_error" in flag for flag in data["document"]["preprocessing_flags"])

    def test_analyze_url_extracts_domain(self, server):
        """Test that domain is correctly extracted from URL."""
        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "url",
                "content": "https://news.ycombinator.com/item?id=12345",
                "include_graphs": False
            },
            timeout=300
        )

        # May succeed or fail depending on network, but should not error
        assert response.status_code == 200
        data = response.json()

        # Domain extraction should work regardless of fetch success
        assert data["document"]["domain"] == "news.ycombinator.com"


# =============================================================================
# Calibration Endpoint Tests (Fast)
# =============================================================================


class TestCalibrationEndpoints:
    """Tests for calibration endpoints."""

    def test_calibration_status_returns_info(self, server):
        """Test that calibration status endpoint returns model info."""
        response = requests.get(f"{server}/calibration/status")

        assert response.status_code == 200
        data = response.json()
        assert "calibration_available" in data

    def test_calibrate_confidence_with_valid_input(self, server):
        """Test calibrating a confidence score."""
        response = requests.post(
            f"{server}/calibration/calibrate",
            json={
                "heuristic_confidence": 0.75
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "calibrated_confidence" in data
        assert "heuristic_confidence" in data
        assert data["heuristic_confidence"] == 0.75

    def test_calibrate_confidence_bounds_check(self, server):
        """Test that calibrated confidence stays within valid bounds."""
        # Test with extreme values
        for raw_conf in [0.0, 0.5, 1.0]:
            response = requests.post(
                f"{server}/calibration/calibrate",
                json={
                    "heuristic_confidence": raw_conf
                }
            )

            assert response.status_code == 200
            data = response.json()
            calibrated = data.get("calibrated_confidence", 0)
            assert 0 <= calibrated <= 1


# =============================================================================
# Static File Endpoint Tests (Fast)
# =============================================================================


class TestStaticEndpoints:
    """Tests for static file serving endpoints."""

    def test_root_serves_web_ui(self, server):
        """Test that root endpoint serves the web UI."""
        response = requests.get(f"{server}/")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_presentation_endpoint_serves_html(self, server):
        """Test that presentation endpoint serves HTML."""
        response = requests.get(f"{server}/presentation")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")


# =============================================================================
# Error Response Tests (Fast)
# =============================================================================


class TestErrorResponses:
    """Tests for error response formatting."""

    def test_invalid_json_returns_error(self, server):
        """Test that invalid JSON body returns appropriate error."""
        response = requests.post(
            f"{server}/analyze",
            data="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_nonexistent_endpoint_returns_404(self, server):
        """Test that nonexistent endpoint returns 404."""
        response = requests.get(f"{server}/nonexistent-endpoint")

        assert response.status_code == 404

    def test_wrong_method_returns_405(self, server):
        """Test that wrong HTTP method returns 405."""
        response = requests.get(f"{server}/analyze")  # Should be POST

        assert response.status_code == 405


# =============================================================================
# Main Analysis Endpoint Tests (SLOW - involve external API calls)
# =============================================================================


@pytest.mark.slow
class TestAnalyzeEndpoint:
    """Tests for the main /analyze endpoint.

    These tests are marked as slow because they involve full analysis
    with external API calls (GDELT, PubMed, RDAP).
    """

    def test_analyze_raw_text_returns_full_response(self, server, sample_raw_text):
        """Test analysis of raw text returns complete response structure."""
        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "raw_text",
                "content": sample_raw_text,
                "include_graphs": False
            },
            timeout=300  # 5 minutes for external API calls
        )

        assert response.status_code == 200
        data = response.json()

        # Check top-level structure
        assert "request_id" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["version"] == "1.0"

        # Check document preprocessing
        assert "document" in data
        doc = data["document"]
        assert doc["input_type"] == "raw_text"
        assert "display_text" in doc
        assert "language" in doc
        assert "sentences" in doc
        assert "tokens" in doc

        # Check all analyzer outputs
        assert "linguistic" in data
        assert "linguistic_risk_score" in data["linguistic"]
        assert 0 <= data["linguistic"]["linguistic_risk_score"] <= 1

        assert "statistical" in data
        assert "statistical_risk_score" in data["statistical"]
        assert 0 <= data["statistical"]["statistical_risk_score"] <= 1

        assert "source" in data
        assert "source_trust_score" in data["source"]
        assert 0 <= data["source"]["source_trust_score"] <= 1

        assert "claims" in data
        assert "claims" in data["claims"]
        assert "claim_items" in data["claims"]

        # Check aggregation output
        assert "aggregation" in data
        agg = data["aggregation"]
        assert "credibility_score" in agg
        assert 0 <= agg["credibility_score"] <= 100
        assert "verdict" in agg
        assert agg["verdict"] in ["Likely Real", "Suspicious", "Likely Fake"]
        assert "world_label" in agg
        assert agg["world_label"] in ["Real World", "Upside Down"]
        assert "confidence" in agg
        assert "reasoning_path" in agg

        # Check explanation
        assert "explanation" in data
        assert "verdict_text" in data["explanation"]
        assert "evidence_bullets" in data["explanation"]

    def test_analyze_suspicious_text_flags_concerns(self, server, sample_suspicious_text):
        """Test that suspicious text is flagged appropriately."""
        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "raw_text",
                "content": sample_suspicious_text,
                "include_graphs": False
            },
            timeout=300
        )

        assert response.status_code == 200
        data = response.json()

        # Suspicious content should have higher linguistic risk
        assert data["linguistic"]["linguistic_risk_score"] > 0.3

        # Should have highlighted phrases (clickbait, urgency, etc.)
        assert len(data["linguistic"]["highlighted_phrases"]) > 0

        # Verdict should reflect concerns
        assert data["aggregation"]["verdict"] in ["Suspicious", "Likely Fake"]
        assert data["aggregation"]["world_label"] == "Upside Down"

    def test_analyze_with_graphs_includes_graph_data(self, server, sample_raw_text):
        """Test that include_graphs=True returns graph structures."""
        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "raw_text",
                "content": sample_raw_text,
                "include_graphs": True
            },
            timeout=300
        )

        assert response.status_code == 200
        data = response.json()

        # Graphs should be present when requested
        assert "claim_graph" in data
        assert "evidence_graph" in data

        # If there are claims, graphs should have content
        if data["claims"]["claim_items"]:
            if data["claim_graph"]:
                assert "nodes" in data["claim_graph"]
                assert "edges" in data["claim_graph"]

    def test_analyze_social_post_type(self, server):
        """Test analysis of social media post type."""
        social_content = "Just heard that drinking lemon water cures cancer! Share this! #health #truth"

        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "social_post",
                "content": social_content,
                "include_graphs": False
            },
            timeout=300
        )

        assert response.status_code == 200
        data = response.json()
        assert data["document"]["input_type"] == "social_post"

    def test_analyze_medical_content_detects_topic(self, server, sample_medical_claim):
        """Test that medical content triggers medical topic detection."""
        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "raw_text",
                "content": sample_medical_claim,
                "include_graphs": False
            },
            timeout=300
        )

        assert response.status_code == 200
        data = response.json()

        # Should detect medical topic
        assert data["claims"]["medical_topic_detected"] is True

    def test_analyze_preserves_request_id_uniqueness(self, server, sample_short_text):
        """Test that each request gets a unique request ID."""
        response1 = requests.post(
            f"{server}/analyze",
            json={"input_type": "raw_text", "content": sample_short_text},
            timeout=300
        )
        response2 = requests.post(
            f"{server}/analyze",
            json={"input_type": "raw_text", "content": sample_short_text},
            timeout=300
        )

        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        assert data1["request_id"] != data2["request_id"]


# =============================================================================
# Graph Endpoint Tests (SLOW)
# =============================================================================


@pytest.mark.slow
class TestGraphEndpoints:
    """Tests for graph export and metrics endpoints."""

    @pytest.fixture
    def analysis_with_graphs(self, server, sample_raw_text):
        """Perform an analysis with graphs to use in graph tests."""
        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "raw_text",
                "content": sample_raw_text,
                "include_graphs": True
            },
            timeout=300
        )
        return response.json()

    def test_graph_export_json_format(self, server, analysis_with_graphs):
        """Test exporting graphs in JSON format."""
        # Skip if no claim graph in response
        if not analysis_with_graphs.get("claim_graph"):
            pytest.skip("No claim graph generated for this content")

        response = requests.post(
            f"{server}/graphs/export",
            json={
                "claim_graph": analysis_with_graphs["claim_graph"],
                "format": "json"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "format" in data
        assert data["format"] == "json"

    def test_graph_metrics_calculation(self, server, analysis_with_graphs):
        """Test calculating graph metrics."""
        # Skip if no claim graph in response
        if not analysis_with_graphs.get("claim_graph"):
            pytest.skip("No claim graph generated for this content")

        response = requests.post(
            f"{server}/graphs/metrics",
            json={
                "claim_graph": analysis_with_graphs["claim_graph"]
            }
        )

        assert response.status_code == 200
        data = response.json()
        # Should return metrics about the graph
        assert "claim_graph_metrics" in data


# =============================================================================
# Edge Case and Stress Tests (SLOW)
# =============================================================================


@pytest.mark.slow
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_analyze_unicode_content(self, server):
        """Test analysis of content with unicode characters."""
        unicode_content = """
        Les scientifiques ont découvert que le café peut améliorer la mémoire.
        日本の研究者は新しい治療法を発見しました。
        Исследователи нашли новый способ лечения рака.
        This study shows promising results for treating diabetes.
        """

        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "raw_text",
                "content": unicode_content,
                "include_graphs": False
            },
            timeout=300
        )

        assert response.status_code == 200
        data = response.json()
        assert data["document"]["display_text"]

    def test_analyze_long_content(self, server):
        """Test analysis of longer content."""
        # Generate longer content (multiple paragraphs)
        long_content = """
        Climate change is one of the most pressing issues of our time. Scientists
        from around the world have been studying the effects of global warming on
        our planet for decades. The Intergovernmental Panel on Climate Change (IPCC)
        has published numerous reports detailing the evidence for human-caused
        climate change and its potential impacts.

        Recent data from NASA and NOAA shows that global temperatures have risen
        significantly over the past century. The Arctic ice sheet has been melting
        at an accelerating rate, leading to rising sea levels. Many coastal cities
        are already experiencing increased flooding and erosion.

        The scientific consensus is clear: we need to reduce greenhouse gas
        emissions to prevent the worst effects of climate change. This will require
        significant changes in how we produce and consume energy, as well as changes
        in transportation, agriculture, and manufacturing.

        However, there is still debate about the best policies to address climate
        change. Some advocate for a carbon tax, while others prefer cap-and-trade
        systems. Still others argue that investment in new technologies is the key
        to solving this problem.
        """ * 3  # Repeat to make it longer

        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "raw_text",
                "content": long_content,
                "include_graphs": False
            },
            timeout=300
        )

        assert response.status_code == 200
        data = response.json()
        # Should process all content
        assert len(data["document"]["sentences"]) > 10

    def test_analyze_content_with_special_characters(self, server):
        """Test analysis of content with special characters."""
        special_content = """
        The price dropped by 50% after the announcement! Researchers found that
        the treatment costs around $500 per patient. The study's p-value was less
        than 0.05, indicating statistical significance. Results showed a 2-3x
        improvement in patient outcomes. Contact study@university.edu for more info.
        """

        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "raw_text",
                "content": special_content,
                "include_graphs": False
            },
            timeout=300
        )

        assert response.status_code == 200
        data = response.json()
        assert "aggregation" in data

    def test_analyze_minimum_valid_content(self, server):
        """Test analysis at minimum content length boundary."""
        # Minimum is typically around 10 characters
        min_content = "Test claim about scientific research findings."

        response = requests.post(
            f"{server}/analyze",
            json={
                "input_type": "raw_text",
                "content": min_content,
                "include_graphs": False
            },
            timeout=300
        )

        assert response.status_code == 200

    def test_concurrent_requests_return_unique_ids(self, server, sample_short_text):
        """Test that concurrent-ish requests get unique IDs."""
        responses = []
        for _ in range(3):  # Reduced from 5 to speed up
            response = requests.post(
                f"{server}/analyze",
                json={"input_type": "raw_text", "content": sample_short_text},
                timeout=300
            )
            responses.append(response)

        request_ids = [r.json()["request_id"] for r in responses]
        assert len(request_ids) == len(set(request_ids)), "Request IDs should be unique"


# =============================================================================
# Response Structure Validation Tests (SLOW)
# =============================================================================


@pytest.mark.slow
class TestResponseStructure:
    """Tests to validate the complete response structure."""

    @pytest.fixture(scope="class")
    def analysis_response(self, server, sample_raw_text):
        """Shared analysis response for structure tests."""
        response = requests.post(
            f"{server}/analyze",
            json={"input_type": "raw_text", "content": sample_raw_text},
            timeout=300
        )
        return response.json()

    def test_linguistic_output_structure(self, analysis_response):
        """Validate linguistic output structure matches schema."""
        linguistic = analysis_response["linguistic"]

        assert "linguistic_risk_score" in linguistic
        assert "signals" in linguistic
        assert "highlighted_phrases" in linguistic
        assert isinstance(linguistic["signals"], list)
        assert isinstance(linguistic["highlighted_phrases"], list)

    def test_statistical_output_structure(self, analysis_response):
        """Validate statistical output structure matches schema."""
        statistical = analysis_response["statistical"]

        assert "statistical_risk_score" in statistical
        assert "evidence" in statistical
        assert isinstance(statistical["evidence"], list)

    def test_source_intel_output_structure(self, analysis_response):
        """Validate source intel output structure matches schema."""
        source = analysis_response["source"]

        assert "source_trust_score" in source
        assert "source_flags" in source
        assert isinstance(source["source_flags"], list)

    def test_claims_output_structure(self, analysis_response):
        """Validate claims output structure matches schema."""
        claims = analysis_response["claims"]

        assert "claims" in claims
        assert "claim_items" in claims
        assert "medical_topic_detected" in claims
        assert isinstance(claims["claim_items"], list)

    def test_aggregation_output_structure(self, analysis_response):
        """Validate aggregation output structure matches schema."""
        agg = analysis_response["aggregation"]

        assert "credibility_score" in agg
        assert "verdict" in agg
        assert "world_label" in agg
        assert "confidence" in agg
        assert "confidence_calibrated" in agg
        assert "uncertainty_flags" in agg
        assert "reasoning_path" in agg

    def test_explanation_output_structure(self, analysis_response):
        """Validate explanation output structure matches schema."""
        explanation = analysis_response["explanation"]

        assert "verdict_text" in explanation
        assert "evidence_bullets" in explanation
        assert "assumptions" in explanation
        assert "blind_spots" in explanation
        assert isinstance(explanation["evidence_bullets"], list)

    def test_reasoning_path_has_rule_evaluations(self, analysis_response):
        """Test that reasoning path includes rule evaluations."""
        reasoning_path = analysis_response["aggregation"]["reasoning_path"]

        # Should have evaluated rules
        assert isinstance(reasoning_path, list)
        if reasoning_path:
            step = reasoning_path[0]
            assert "rule_id" in step
            assert "triggered" in step
