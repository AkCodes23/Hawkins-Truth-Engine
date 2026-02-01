import argparse
import json
import os
import sys
import requests

# Default server URL - can be overridden via environment or CLI
DEFAULT_URL = os.getenv("HTE_TEST_URL", "http://127.0.0.1:8000")

def run_test(url: str, content: str = None, input_type: str = "raw_text"):
    """Run E2E test against the specified server URL."""
    payload = {
        "content": content or "You won't believe what happened next! The mainstream media is covering up the truth and experts say it is definitely real. This secret was exposed and it's urgent that you share now!",
        "input_type": input_type,
        "include_graphs": True
    }
    
    try:
        print(f"=== Hawkins Truth Engine E2E Test ===")
        print(f"Target URL: {url}")
        print(f"Testing with: {payload['content'][:60]}...")
        print()
        
        r = requests.post(f"{url}/analyze", json=payload, timeout=120)
        
        if r.status_code == 200:
            data = r.json()
            print("=== RESULT: SUCCESS ===")
            print()
            
            # Aggregation
            agg = data.get("aggregation", {})
            print(f"Credibility Score: {agg.get('credibility_score', 'N/A')}")
            print(f"Verdict: {agg.get('verdict', 'N/A')}")
            print(f"World Label: {agg.get('world_label', 'N/A')}")
            print(f"Confidence: {agg.get('confidence', 0):.2f}")
            print()
            
            # Claims
            claims = data.get("claims", {})
            claims_summary = claims.get("claims", {})
            claim_items = claims.get("claim_items", [])
            print(f"Claims Extracted: {len(claim_items)}")
            print(f"  - Supported: {claims_summary.get('supported', 0)}")
            print(f"  - Unsupported: {claims_summary.get('unsupported', 0)}")
            print(f"  - Unverifiable: {claims_summary.get('unverifiable', 0)}")
            print()
            
            # Linguistic
            ling = data.get("linguistic", {})
            signals = ling.get("signals", [])
            print(f"Linguistic Risk Score: {ling.get('linguistic_risk_score', 0):.3f}")
            print(f"Signals Detected: {len(signals)}")
            for s in signals[:5]:
                print(f"  - {s.get('id', 'unknown')}: {s.get('evidence', '')[:60]}...")
            print()
            
            # Graphs
            if data.get("claim_graph"):
                nodes = data["claim_graph"].get("nodes", {})
                edges = data["claim_graph"].get("edges", {})
                print(f"Claim Graph: {len(nodes)} nodes, {len(edges)} edges")
            if data.get("evidence_graph"):
                edges = data["evidence_graph"].get("edges", {})
                print(f"Evidence Graph: {len(edges)} edges")
            print()
            
            print("=== E2E TEST PASSED ===")
            return True
            
        else:
            print(f"Error: HTTP {r.status_code}")
            print(r.text[:1000])
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to server at {url}")
        print("Make sure the server is running: python -m hawkins_truth_engine.app")
        return False
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E2E Test for Hawkins Truth Engine")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Server URL (default: {DEFAULT_URL})")
    parser.add_argument("--text", help="Custom text to test")
    parser.add_argument("--type", default="raw_text", choices=["raw_text", "url", "social_post"], help="Input type")
    
    args = parser.parse_args()
    
    success = run_test(args.url, args.text, args.type)
    sys.exit(0 if success else 1)
