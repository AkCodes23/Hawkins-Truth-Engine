"""E2E Test script for Hawkins Truth Engine."""
import requests
import json
import sys

# Test data
payload = {
    "content": "You won't believe what happened next! The mainstream media is covering up the truth and experts say it is definitely real. This secret was exposed and it's urgent that you share now!",
    "input_type": "raw_text"
}

try:
    print("=== Hawkins Truth Engine E2E Test ===")
    print(f"Testing with: {payload['content'][:60]}...")
    print()
    
    r = requests.post("http://127.0.0.1:8000/analyze", json=payload, timeout=120)
    
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
        
        # Highlighted phrases
        highlights = ling.get("highlighted_phrases", [])
        if highlights:
            print("Highlighted Phrases:")
            for p in highlights:
                print(f"  - {p}")
        print()
        
        # Source analysis
        source = data.get("source", {})
        print(f"Domain Trust: {source.get('trust_score', 'N/A')}")
        print()
        
        # Uncertainty flags
        uncertainty = agg.get("uncertainty_flags", [])
        if uncertainty:
            print("Uncertainty Flags:")
            for f in uncertainty:
                print(f"  - {f}")
        
        print()
        print("=== E2E TEST PASSED ===")
        
    else:
        print(f"Error: HTTP {r.status_code}")
        print(r.text[:1000])
        sys.exit(1)
        
except requests.exceptions.ConnectionError:
    print("Error: Cannot connect to server at http://127.0.0.1:8000")
    print("Make sure the server is running: python -m hawkins_truth_engine.app")
    sys.exit(1)
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    sys.exit(1)
