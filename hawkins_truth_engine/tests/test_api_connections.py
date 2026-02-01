"""
API Connection Test Script for Hawkins Truth Engine.

Tests all external API connections to verify they are working.
"""

import asyncio
import os
from datetime import datetime


async def test_gdelt_api():
    """Test GDELT API (no authentication required)."""
    print("\n" + "="*60)
    print("Testing GDELT API (News Search)")
    print("="*60)
    
    try:
        from hawkins_truth_engine.external.gdelt import gdelt_doc_search
        
        result = await gdelt_doc_search("climate change", maxrecords=3)
        
        if "data" in result:
            articles = result["data"].get("articles", [])
            print(f"✓ GDELT API is WORKING")
            print(f"  - Returned {len(articles)} articles")
            if articles:
                print(f"  - Sample: {articles[0].get('title', 'N/A')[:50]}...")
            return True
        else:
            print(f"✗ GDELT API returned unexpected response")
            return False
            
    except Exception as e:
        print(f"✗ GDELT API FAILED: {type(e).__name__}: {e}")
        return False


async def test_ncbi_pubmed_api():
    """Test NCBI/PubMed API (optional API key for higher rate limits)."""
    print("\n" + "="*60)
    print("Testing NCBI/PubMed API (Medical Research)")
    print("="*60)
    
    try:
        from hawkins_truth_engine.external.ncbi import pubmed_esearch
        from hawkins_truth_engine.config import NCBI_API_KEY
        
        if NCBI_API_KEY:
            print(f"  API Key: Configured (higher rate limits)")
        else:
            print(f"  API Key: Not configured (using public rate limits)")
        
        result = await pubmed_esearch("vaccine efficacy", retmax=3)
        
        if "data" in result:
            esearch_result = result["data"].get("esearchresult", {})
            count = esearch_result.get("count", 0)
            ids = esearch_result.get("idlist", [])
            print(f"✓ NCBI/PubMed API is WORKING")
            print(f"  - Found {count} total results")
            print(f"  - Returned {len(ids)} PMIDs: {ids[:3]}...")
            return True
        else:
            print(f"✗ NCBI/PubMed API returned unexpected response")
            return False
            
    except Exception as e:
        print(f"✗ NCBI/PubMed API FAILED: {type(e).__name__}: {e}")
        return False


async def test_rdap_api():
    """Test RDAP API (no authentication required)."""
    print("\n" + "="*60)
    print("Testing RDAP API (Domain Lookup)")
    print("="*60)
    
    try:
        from hawkins_truth_engine.external.rdap import rdap_domain
        
        result = await rdap_domain("google.com")
        
        if "data" in result:
            data = result["data"]
            name = data.get("ldhName", "N/A")
            status = data.get("status", [])
            print(f"✓ RDAP API is WORKING")
            print(f"  - Domain: {name}")
            print(f"  - Status: {status[:2]}...")
            return True
        else:
            print(f"✗ RDAP API returned unexpected response")
            return False
            
    except Exception as e:
        print(f"✗ RDAP API FAILED: {type(e).__name__}: {e}")
        return False


async def test_tavily_api():
    """Test Tavily API (requires API key)."""
    print("\n" + "="*60)
    print("Testing Tavily API (Web Search)")
    print("="*60)
    
    try:
        from hawkins_truth_engine.config import TAVILY_API_KEY
        
        if not TAVILY_API_KEY:
            print("⚠ Tavily API Key NOT CONFIGURED")
            print("  Set HTE_TAVILY_API_KEY environment variable")
            print("  Get a free key at: https://tavily.com/")
            return None  # Not configured, not a failure
        
        print(f"  API Key: Configured (****{TAVILY_API_KEY[-4:]})")
        
        from hawkins_truth_engine.external.tavily import tavily_search
        
        result = await tavily_search("artificial intelligence latest news", max_results=3)
        
        if "data" in result:
            data = result["data"]
            results = data.get("results", [])
            print(f"✓ Tavily API is WORKING")
            print(f"  - Returned {len(results)} results")
            if results:
                print(f"  - Sample: {results[0].get('title', 'N/A')[:50]}...")
            return True
        else:
            print(f"✗ Tavily API returned unexpected response")
            return False
            
    except Exception as e:
        print(f"✗ Tavily API FAILED: {type(e).__name__}: {e}")
        return False


async def main():
    """Run all API tests."""
    print("\n" + "#"*60)
    print("  HAWKINS TRUTH ENGINE - EXTERNAL API CONNECTION TEST")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*60)
    
    results = {}
    
    # Test each API
    results["GDELT"] = await test_gdelt_api()
    results["NCBI/PubMed"] = await test_ncbi_pubmed_api()
    results["RDAP"] = await test_rdap_api()
    results["Tavily"] = await test_tavily_api()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for api, status in results.items():
        if status is True:
            emoji = "✓"
            text = "WORKING"
        elif status is False:
            emoji = "✗"
            text = "FAILED"
        else:
            emoji = "⚠"
            text = "NOT CONFIGURED"
        print(f"  {emoji} {api}: {text}")
    
    # Count failures
    failures = sum(1 for v in results.values() if v is False)
    not_configured = sum(1 for v in results.values() if v is None)
    
    print("\n" + "="*60)
    if failures == 0:
        print("✓ All configured APIs are working!")
    else:
        print(f"✗ {failures} API(s) failed - see details above")
    
    if not_configured > 0:
        print(f"⚠ {not_configured} API(s) not configured (optional)")
    print("="*60 + "\n")
    
    return failures == 0


if __name__ == "__main__":
    asyncio.run(main())
