import requests
import json

# This script simulates exactly what the frontend sends to the backend
def test_frontend_payload(input_type, content):
    url = "http://127.0.0.1:8005/analyze"
    payload = {
        "input_type": input_type,
        "content": content
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    print(f"Testing {input_type} with content: {content[:30]}...")
    try:
        r = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print("Response structure check:")
            print(f"- request_id: {data.get('request_id')}")
            print(f"- aggregation: {bool(data.get('aggregation'))}")
            print(f"- explanation: {bool(data.get('explanation'))}")
            print(f"- verdictStat exists? {data.get('aggregation', {}).get('verdict')}")
            # Check fields used in index.html line 1994
            print(f"- explanation.verdict_text: {data.get('explanation', {}).get('verdict_text')}")
        else:
            print(f"Error: {r.text}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    # Test cases representing common frontend inputs
    test_frontend_payload("raw_text", "This is a short test sentence for verify.")
    print("-" * 20)
    test_frontend_payload("url", "https://example.com")
    print("-" * 20)
    test_frontend_payload("social_post", "Check this out! It's a miracle cure for everything!")
