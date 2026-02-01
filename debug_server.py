import os
import subprocess
import time
import requests

def run_server_and_test():
    # Kill any existing server
    os.system("taskkill /f /im python.exe /t")
    
    # Start server with logging
    log_file = "server_debug.log"
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            ["python", "-u", "-m", "hawkins_truth_engine.app", "--port", "8001"],
            stdout=f,
            stderr=f,
            cwd="a:\\Projects\\Hawkins Truth Engine",
            env={**os.environ, "PYTHONPATH": "a:\\Projects\\Hawkins Truth Engine"}
        )
    
    print("Waiting for server to start...")
    time.sleep(5)
    
    # Trigger crash
    print("Triggering crash with URL input...")
    try:
        r = requests.post(
            "http://127.0.0.1:8001/analyze",
            json={"input_type": "url", "content": "https://example.com"},
            timeout=10
        )
        print(f"Status: {r.status_code}")
        print(f"Body: {r.text}")
    except Exception as e:
        print(f"Error calling server: {e}")
    
    # Keep server running for a bit to ensure log is written
    time.sleep(2)
    process.terminate()
    
    # Read log
    print("\n--- SERVER LOG ---")
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            print(f.read())
    else:
        print("Log file not found")

if __name__ == "__main__":
    run_server_and_test()
