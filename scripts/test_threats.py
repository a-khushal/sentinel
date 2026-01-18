#!/usr/bin/env python3
"""
Script to inject test threats into the API for testing Threat Monitor UI.
Run this while the backend is running.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import time
from scripts.generate_dga import random_dga, cryptolocker_dga, necurs_dga

API_BASE = "http://localhost:8000/api/v1"

def inject_test_threats():
    print("Injecting test threats into Threat Monitor...")
    print(f"API: {API_BASE}")
    print()
    
    # Generate some DGA domains
    dga_domains = random_dga(count=10) + cryptolocker_dga(count=5) + necurs_dga(count=5)
    benign_domains = [
        "google.com", "facebook.com", "github.com", "stackoverflow.com",
        "reddit.com", "youtube.com", "amazon.com", "microsoft.com"
    ]
    
    all_domains = dga_domains + benign_domains
    
    print(f"Analyzing {len(all_domains)} domains...")
    
    threats_found = 0
    for i, domain in enumerate(all_domains):
        try:
            response = requests.post(
                f"{API_BASE}/threats/analyze",
                json={"domain": domain},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('is_suspicious'):
                    threats_found += 1
                    print(f"  ✓ {domain[:50]:<50} [THREAT] conf={result['confidence']:.2f}")
                else:
                    print(f"  - {domain[:50]:<50} [clean]")
            else:
                print(f"  ✗ {domain[:50]:<50} [error: {response.status_code}]")
            
            time.sleep(0.1)  # Small delay
            
        except requests.exceptions.ConnectionError:
            print(f"\n✗ Error: Cannot connect to API at {API_BASE}")
            print("  Make sure the backend is running: python -m uvicorn api.main:app --port 8000")
            return
        except Exception as e:
            print(f"  ✗ Error analyzing {domain}: {e}")
    
    print()
    print(f"✓ Injected {threats_found} threats")
    print(f"  Check Threat Monitor at: http://localhost:3000/threats")
    print()
    
    # Get stats
    try:
        stats = requests.get(f"{API_BASE}/stats", timeout=5).json()
        print("Current Stats:")
        print(f"  Queries Processed: {stats.get('total_queries_processed', 0)}")
        print(f"  Threats Detected: {stats.get('threats_detected', 0)}")
        print(f"  Domains Analyzed: {stats.get('domains_analyzed', 0)}")
    except:
        pass

if __name__ == "__main__":
    inject_test_threats()



