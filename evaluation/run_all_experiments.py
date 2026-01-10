#!/usr/bin/env python3
import sys
import os
import subprocess
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_experiment(script_name, description):
    print("\n" + "=" * 70)
    print(f"Running {description}")
    print("=" * 70)
    print()
    
    script_path = os.path.join(os.path.dirname(__file__), 'experiments', script_name)
    
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found!")
        return False
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(script_path))),
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"\n✓ {description} completed in {elapsed:.2f}s")
            return True
        else:
            print(f"\n✗ {description} failed (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"\n✗ Error running {description}: {e}")
        return False

def main():
    print("=" * 70)
    print("SENTINEL - Running All Evaluation Experiments")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    experiments = [
        ('rq1_detection.py', 'RQ1: Detection Accuracy'),
        ('rq2_federated.py', 'RQ2: Federated vs Centralized'),
        ('rq3_privacy.py', 'RQ3: Privacy-Utility Trade-off'),
        ('rq4_blockchain.py', 'RQ4: Blockchain Performance'),
        ('rq5_adversarial.py', 'RQ5: Adversarial Robustness'),
    ]
    
    results = {}
    total_start = time.time()
    
    for script, description in experiments:
        success = run_experiment(script, description)
        results[description] = success
        
        if not success:
            print(f"\n⚠ Warning: {description} failed. Continuing with next experiment...")
            time.sleep(2)
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    
    for desc, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status:12} {desc}")
    
    print()
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Results saved to: evaluation/results/")
    print("=" * 70)

if __name__ == "__main__":
    main()

