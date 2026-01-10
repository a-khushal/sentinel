#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ctu13_loader import load_all_ctu13_scenarios
from models.ensemble import ThreatEnsemble
from models.dga_detector import FeatureBasedDGA

def main():
    print("=" * 60)
    print("CTU-13 Dataset Evaluation")
    print("=" * 60)
    print()
    
    print("Loading CTU-13 scenarios...")
    domains, labels, metadata = load_all_ctu13_scenarios()
    
    if len(domains) == 0:
        print("Error: No CTU-13 data found!")
        print("Please ensure CTU-13 .binetflow files are in data/ctu13/")
        return
    
    print(f"Loaded {len(domains)} samples")
    print(f"  Botnet: {sum(labels)}")
    print(f"  Normal: {len(labels) - sum(labels)}")
    print()
    
    print("Dataset Statistics:")
    for scenario in metadata['scenarios']:
        print(f"  {scenario['file']}:")
        print(f"    Flows: {scenario['flows']}")
        print(f"    DNS Queries: {scenario['dns_queries']}")
        print(f"    Botnet IPs: {scenario['botnet_ips']}")
        print(f"    Normal IPs: {scenario['normal_ips']}")
    print()
    
    print("Loading models...")
    ensemble = ThreatEnsemble(auto_load=True)
    print("Models loaded")
    print()
    
    print("Evaluating on CTU-13...")
    correct = 0
    total = min(100, len(domains))
    
    for i, (domain, label) in enumerate(zip(domains[:total], labels[:total])):
        result = ensemble.analyze_domain(domain)
        pred = 1 if result.get('confidence', 0) > 50 else 0
        
        if pred == label:
            correct += 1
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{total} samples...")
    
    accuracy = correct / total
    print()
    print(f"Accuracy on CTU-13: {accuracy:.4f} ({correct}/{total})")
    print("=" * 60)

if __name__ == "__main__":
    main()

