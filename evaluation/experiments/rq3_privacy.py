#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
from typing import Dict, List
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available")
    sys.exit(1)

from scripts.ctu13_loader import load_all_ctu13_scenarios
from scripts.dataset import load_dataset as load_synthetic_dataset
from models.dga_detector import FeatureBasedDGA
from features.lexical import LexicalFeatures

def train_with_dp_noise(model, domains: List[str], labels: List[int], epsilon: float, epochs: int = 5) -> Dict:
    lexical = LexicalFeatures()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    
    features_list = []
    for domain in domains:
        feat = lexical.extract(domain)
        feature_vec = [
            feat.get('entropy', 0) / 5.0,
            feat.get('length', 0) / 50.0,
            feat.get('sld_length', 0) / 30.0,
            feat.get('digit_ratio', 0),
            feat.get('vowel_ratio', 0),
            feat.get('consonant_ratio', 0),
            feat.get('special_ratio', 0),
            feat.get('bigram_entropy', 0) / 5.0,
            feat.get('trigram_entropy', 0) / 5.0,
            feat.get('hex_ratio', 0),
            feat.get('consonant_sequence', 0) / 10.0,
            feat.get('numeric_sequence', 0) / 10.0,
        ]
        features_list.append(feature_vec)
    
    X = torch.tensor(features_list, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    
    noise_scale = 0.01 / epsilon if epsilon > 0 else 0.1
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad.add_(noise)
        
        optimizer.step()
    
    return evaluate_model(model, domains, labels)

def evaluate_model(model, domains: List[str], labels: List[int]) -> Dict:
    model.eval()
    lexical = LexicalFeatures()
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for domain, label in zip(domains, labels):
            feat = lexical.extract(domain)
            feature_vec = torch.tensor([
                feat.get('entropy', 0) / 5.0,
                feat.get('length', 0) / 50.0,
                feat.get('sld_length', 0) / 30.0,
                feat.get('digit_ratio', 0),
                feat.get('vowel_ratio', 0),
                feat.get('consonant_ratio', 0),
                feat.get('special_ratio', 0),
                feat.get('bigram_entropy', 0) / 5.0,
                feat.get('trigram_entropy', 0) / 5.0,
                feat.get('hex_ratio', 0),
                feat.get('consonant_sequence', 0) / 10.0,
                feat.get('numeric_sequence', 0) / 10.0,
            ], dtype=torch.float32).unsqueeze(0)
            
            prob = model(feature_vec).item()
            pred = 1 if prob > 0.5 else 0
            
            all_preds.append(pred)
            all_probs.append(prob)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(labels)
    all_probs = np.array(all_probs)
    
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    
    accuracy = (tp + tn) / len(labels) if len(labels) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.5
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
    }

def main():
    print("=" * 60)
    print("RQ3: Privacy-Utility Trade-off (Differential Privacy)")
    print("=" * 60)
    print()
    
    print("[1/4] Loading dataset...")
    domains, labels, _ = load_all_ctu13_scenarios()
    
    if len(domains) == 0:
        print("Warning: No CTU-13 data. Using synthetic data...")
        domains, labels = load_synthetic_dataset()
        domains = domains[:2000]
        labels = labels[:2000]
    
    train_size = int(len(domains) * 0.8)
    train_domains, train_labels = domains[:train_size], labels[:train_size]
    test_domains, test_labels = domains[train_size:], labels[train_size:]
    
    print(f"Training: {len(train_domains)}, Test: {len(test_domains)}")
    print()
    
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = {}
    
    print("[2/4] Training baseline (no DP)...")
    baseline_model = FeatureBasedDGA(input_dim=12)
    baseline_results = train_with_dp_noise(baseline_model, train_domains, train_labels, epsilon=100.0, epochs=5)
    test_baseline = evaluate_model(baseline_model, test_domains, test_labels)
    results['baseline'] = test_baseline
    print(f"  Baseline F1: {test_baseline['f1']:.4f}")
    print()
    
    print("[3/4] Testing different epsilon values...")
    for epsilon in epsilon_values:
        print(f"  Testing epsilon={epsilon}...")
        model = FeatureBasedDGA(input_dim=12)
        train_results = train_with_dp_noise(model, train_domains, train_labels, epsilon=epsilon, epochs=5)
        test_results = evaluate_model(model, test_domains, test_labels)
        results[f'epsilon_{epsilon}'] = {
            **test_results,
            'privacy_budget': epsilon,
        }
        print(f"    F1: {test_results['f1']:.4f}")
    print()
    
    print("[4/4] Saving results...")
    os.makedirs('evaluation/results', exist_ok=True)
    
    output = {
        'experiment': 'RQ3_Privacy_Utility_Tradeoff',
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'epsilon_values': epsilon_values,
        'baseline_f1': test_baseline['f1'],
    }
    
    with open('evaluation/results/rq3_privacy.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Results saved to evaluation/results/rq3_privacy.json")
    print()
    print("=" * 60)
    print("Privacy-Utility Trade-off:")
    print("=" * 60)
    print(f"{'Epsilon':<12} {'F1-Score':<12} {'Drop from Baseline':<20}")
    print("-" * 60)
    print(f"{'Baseline':<12} {test_baseline['f1']:<12.4f} {'0.00%':<20}")
    
    for epsilon in epsilon_values:
        key = f'epsilon_{epsilon}'
        f1 = results[key]['f1']
        drop = ((test_baseline['f1'] - f1) / test_baseline['f1']) * 100 if test_baseline['f1'] > 0 else 0
        print(f"{epsilon:<12.1f} {f1:<12.4f} {drop:<20.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()

