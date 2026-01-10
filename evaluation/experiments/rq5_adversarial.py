#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
import random
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

def poison_data(domains: List[str], labels: List[int], poison_ratio: float = 0.1) -> tuple:
    num_poison = int(len(domains) * poison_ratio)
    poisoned_domains = list(domains)
    poisoned_labels = list(labels)
    
    for i in range(num_poison):
        idx = random.randint(0, len(poisoned_domains) - 1)
        poisoned_labels[idx] = 1 - poisoned_labels[idx]
    
    return poisoned_domains, poisoned_labels

_poison_strength = 0.3

def poison_model_updates(model, poison_strength: float = None):
    global _poison_strength
    if poison_strength is not None:
        _poison_strength = poison_strength
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * _poison_strength
            param.grad.add_(noise)

def train_model(model, domains: List[str], labels: List[int], epochs: int = 10, use_poison: bool = False) -> Dict:
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
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y.squeeze(1))
        loss.backward()
        
        if use_poison:
            poison_model_updates(model)
        
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
    print("RQ5: Adversarial Robustness")
    print("=" * 60)
    print()
    
    print("[1/4] Loading dataset...")
    domains, labels, _ = load_all_ctu13_scenarios()
    
    if len(domains) == 0:
        print("Warning: No CTU-13 data. Using synthetic data...")
        domains, labels = load_synthetic_dataset()
        domains = domains[:2000]
        labels = labels[:2000]
    
    # Use synthetic DGA domains for better training
    from scripts.generate_dga import random_dga
    from scripts.dataset import load_dataset as load_synthetic
    
    synthetic_domains, synthetic_labels = load_synthetic()
    if len(synthetic_domains) > 5000:
        synthetic_domains = synthetic_domains[:5000]
        synthetic_labels = synthetic_labels[:5000]
    
    # Combine for training
    all_train_domains = list(domains[:len(domains)//2]) + list(synthetic_domains[:len(synthetic_domains)//2])
    all_train_labels = list(labels[:len(labels)//2]) + list(synthetic_labels[:len(synthetic_labels)//2])
    
    import random
    combined = list(zip(all_train_domains, all_train_labels))
    random.shuffle(combined)
    all_train_domains, all_train_labels = zip(*combined)
    all_train_domains, all_train_labels = list(all_train_domains), list(all_train_labels)
    
    train_size = int(len(all_train_domains) * 0.8)
    train_domains, train_labels = all_train_domains[:train_size], all_train_labels[:train_size]
    test_domains, test_labels = all_train_domains[train_size:], all_train_labels[train_size:]
    
    print(f"Training: {len(train_domains)} (CTU-13 + Synthetic), Test: {len(test_domains)}")
    print(f"  Train - Botnet: {sum(train_labels)}, Normal: {len(train_labels) - sum(train_labels)}")
    print(f"  Test - Botnet: {sum(test_labels)}, Normal: {len(test_labels) - sum(test_labels)}")
    print()
    
    print("[2/4] Loading pre-trained baseline model...")
    baseline_model = FeatureBasedDGA.load_trained()
    if baseline_model is None:
        print("  No pre-trained model found, training baseline...")
        baseline_model = FeatureBasedDGA(input_dim=12)
        baseline_results = train_model(baseline_model, train_domains, train_labels, epochs=20, use_poison=False)
    else:
        print("  Using pre-trained model")
    test_baseline = evaluate_model(baseline_model, test_domains, test_labels)
    print(f"  Baseline F1: {test_baseline['f1']:.4f}")
    print()
    
    print("[3/4] Testing data poisoning attacks...")
    poison_ratios = [0.05, 0.1, 0.2, 0.3]
    data_poison_results = {}
    
    for ratio in poison_ratios:
        print(f"  Testing {ratio*100:.0f}% data poisoning...")
        poisoned_domains, poisoned_labels = poison_data(train_domains, train_labels, poison_ratio=ratio)
        model = FeatureBasedDGA(input_dim=12)
        if baseline_model is not None:
            model.load_state_dict(baseline_model.state_dict())
        train_results = train_model(model, poisoned_domains, poisoned_labels, epochs=10, use_poison=False)
        test_results = evaluate_model(model, test_domains, test_labels)
        data_poison_results[f'ratio_{ratio}'] = test_results
        print(f"    F1: {test_results['f1']:.4f}")
    print()
    
    print("[4/4] Testing model poisoning attacks...")
    model_poison_results = {}
    for strength in [0.1, 0.3, 0.5]:
        print(f"  Testing model poisoning (strength={strength})...")
        model = FeatureBasedDGA(input_dim=12)
        if baseline_model is not None:
            model.load_state_dict(baseline_model.state_dict())
        # Set poison strength
        poison_model_updates(model, poison_strength=strength)
        train_results = train_model(model, train_domains, train_labels, epochs=10, use_poison=True)
        test_results = evaluate_model(model, test_domains, test_labels)
        model_poison_results[f'strength_{strength}'] = test_results
        print(f"    F1: {test_results['f1']:.4f}")
    print()
    
    print("Saving results...")
    os.makedirs('evaluation/results', exist_ok=True)
    
    output = {
        'experiment': 'RQ5_Adversarial_Robustness',
        'timestamp': datetime.now().isoformat(),
        'baseline': test_baseline,
        'data_poisoning': data_poison_results,
        'model_poisoning': model_poison_results,
    }
    
    with open('evaluation/results/rq5_adversarial.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Results saved to evaluation/results/rq5_adversarial.json")
    print()
    print("=" * 60)
    print("Adversarial Robustness Results:")
    print("=" * 60)
    print(f"Baseline F1: {test_baseline['f1']:.4f}")
    print()
    print("Data Poisoning:")
    for ratio in poison_ratios:
        key = f'ratio_{ratio}'
        f1 = data_poison_results[key]['f1']
        drop = ((test_baseline['f1'] - f1) / test_baseline['f1']) * 100 if test_baseline['f1'] > 0 else 0
        print(f"  {ratio*100:.0f}% poison: F1={f1:.4f} (drop: {drop:.2f}%)")
    print()
    print("Model Poisoning:")
    for strength in [0.1, 0.3, 0.5]:
        key = f'strength_{strength}'
        f1 = model_poison_results[key]['f1']
        drop = ((test_baseline['f1'] - f1) / test_baseline['f1']) * 100 if test_baseline['f1'] > 0 else 0
        print(f"  Strength {strength}: F1={f1:.4f} (drop: {drop:.2f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()

