#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
from typing import Dict, List
from datetime import datetime
import time

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
from api.routes.federation import simulate_federated_training, FederationConfig
from features.lexical import LexicalFeatures

def train_centralized(model, domains: List[str], labels: List[int], epochs: int = 5) -> Dict:
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
    print("RQ2: Federated Learning vs Centralized Training")
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
    
    print("[2/4] Training centralized model...")
    centralized_model = FeatureBasedDGA(input_dim=12)
    centralized_start = time.time()
    centralized_results = train_centralized(centralized_model, train_domains, train_labels, epochs=5)
    centralized_time = time.time() - centralized_start
    
    test_results_centralized = evaluate_model(centralized_model, test_domains, test_labels)
    print(f"  Centralized - F1: {test_results_centralized['f1']:.4f}, Time: {centralized_time:.2f}s")
    print()
    
    print("[3/4] Simulating federated training...")
    federated_model = FeatureBasedDGA(input_dim=12)
    federated_start = time.time()
    
    config = FederationConfig(
        num_clients=3,
        rounds=5,
        local_epochs=2,
        dp_epsilon=10.0,
        learning_rate=0.01
    )
    
    simulate_federated_training(config)
    federated_time = time.time() - federated_start
    
    test_results_federated = evaluate_model(federated_model, test_domains, test_labels)
    print(f"  Federated - F1: {test_results_federated['f1']:.4f}, Time: {federated_time:.2f}s")
    print()
    
    print("[4/4] Saving results...")
    os.makedirs('evaluation/results', exist_ok=True)
    
    accuracy_drop = test_results_centralized['f1'] - test_results_federated['f1']
    time_overhead = federated_time - centralized_time
    
    output = {
        'experiment': 'RQ2_Federated_vs_Centralized',
        'timestamp': datetime.now().isoformat(),
        'centralized': {
            **test_results_centralized,
            'training_time': centralized_time,
        },
        'federated': {
            **test_results_federated,
            'training_time': federated_time,
        },
        'comparison': {
            'accuracy_drop': accuracy_drop,
            'time_overhead': time_overhead,
            'relative_drop': accuracy_drop / test_results_centralized['f1'] if test_results_centralized['f1'] > 0 else 0,
        }
    }
    
    with open('evaluation/results/rq2_federated.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Results saved to evaluation/results/rq2_federated.json")
    print()
    print("=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"Centralized F1: {test_results_centralized['f1']:.4f}")
    print(f"Federated F1:  {test_results_federated['f1']:.4f}")
    print(f"Accuracy Drop:  {accuracy_drop:.4f} ({output['comparison']['relative_drop']*100:.2f}%)")
    print(f"Time Overhead:  {time_overhead:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    main()

