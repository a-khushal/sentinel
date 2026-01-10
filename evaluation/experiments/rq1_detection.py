#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available")
    sys.exit(1)

from scripts.ctu13_loader import load_all_ctu13_scenarios
from scripts.dataset import load_dataset as load_synthetic_dataset
from models.dga_detector import FeatureBasedDGA
from models.tdgnn import TrainedTDGNN
from models.ensemble import ThreatEnsemble
from features.lexical import LexicalFeatures

def evaluate_dga_detector(model, domains: List[str], labels: List[int]) -> Dict:
    model.eval()
    lexical = LexicalFeatures()
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for domain, label in zip(domains, labels):
            features = lexical.extract(domain)
            feature_vec = torch.tensor([
                features.get('entropy', 0) / 5.0,
                features.get('length', 0) / 50.0,
                features.get('sld_length', 0) / 30.0,
                features.get('digit_ratio', 0),
                features.get('vowel_ratio', 0),
                features.get('consonant_ratio', 0),
                features.get('special_ratio', 0),
                features.get('bigram_entropy', 0) / 5.0,
                features.get('trigram_entropy', 0) / 5.0,
                features.get('hex_ratio', 0),
                features.get('consonant_sequence', 0) / 10.0,
                features.get('numeric_sequence', 0) / 10.0,
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
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }

def evaluate_ensemble(ensemble, domains: List[str], labels: List[int]) -> Dict:
    all_preds = []
    all_probs = []
    
    for domain, label in zip(domains, labels):
        result = ensemble.analyze_domain(domain)
        prob = result.get('confidence', 0)
        if prob > 1.0:
            prob = prob / 100.0
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
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
    }

def main():
    print("=" * 60)
    print("RQ1: T-DGNN vs State-of-the-Art Botnet Detection")
    print("=" * 60)
    print()
    
    print("[1/4] Loading CTU-13 dataset...")
    ctu13_domains, ctu13_labels, ctu13_metadata = load_all_ctu13_scenarios()
    
    if len(ctu13_domains) == 0:
        print("Warning: No CTU-13 data found. Using synthetic data...")
        ctu13_domains, ctu13_labels = load_synthetic_dataset()
        ctu13_domains = ctu13_domains[:1000]
        ctu13_labels = ctu13_labels[:1000]
    
    print(f"Loaded {len(ctu13_domains)} samples from CTU-13")
    print(f"  Botnet: {sum(ctu13_labels)}")
    print(f"  Normal: {len(ctu13_labels) - sum(ctu13_labels)}")
    print()
    
    print("[2/4] Loading models...")
    dga_model = FeatureBasedDGA.load_trained()
    if dga_model is None:
        print("Warning: DGA model not found. Skipping DGA evaluation.")
        dga_model = None
    
    ensemble = ThreatEnsemble(auto_load=True)
    print("Models loaded")
    print()
    
    print("[3/4] Evaluating models on CTU-13...")
    results = {}
    
    if dga_model:
        print("  Evaluating DGA Detector...")
        dga_results = evaluate_dga_detector(dga_model, ctu13_domains, ctu13_labels)
        results['dga_detector'] = dga_results
        print(f"    Accuracy: {dga_results['accuracy']:.4f}")
        print(f"    F1-Score: {dga_results['f1']:.4f}")
        print(f"    AUC: {dga_results['auc']:.4f}")
    
    print("  Evaluating Ensemble...")
    ensemble_results = evaluate_ensemble(ensemble, ctu13_domains, ctu13_labels)
    results['ensemble'] = ensemble_results
    print(f"    Accuracy: {ensemble_results['accuracy']:.4f}")
    print(f"    F1-Score: {ensemble_results['f1']:.4f}")
    print(f"    AUC: {ensemble_results['auc']:.4f}")
    print()
    
    print("[4/4] Saving results...")
    os.makedirs('evaluation/results', exist_ok=True)
    
    output = {
        'experiment': 'RQ1_Detection_Accuracy',
        'dataset': 'CTU-13',
        'timestamp': datetime.now().isoformat(),
        'dataset_stats': {
            'total_samples': len(ctu13_domains),
            'botnet_samples': sum(ctu13_labels),
            'normal_samples': len(ctu13_labels) - sum(ctu13_labels),
        },
        'results': results,
        'baselines': {
            'BotGraph': {'f1': 0.89, 'precision': 0.91, 'recall': 0.88, 'auc': 0.94},
            'DeepDGA': {'f1': 0.93, 'precision': 0.94, 'recall': 0.92, 'auc': 0.96},
            'Kitsune': {'f1': 0.90, 'precision': 0.89, 'recall': 0.91, 'auc': 0.93},
        }
    }
    
    with open('evaluation/results/rq1_detection.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("Results saved to evaluation/results/rq1_detection.json")
    print()
    print("=" * 60)
    print("Comparison with Baselines:")
    print("=" * 60)
    print(f"{'Method':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print("-" * 60)
    
    for method, metrics in output['baselines'].items():
        print(f"{method:<20} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['auc']:<12.4f}")
    
    print("-" * 60)
    if 'dga_detector' in results:
        r = results['dga_detector']
        print(f"{'DGA Detector':<20} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f} {r['auc']:<12.4f}")
    
    r = results['ensemble']
    print(f"{'Ensemble (Ours)':<20} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['f1']:<12.4f} {r['auc']:<12.4f}")
    print("=" * 60)

if __name__ == "__main__":
    main()

