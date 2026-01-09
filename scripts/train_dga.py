#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed. Install with:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

from scripts.dataset import load_dataset, split_dataset
from features.lexical import LexicalFeatures

class DGADataset(Dataset):
    def __init__(self, domains, labels):
        self.domains = domains
        self.labels = labels
        self.lexical = LexicalFeatures()
        
    def __len__(self):
        return len(self.domains)
    
    def __getitem__(self, idx):
        domain = self.domains[idx]
        label = self.labels[idx]
        
        features = self.lexical.extract(domain)
        
        feature_vec = [
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
        ]
        
        return torch.tensor(feature_vec, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

class DGAClassifier(nn.Module):
    def __init__(self, input_dim=12, hidden_dims=[64, 32]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze(-1)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    tp = ((all_preds > 0.5) & (all_labels == 1)).sum()
    fp = ((all_preds > 0.5) & (all_labels == 0)).sum()
    fn = ((all_preds <= 0.5) & (all_labels == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return total_loss / len(loader), correct / total, precision, recall, f1

def main():
    print("=" * 60)
    print("DGA Detector Training")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\nLoading dataset...")
    domains, labels = load_dataset()
    splits = split_dataset(domains, labels)
    
    train_dataset = DGADataset(*splits['train'])
    val_dataset = DGADataset(*splits['val'])
    test_dataset = DGADataset(*splits['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    model = DGAClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    print("\nTraining...")
    epochs = 30
    best_val_f1 = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs('models/weights', exist_ok=True)
            torch.save(model.state_dict(), 'models/weights/dga_classifier.pt')
        
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")
    
    print("\nLoading best model and evaluating on test set...")
    model.load_state_dict(torch.load('models/weights/dga_classifier.pt'))
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, criterion, device)
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    print("=" * 60)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'train_samples': len(train_dataset),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_prec),
        'test_recall': float(test_rec),
        'test_f1': float(test_f1)
    }
    
    with open('models/weights/dga_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel saved to models/weights/dga_classifier.pt")

if __name__ == "__main__":
    main()

