#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random
import string

from models.dga_detector import DGADetector, DGATrainer

def generate_training_data(num_samples: int = 10000):
    benign_words = ['google', 'facebook', 'amazon', 'microsoft', 'github', 
                   'stackoverflow', 'reddit', 'twitter', 'youtube', 'linkedin',
                   'netflix', 'spotify', 'wikipedia', 'yahoo', 'bing', 'apple',
                   'instagram', 'pinterest', 'tumblr', 'wordpress', 'medium']
    tlds = ['com', 'org', 'net', 'io', 'co', 'edu', 'gov']
    
    domains = []
    labels = []
    
    for _ in range(num_samples // 2):
        word = random.choice(benign_words)
        if random.random() > 0.5:
            word = word + random.choice(['mail', 'api', 'cdn', 'static', 'www', ''])
        domain = f"{word}.{random.choice(tlds)}"
        domains.append(domain)
        labels.append(0)
    
    for _ in range(num_samples // 2):
        length = random.randint(10, 30)
        chars = string.ascii_lowercase + string.digits
        domain = ''.join(random.choice(chars) for _ in range(length))
        tld = random.choice(['com', 'net', 'org', 'xyz', 'top', 'info'])
        domains.append(f"{domain}.{tld}")
        labels.append(1)
    
    combined = list(zip(domains, labels))
    random.shuffle(combined)
    domains, labels = zip(*combined)
    
    return list(domains), list(labels)

def encode_domains(domains, max_len=128):
    encoded = np.zeros((len(domains), max_len), dtype=np.int64)
    for i, domain in enumerate(domains):
        domain = domain.lower().rstrip('.')
        for j, char in enumerate(domain[:max_len]):
            encoded[i, j] = min(ord(char), 127)
    return encoded

def main():
    print("=" * 60)
    print("SENTINEL - DGA Detector Training")
    print("=" * 60)
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    print("[1/4] Generating training data...")
    train_domains, train_labels = generate_training_data(8000)
    val_domains, val_labels = generate_training_data(2000)
    print(f"      Training samples: {len(train_domains)}")
    print(f"      Validation samples: {len(val_domains)}")
    print()
    
    print("[2/4] Encoding domains...")
    X_train = torch.tensor(encode_domains(train_domains), dtype=torch.long)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_val = torch.tensor(encode_domains(val_domains), dtype=torch.long)
    y_val = torch.tensor(val_labels, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    print("      Data encoded and loaded")
    print()
    
    print("[3/4] Training model...")
    model = DGADetector(
        vocab_size=128,
        embed_dim=64,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.3
    )
    
    trainer = DGATrainer(model, device=device)
    
    num_epochs = 10
    best_f1 = 0
    
    for epoch in range(num_epochs):
        train_loss = trainer.train_epoch(train_loader)
        metrics = trainer.evaluate(val_loader)
        
        print(f"      Epoch {epoch+1}/{num_epochs}: loss={train_loss:.4f}, "
              f"acc={metrics['accuracy']:.4f}, f1={metrics['f1']:.4f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            os.makedirs('models/weights', exist_ok=True)
            trainer.save('models/weights/dga_detector.pt')
    
    print()
    print("[4/4] Final evaluation...")
    final_metrics = trainer.evaluate(val_loader)
    print(f"      Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"      Precision: {final_metrics['precision']:.4f}")
    print(f"      Recall:    {final_metrics['recall']:.4f}")
    print(f"      F1 Score:  {final_metrics['f1']:.4f}")
    print()
    
    print("=" * 60)
    print("Training complete!")
    print(f"Model saved to: models/weights/dga_detector.pt")
    print("=" * 60)
    
    print()
    print("Testing on sample domains:")
    test_domains = [
        "google.com",
        "facebook.com",
        "x7k9m2p4q8w3e5r1.com",
        "asjhd7823hdkjashd.xyz",
        "github.io",
        "qw3rt7yu1op2asdf.net"
    ]
    
    results = model.predict(test_domains)
    for r in results:
        status = "DGA" if r['is_dga'] else "BENIGN"
        print(f"  {r['domain']:<35} [{status}] conf={r['confidence']:.3f}")

if __name__ == "__main__":
    main()

