import os
import random
from typing import List, Tuple

def load_dga_domains(path: str = "data/dga/dga_domains.txt") -> List[str]:
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def load_benign_domains(path: str = "data/benign/top-1m.csv", limit: int = 50000) -> List[str]:
    domains = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            parts = line.strip().split(',')
            if len(parts) >= 2:
                domains.append(parts[1])
    return domains

def load_dataset(dga_path: str = "data/dga/dga_domains.txt",
                 benign_path: str = "data/benign/top-1m.csv",
                 benign_limit: int = 50000,
                 balance: bool = True) -> Tuple[List[str], List[int]]:
    dga = load_dga_domains(dga_path)
    benign = load_benign_domains(benign_path, benign_limit)
    
    if balance:
        min_size = min(len(dga), len(benign))
        dga = random.sample(dga, min_size)
        benign = random.sample(benign, min_size)
    
    domains = dga + benign
    labels = [1] * len(dga) + [0] * len(benign)
    
    combined = list(zip(domains, labels))
    random.shuffle(combined)
    domains, labels = zip(*combined)
    
    return list(domains), list(labels)

def split_dataset(domains: List[str], labels: List[int], 
                  train_ratio: float = 0.8, val_ratio: float = 0.1):
    n = len(domains)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return {
        'train': (domains[:train_end], labels[:train_end]),
        'val': (domains[train_end:val_end], labels[train_end:val_end]),
        'test': (domains[val_end:], labels[val_end:])
    }

if __name__ == "__main__":
    domains, labels = load_dataset()
    print(f"Total samples: {len(domains)}")
    print(f"DGA: {sum(labels)}, Benign: {len(labels) - sum(labels)}")
    
    splits = split_dataset(domains, labels)
    print(f"Train: {len(splits['train'][0])}")
    print(f"Val: {len(splits['val'][0])}")
    print(f"Test: {len(splits['test'][0])}")
    
    print("\nSample DGA:", [d for d, l in zip(domains[:20], labels[:20]) if l == 1][:3])
    print("Sample Benign:", [d for d, l in zip(domains[:20], labels[:20]) if l == 0][:3])

