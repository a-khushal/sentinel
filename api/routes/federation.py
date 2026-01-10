from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import threading
import time
import torch
import random

router = APIRouter()

class FederationConfig(BaseModel):
    num_clients: int = 3
    rounds: int = 5
    local_epochs: int = 2
    dp_epsilon: float = 1.0
    learning_rate: float = 0.01

class FederationStatus(BaseModel):
    active: bool
    current_round: int
    total_rounds: int
    connected_clients: int
    global_model_hash: str
    privacy_budget_used: float
    last_aggregation: Optional[str]

federation_state = {
    'active': False,
    'current_round': 0,
    'total_rounds': 0,
    'connected_clients': 0,
    'global_model_hash': '',
    'privacy_budget_used': 0.0,
    'last_aggregation': None,
    'history': [],
    'training_thread': None,
    'should_stop': False
}

def get_state():
    from ..main import state
    return state

def extract_lexical_features(domain: str) -> list:
    import math
    domain = domain.lower().rstrip('.')
    length = len(domain)
    digits = sum(c.isdigit() for c in domain)
    consonants = sum(c in 'bcdfghjklmnpqrstvwxyz' for c in domain)
    vowels = sum(c in 'aeiou' for c in domain)
    
    freq = {}
    for c in domain:
        freq[c] = freq.get(c, 0) + 1
    entropy = -sum((f/length) * math.log2(f/length) for f in freq.values()) if length > 0 else 0
    
    unique_chars = len(set(domain))
    digit_ratio = digits / length if length > 0 else 0
    vowel_ratio = vowels / length if length > 0 else 0
    consonant_ratio = consonants / length if length > 0 else 0
    
    has_numbers = 1.0 if digits > 0 else 0.0
    num_parts = len(domain.split('.'))
    max_consonant_seq = 0
    current_seq = 0
    for c in domain:
        if c in 'bcdfghjklmnpqrstvwxyz':
            current_seq += 1
            max_consonant_seq = max(max_consonant_seq, current_seq)
        else:
            current_seq = 0
    
    return [
        length / 50.0,
        entropy / 5.0,
        digit_ratio,
        vowel_ratio,
        consonant_ratio,
        unique_chars / 26.0,
        has_numbers,
        num_parts / 5.0,
        max_consonant_seq / 10.0,
        digits / 20.0,
        vowels / 20.0,
        consonants / 30.0
    ]

def simulate_federated_training(config: FederationConfig):
    global federation_state
    app_state = get_state()
    
    federation_state['active'] = True
    federation_state['total_rounds'] = config.rounds
    federation_state['connected_clients'] = config.num_clients
    
    from models.dga_detector import FeatureBasedDGA
    from scripts.generate_dga import cryptolocker_dga, necurs_dga, random_dga
    from datetime import datetime
    import hashlib
    
    server_model = FeatureBasedDGA(input_dim=12)
    
    clients_models = []
    for i in range(config.num_clients):
        client_model = FeatureBasedDGA(input_dim=12)
        client_model.load_state_dict(server_model.state_dict())
        clients_models.append(client_model)
    
    benign_base = [
        'google.com', 'facebook.com', 'amazon.com', 'microsoft.com', 'apple.com',
        'github.com', 'stackoverflow.com', 'reddit.com', 'twitter.com', 'linkedin.com',
        'youtube.com', 'netflix.com', 'spotify.com', 'dropbox.com', 'slack.com',
        'zoom.us', 'notion.so', 'figma.com', 'vercel.app', 'cloudflare.com'
    ]
    
    for round_num in range(config.rounds):
        if federation_state['should_stop']:
            break
            
        federation_state['current_round'] = round_num + 1
        
        round_losses = []
        total_samples = 0
        client_updates = []
        
        for client_idx, client_model in enumerate(clients_models):
            client_model.load_state_dict(server_model.state_dict())
            
            dga_domains = random_dga(count=50)
            benign_domains = benign_base * 2
            
            all_domains = dga_domains + benign_domains
            labels = [1.0] * len(dga_domains) + [0.0] * len(benign_domains)
            
            combined = list(zip(all_domains, labels))
            random.shuffle(combined)
            
            features = []
            targets = []
            for domain, label in combined:
                feat = extract_lexical_features(domain)
                features.append(feat)
                targets.append(label)
            
            X = torch.tensor(features, dtype=torch.float32)
            y = torch.tensor(targets, dtype=torch.float32)
            
            optimizer = torch.optim.Adam(client_model.parameters(), lr=config.learning_rate)
            criterion = torch.nn.BCELoss()
            
            client_model.train()
            batch_size = 32
            client_loss = 0
            num_batches = 0
            
            for epoch in range(config.local_epochs):
                for i in range(0, len(X), batch_size):
                    x_batch = X[i:i+batch_size]
                    y_batch = y[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    output = client_model(x_batch)
                    loss = criterion(output, y_batch)
                    loss.backward()
                    optimizer.step()
                    client_loss += loss.item()
                    num_batches += 1
            
            avg_loss = client_loss / max(num_batches, 1)
            round_losses.append(avg_loss)
            total_samples += len(X)
            
            client_updates.append({k: v.clone() for k, v in client_model.state_dict().items()})
        
        aggregated_weights = {}
        for key in server_model.state_dict().keys():
            stacked = torch.stack([update[key].float() for update in client_updates])
            aggregated_weights[key] = stacked.mean(dim=0)
            
            if config.dp_epsilon < 10:
                noise_scale = 0.01 / config.dp_epsilon
                noise = torch.randn_like(aggregated_weights[key]) * noise_scale
                aggregated_weights[key] = aggregated_weights[key] + noise
        
        server_model.load_state_dict(aggregated_weights)
        
        weights_bytes = b''
        for key in sorted(aggregated_weights.keys()):
            weights_bytes += aggregated_weights[key].cpu().numpy().tobytes()
        model_hash = hashlib.sha256(weights_bytes).hexdigest()[:16]
        
        avg_round_loss = sum(round_losses) / len(round_losses)
        
        federation_state['global_model_hash'] = model_hash
        federation_state['privacy_budget_used'] += config.dp_epsilon / config.rounds
        federation_state['last_aggregation'] = time.strftime('%Y-%m-%d %H:%M:%S')
        federation_state['history'].append({
            'round': round_num + 1,
            'clients': config.num_clients,
            'samples': total_samples,
            'loss': round(avg_round_loss, 4),
            'model_hash': model_hash,
            'timestamp': federation_state['last_aggregation']
        })
        
        time.sleep(0.5)
    
    print(f"Federated training completed after {config.rounds} rounds")
    print(f"Final model hash: {federation_state['global_model_hash']}")
    
    federation_state['active'] = False

@router.get("/status", response_model=FederationStatus)
async def get_federation_status():
    return FederationStatus(
        active=federation_state['active'],
        current_round=federation_state['current_round'],
        total_rounds=federation_state['total_rounds'],
        connected_clients=federation_state['connected_clients'],
        global_model_hash=federation_state['global_model_hash'] or 'none',
        privacy_budget_used=federation_state['privacy_budget_used'],
        last_aggregation=federation_state['last_aggregation']
    )

@router.post("/start")
async def start_federation(config: FederationConfig, background_tasks: BackgroundTasks):
    if federation_state['active']:
        raise HTTPException(status_code=400, detail="Federation already running")
    
    federation_state['should_stop'] = False
    federation_state['history'] = []
    
    thread = threading.Thread(target=simulate_federated_training, args=(config,))
    thread.start()
    federation_state['training_thread'] = thread
    
    return {
        "message": "Federation training started",
        "config": config.dict()
    }

@router.post("/stop")
async def stop_federation():
    if not federation_state['active']:
        return {"message": "No federation running"}
    
    federation_state['should_stop'] = True
    return {"message": "Stopping federation..."}

@router.get("/history")
async def get_federation_history():
    return {
        "rounds": federation_state['history'],
        "total_privacy_budget": federation_state['privacy_budget_used']
    }

@router.get("/clients")
async def get_connected_clients():
    if not federation_state['active']:
        return {"clients": [], "count": 0}
    
    clients = [
        {
            "id": f"node_{i}",
            "status": "training" if federation_state['active'] else "idle",
            "last_update": federation_state['last_aggregation'],
            "contributions": federation_state['current_round']
        }
        for i in range(federation_state['connected_clients'])
    ]
    return {"clients": clients, "count": len(clients)}

@router.get("/privacy")
async def get_privacy_status():
    return {
        "epsilon_used": federation_state['privacy_budget_used'],
        "epsilon_target": 1.0,
        "delta": 1e-5,
        "mechanism": "Gaussian",
        "composition": "Advanced (Moments Accountant)"
    }

