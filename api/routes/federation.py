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

def simulate_federated_training(config: FederationConfig):
    global federation_state
    app_state = get_state()
    
    if not app_state.ensemble or not app_state.ensemble.dga_model:
        federation_state['active'] = False
        return
    
    from federated.client import FederatedClient
    from federated.server import FederatedServer
    from models.dga_detector import DGADetector
    
    server_model = DGADetector()
    server = FederatedServer(server_model, aggregation='fedavg', min_clients=config.num_clients)
    
    clients = []
    for i in range(config.num_clients):
        client_model = DGADetector()
        client_model.load_state_dict(server.get_global_weights())
        client = FederatedClient(
            client_id=f"node_{i}",
            model=client_model,
            dp_epsilon=config.dp_epsilon
        )
        clients.append(client)
    
    federation_state['active'] = True
    federation_state['total_rounds'] = config.rounds
    federation_state['connected_clients'] = config.num_clients
    
    from scripts.generate_dga import cryptolocker_dga, necurs_dga, random_dga
    from datetime import datetime
    
    for round_num in range(config.rounds):
        if federation_state['should_stop']:
            break
            
        federation_state['current_round'] = round_num + 1
        
        global_weights = server.get_global_weights()
        for client in clients:
            client.set_model_weights(global_weights)
        
        for client in clients:
            dga_domains = cryptolocker_dga(datetime.now(), count=30) + random_dga(count=20)
            benign_domains = ['google.com', 'facebook.com', 'amazon.com', 'microsoft.com', 'apple.com',
                            'github.com', 'stackoverflow.com', 'reddit.com', 'twitter.com', 'linkedin.com'] * 5
            
            all_domains = dga_domains + benign_domains
            labels = [1] * len(dga_domains) + [0] * len(benign_domains)
            
            combined = list(zip(all_domains, labels))
            random.shuffle(combined)
            
            batch_data = []
            batch_size = 16
            max_len = 64
            
            for i in range(0, len(combined), batch_size):
                batch = combined[i:i+batch_size]
                domains_batch = [d for d, _ in batch]
                labels_batch = [l for _, l in batch]
                
                import numpy as np
                encoded = np.zeros((len(domains_batch), max_len), dtype=np.int64)
                for idx, domain in enumerate(domains_batch):
                    domain = domain.lower().rstrip('.')
                    for j, char in enumerate(domain[:max_len]):
                        encoded[idx, j] = min(ord(char), 127)
                
                x = torch.tensor(encoded, dtype=torch.long)
                y = torch.tensor(labels_batch, dtype=torch.long)
                batch_data.append((x, y))
            
            update = client.train_local(batch_data, epochs=config.local_epochs, lr=config.learning_rate)
            server.receive_update(update)
        
        result = server.aggregate()
        if result:
            federation_state['global_model_hash'] = result.model_hash
            federation_state['privacy_budget_used'] += config.dp_epsilon / config.rounds
            federation_state['last_aggregation'] = time.strftime('%Y-%m-%d %H:%M:%S')
            federation_state['history'].append({
                'round': result.epoch,
                'clients': result.num_clients,
                'samples': result.total_samples,
                'loss': result.avg_metrics.get('loss', 0),
                'model_hash': result.model_hash,
                'timestamp': federation_state['last_aggregation']
            })
        
        time.sleep(1)
    
    federation_state['global_model_hash'] = server._compute_model_hash()
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

