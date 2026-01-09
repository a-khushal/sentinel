import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
import hashlib
from .client import ModelUpdate

@dataclass
class AggregationResult:
    epoch: int
    model_hash: str
    num_clients: int
    total_samples: int
    avg_metrics: Dict[str, float]

class FederatedServer:
    def __init__(self, 
                 model: torch.nn.Module,
                 aggregation: str = 'fedavg',
                 min_clients: int = 2):
        self.global_model = model
        self.aggregation = aggregation
        self.min_clients = min_clients
        self.current_epoch = 0
        self.history: List[AggregationResult] = []
        self.pending_updates: List[ModelUpdate] = []
        
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.global_model.state_dict().items()}
    
    def receive_update(self, update: ModelUpdate):
        self.pending_updates.append(update)
    
    def aggregate(self) -> Optional[AggregationResult]:
        if len(self.pending_updates) < self.min_clients:
            return None
        
        updates = self.pending_updates
        self.pending_updates = []
        
        if self.aggregation == 'fedavg':
            aggregated = self._fedavg(updates)
        elif self.aggregation == 'fedprox':
            aggregated = self._fedprox(updates)
        else:
            aggregated = self._fedavg(updates)
        
        current_weights = self.get_global_weights()
        new_weights = {}
        for key in current_weights:
            new_weights[key] = current_weights[key] - aggregated[key]
        
        self.global_model.load_state_dict(new_weights)
        self.current_epoch += 1
        
        total_samples = sum(u.num_samples for u in updates)
        avg_loss = sum(u.metrics.get('loss', 0) * u.num_samples for u in updates) / total_samples
        
        result = AggregationResult(
            epoch=self.current_epoch,
            model_hash=self._compute_model_hash(),
            num_clients=len(updates),
            total_samples=total_samples,
            avg_metrics={'loss': avg_loss}
        )
        
        self.history.append(result)
        return result
    
    def _fedavg(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        total_samples = sum(u.num_samples for u in updates)
        
        aggregated = {}
        for key in updates[0].gradients:
            weighted_sum = torch.zeros_like(updates[0].gradients[key])
            for update in updates:
                weight = update.num_samples / total_samples
                weighted_sum += update.gradients[key] * weight
            aggregated[key] = weighted_sum
        
        return aggregated
    
    def _fedprox(self, updates: List[ModelUpdate], mu: float = 0.01) -> Dict[str, torch.Tensor]:
        base = self._fedavg(updates)
        
        global_weights = self.get_global_weights()
        for key in base:
            base[key] += mu * (global_weights[key] - base[key])
        
        return base
    
    def _compute_model_hash(self) -> str:
        weights = self.get_global_weights()
        serialized = b''
        for key in sorted(weights.keys()):
            serialized += weights[key].cpu().numpy().tobytes()
        return hashlib.sha256(serialized).hexdigest()[:16]
    
    def get_training_history(self) -> List[Dict]:
        return [
            {
                'epoch': r.epoch,
                'model_hash': r.model_hash,
                'num_clients': r.num_clients,
                'total_samples': r.total_samples,
                'metrics': r.avg_metrics
            }
            for r in self.history
        ]

