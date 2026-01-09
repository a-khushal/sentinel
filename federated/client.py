import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib

@dataclass
class ModelUpdate:
    epoch: int
    gradients: Dict[str, torch.Tensor]
    num_samples: int
    metrics: Dict[str, float]

class FederatedClient:
    def __init__(self, 
                 client_id: str,
                 model: torch.nn.Module,
                 device: str = 'cpu',
                 dp_epsilon: float = 1.0,
                 dp_delta: float = 1e-5):
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.current_epoch = 0
        self.local_data = []
        
    def set_model_weights(self, weights: Dict[str, torch.Tensor]):
        self.model.load_state_dict(weights)
    
    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.model.state_dict().items()}
    
    def train_local(self, 
                    data: List[Tuple[torch.Tensor, torch.Tensor]],
                    epochs: int = 1,
                    lr: float = 0.01) -> ModelUpdate:
        initial_weights = self.get_model_weights()
        
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for _ in range(epochs):
            for x, y in data:
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                
                self._clip_gradients(max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        final_weights = self.get_model_weights()
        gradients = {}
        for key in initial_weights:
            gradients[key] = initial_weights[key] - final_weights[key]
        
        noisy_gradients = self._add_dp_noise(gradients, len(data))
        
        self.current_epoch += 1
        
        return ModelUpdate(
            epoch=self.current_epoch,
            gradients=noisy_gradients,
            num_samples=len(data),
            metrics={'loss': total_loss / max(num_batches, 1)}
        )
    
    def _clip_gradients(self, max_norm: float):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
    
    def _add_dp_noise(self, 
                      gradients: Dict[str, torch.Tensor],
                      num_samples: int) -> Dict[str, torch.Tensor]:
        sensitivity = 2.0 / num_samples
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / self.dp_delta)) / self.dp_epsilon
        
        noisy_gradients = {}
        for key, grad in gradients.items():
            noise = torch.randn_like(grad) * noise_scale
            noisy_gradients[key] = grad + noise
        
        return noisy_gradients
    
    def compute_model_hash(self) -> str:
        weights = self.get_model_weights()
        serialized = b''
        for key in sorted(weights.keys()):
            serialized += weights[key].cpu().numpy().tobytes()
        return hashlib.sha256(serialized).hexdigest()
    
    def evaluate(self, data: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in data:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                pred = output.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        return {
            'accuracy': correct / max(total, 1),
            'total_samples': total
        }

