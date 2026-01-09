import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'weights', 'dga_classifier.pt')

class FeatureBasedDGA(nn.Module):
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
    
    @classmethod
    def load_trained(cls, path: str = WEIGHTS_PATH) -> Optional['FeatureBasedDGA']:
        if not os.path.exists(path):
            return None
        model = cls()
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        model.eval()
        return model
    
    def predict_from_features(self, feature_vec: List[float]) -> float:
        self.eval()
        with torch.no_grad():
            x = torch.tensor([feature_vec], dtype=torch.float32)
            return float(self.model(x).item())

class CharEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 128, embed_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
    
    def forward(self, x):
        return self.embedding(x)

class DGADetector(nn.Module):
    def __init__(self, 
                 vocab_size: int = 128,
                 embed_dim: int = 64,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.3,
                 max_len: int = 128):
        super().__init__()
        
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        self.char_embed = CharEmbedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            num_layers=1, 
            batch_first=True,
            bidirectional=True
        )
        
        self.fc1 = nn.Linear(hidden_dim * 2 + embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 2)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape
        
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        char_emb = self.char_embed(x)
        pos_emb = self.pos_embed(positions)
        x_emb = self.layer_norm(char_emb + pos_emb)
        
        padding_mask = (x == 0)
        
        transformer_out = self.transformer(x_emb, src_key_padding_mask=padding_mask)
        
        lstm_out, (h_n, _) = self.lstm(x_emb)
        
        transformer_pooled = transformer_out.mean(dim=1)
        lstm_pooled = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        combined = torch.cat([transformer_pooled, lstm_pooled], dim=1)
        
        x = self.dropout(F.relu(self.fc1(combined)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
    def predict(self, domains: List[str]) -> List[Dict]:
        self.eval()
        device = next(self.parameters()).device
        
        encoded = self._encode_domains(domains)
        x = torch.tensor(encoded, dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
        
        results = []
        for i, domain in enumerate(domains):
            results.append({
                'domain': domain,
                'is_dga': bool(preds[i].item()),
                'confidence': float(probs[i, 1].item()),
                'benign_prob': float(probs[i, 0].item()),
                'dga_prob': float(probs[i, 1].item()),
            })
        
        return results
    
    def _encode_domains(self, domains: List[str]) -> np.ndarray:
        encoded = np.zeros((len(domains), self.max_len), dtype=np.int64)
        
        for i, domain in enumerate(domains):
            domain = domain.lower().rstrip('.')
            for j, char in enumerate(domain[:self.max_len]):
                encoded[i, j] = min(ord(char), self.vocab_size - 1)
        
        return encoded
    
    def get_attention_weights(self, domain: str) -> np.ndarray:
        self.eval()
        device = next(self.parameters()).device
        
        encoded = self._encode_domains([domain])
        x = torch.tensor(encoded, dtype=torch.long, device=device)
        
        char_emb = self.char_embed(x)
        positions = torch.arange(x.shape[1], device=device).unsqueeze(0)
        pos_emb = self.pos_embed(positions)
        x_emb = self.layer_norm(char_emb + pos_emb)
        
        return x_emb[0].detach().cpu().numpy()


class DGATrainer:
    def __init__(self, model: DGADetector, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader) -> Dict:
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)
                
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        accuracy = correct / total
        
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

