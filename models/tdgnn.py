import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from typing import List, Dict, Optional, Tuple
import numpy as np

class GraphEncoder(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        
        self.dropout = dropout
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return x


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, temporal_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_weights = self.attention(temporal_embeddings)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        weighted = temporal_embeddings * attn_weights
        output = weighted.sum(dim=1)
        
        return output, attn_weights.squeeze(-1)


class TDGNN(nn.Module):
    def __init__(self,
                 node_features: int = 12,
                 hidden_dim: int = 64,
                 temporal_dim: int = 32,
                 num_classes: int = 2,
                 num_gnn_layers: int = 2,
                 num_temporal_steps: int = 5,
                 dropout: float = 0.3):
        super().__init__()
        
        self.num_temporal_steps = num_temporal_steps
        self.hidden_dim = hidden_dim
        
        self.graph_encoder = GraphEncoder(
            in_channels=node_features,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=num_gnn_layers,
            dropout=dropout
        )
        
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=temporal_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.temporal_attention = TemporalAttention(temporal_dim * 2)
        
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim + temporal_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.graph_classifier = nn.Sequential(
            nn.Linear(temporal_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                graphs: List[Data],
                node_indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        temporal_embeddings = []
        node_embeddings_list = []
        
        for graph in graphs:
            node_emb = self.graph_encoder(graph.x, graph.edge_index)
            node_embeddings_list.append(node_emb)
            
            graph_emb = global_mean_pool(node_emb, torch.zeros(node_emb.size(0), dtype=torch.long, device=node_emb.device))
            temporal_embeddings.append(graph_emb)
        
        temporal_seq = torch.stack(temporal_embeddings, dim=1)
        
        gru_out, _ = self.temporal_gru(temporal_seq)
        
        attended, attn_weights = self.temporal_attention(gru_out)
        
        graph_logits = self.graph_classifier(attended)
        
        node_logits = None
        if node_indices is not None and len(node_embeddings_list) > 0:
            last_node_emb = node_embeddings_list[-1]
            
            attended_expanded = attended.expand(last_node_emb.size(0), -1)
            combined = torch.cat([last_node_emb, attended_expanded], dim=1)
            
            node_logits = self.node_classifier(combined)
        
        return {
            'graph_logits': graph_logits,
            'node_logits': node_logits,
            'attention_weights': attn_weights,
            'node_embeddings': node_embeddings_list[-1] if node_embeddings_list else None,
        }
    
    def predict_graph(self, graphs: List[Data]) -> Dict:
        self.eval()
        
        with torch.no_grad():
            output = self(graphs)
            probs = F.softmax(output['graph_logits'], dim=1)
            pred = torch.argmax(probs, dim=1)
        
        return {
            'prediction': pred.item(),
            'is_malicious': bool(pred.item()),
            'confidence': float(probs[0, pred.item()].item()),
            'benign_prob': float(probs[0, 0].item()),
            'malicious_prob': float(probs[0, 1].item()),
            'attention_weights': output['attention_weights'].cpu().numpy(),
        }
    
    def predict_nodes(self, graphs: List[Data], node_mapping: Dict[str, int]) -> List[Dict]:
        self.eval()
        
        with torch.no_grad():
            num_nodes = graphs[-1].x.size(0)
            node_indices = torch.arange(num_nodes)
            
            output = self(graphs, node_indices)
            
            if output['node_logits'] is None:
                return []
            
            probs = F.softmax(output['node_logits'], dim=1)
            preds = torch.argmax(probs, dim=1)
        
        results = []
        inv_mapping = {v: k for k, v in node_mapping.items()}
        
        for i in range(num_nodes):
            node_id = inv_mapping.get(i, f"node_{i}")
            results.append({
                'node_id': node_id,
                'prediction': int(preds[i].item()),
                'is_infected': bool(preds[i].item()),
                'confidence': float(probs[i, preds[i].item()].item()),
                'benign_prob': float(probs[i, 0].item()),
                'infected_prob': float(probs[i, 1].item()),
            })
        
        return results


class TDGNNTrainer:
    def __init__(self, model: TDGNN, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_data: List[Tuple[List[Data], int]]) -> float:
        self.model.train()
        total_loss = 0
        
        for graphs, label in train_data:
            graphs = [g.to(self.device) for g in graphs]
            label = torch.tensor([label], device=self.device)
            
            self.optimizer.zero_grad()
            output = self.model(graphs)
            loss = self.criterion(output['graph_logits'], label)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_data)
    
    def evaluate(self, val_data: List[Tuple[List[Data], int]]) -> Dict:
        self.model.eval()
        correct = 0
        total = len(val_data)
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for graphs, label in val_data:
                graphs = [g.to(self.device) for g in graphs]
                
                output = self.model(graphs)
                pred = torch.argmax(output['graph_logits'], dim=1).item()
                
                correct += int(pred == label)
                all_preds.append(pred)
                all_labels.append(label)
        
        accuracy = correct / total if total > 0 else 0
        
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

