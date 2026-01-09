#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import numpy as np
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import SAGEConv, global_mean_pool
    HAS_TORCH = True
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install torch torch-geometric")
    sys.exit(1)

class TDGNN(nn.Module):
    def __init__(self, node_features=8, hidden_dim=64, num_classes=2):
        super().__init__()
        
        self.conv1 = SAGEConv(node_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )
        
        self.graph_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        
        node_out = self.node_classifier(x)
        
        graph_emb = global_mean_pool(x, batch)
        graph_out = self.graph_classifier(graph_emb)
        
        return node_out, graph_out, x

def load_graph_dataset(path="data/graph_dataset.json"):
    with open(path, 'r') as f:
        raw_data = json.load(f)
    
    graphs = []
    for item in raw_data:
        x = torch.tensor(item['node_features'], dtype=torch.float)
        
        if len(item['edge_index']) > 0:
            edge_index = torch.tensor(item['edge_index'], dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        y_node = torch.tensor(item['node_labels'], dtype=torch.long)
        y_graph = torch.tensor(item['graph_label'], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y_node=y_node, y_graph=y_graph)
        graphs.append(data)
    
    return graphs

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    total_node_correct = 0
    total_graph_correct = 0
    total_nodes = 0
    total_graphs = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        node_out, graph_out, _ = model(batch)
        
        node_loss = F.cross_entropy(node_out, batch.y_node)
        graph_loss = F.cross_entropy(graph_out, batch.y_graph)
        
        loss = 0.5 * node_loss + 0.5 * graph_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        node_pred = node_out.argmax(dim=1)
        total_node_correct += (node_pred == batch.y_node).sum().item()
        total_nodes += batch.y_node.size(0)
        
        graph_pred = graph_out.argmax(dim=1)
        total_graph_correct += (graph_pred == batch.y_graph).sum().item()
        total_graphs += batch.y_graph.size(0)
    
    return (
        total_loss / len(loader),
        total_node_correct / total_nodes,
        total_graph_correct / total_graphs
    )

def evaluate(model, loader, device):
    model.eval()
    total_node_correct = 0
    total_graph_correct = 0
    total_nodes = 0
    total_graphs = 0
    
    all_graph_preds = []
    all_graph_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            node_out, graph_out, _ = model(batch)
            
            node_pred = node_out.argmax(dim=1)
            total_node_correct += (node_pred == batch.y_node).sum().item()
            total_nodes += batch.y_node.size(0)
            
            graph_pred = graph_out.argmax(dim=1)
            total_graph_correct += (graph_pred == batch.y_graph).sum().item()
            total_graphs += batch.y_graph.size(0)
            
            all_graph_preds.extend(graph_pred.cpu().numpy())
            all_graph_labels.extend(batch.y_graph.cpu().numpy())
    
    all_graph_preds = np.array(all_graph_preds)
    all_graph_labels = np.array(all_graph_labels)
    
    tp = ((all_graph_preds == 1) & (all_graph_labels == 1)).sum()
    fp = ((all_graph_preds == 1) & (all_graph_labels == 0)).sum()
    fn = ((all_graph_preds == 0) & (all_graph_labels == 1)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'node_acc': total_node_correct / total_nodes,
        'graph_acc': total_graph_correct / total_graphs,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    print("=" * 60)
    print("T-DGNN Training (Graph Neural Network)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("\nLoading graph dataset...")
    graphs = load_graph_dataset()
    
    random.shuffle(graphs)
    n = len(graphs)
    train_idx = int(n * 0.7)
    val_idx = int(n * 0.85)
    
    train_graphs = graphs[:train_idx]
    val_graphs = graphs[train_idx:val_idx]
    test_graphs = graphs[val_idx:]
    
    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")
    
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=16)
    test_loader = DataLoader(test_graphs, batch_size=16)
    
    model = TDGNN(node_features=8, hidden_dim=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    print("\nTraining...")
    epochs = 50
    best_val_f1 = 0
    
    for epoch in range(epochs):
        train_loss, train_node_acc, train_graph_acc = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        
        scheduler.step(1 - val_metrics['f1'])
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            os.makedirs('models/weights', exist_ok=True)
            torch.save(model.state_dict(), 'models/weights/tdgnn.pt')
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/{epochs} | "
                  f"Loss: {train_loss:.4f} | "
                  f"Node Acc: {train_node_acc:.3f} | "
                  f"Graph Acc: {val_metrics['graph_acc']:.3f} | "
                  f"F1: {val_metrics['f1']:.3f}")
    
    print("\nLoading best model and evaluating on test set...")
    model.load_state_dict(torch.load('models/weights/tdgnn.pt', weights_only=True))
    test_metrics = evaluate(model, test_loader, device)
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Node Accuracy:  {test_metrics['node_acc']:.4f}")
    print(f"  Graph Accuracy: {test_metrics['graph_acc']:.4f}")
    print(f"  Precision:      {test_metrics['precision']:.4f}")
    print(f"  Recall:         {test_metrics['recall']:.4f}")
    print(f"  F1 Score:       {test_metrics['f1']:.4f}")
    print("=" * 60)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_node_acc': float(test_metrics['node_acc']),
        'test_graph_acc': float(test_metrics['graph_acc']),
        'test_precision': float(test_metrics['precision']),
        'test_recall': float(test_metrics['recall']),
        'test_f1': float(test_metrics['f1'])
    }
    
    with open('models/weights/tdgnn_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nModel saved to models/weights/tdgnn.pt")

if __name__ == "__main__":
    main()

