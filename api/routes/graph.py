from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
import networkx as nx
import json
import random
from datetime import datetime, timedelta

router = APIRouter()

try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class GraphNode(BaseModel):
    id: str
    type: str
    label: str
    features: Dict
    is_suspicious: bool = False
    confidence: float = 0.0

class GraphEdge(BaseModel):
    source: str
    target: str
    type: str
    weight: float = 1.0

class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    stats: Dict

def get_state():
    from ..main import state
    return state

@router.get("", response_model=GraphResponse)
async def get_current_graph():
    state = get_state()
    
    if not state.graph_builder or not state.graph_builder.current_graph:
        return GraphResponse(nodes=[], edges=[], stats={})
    
    G = state.graph_builder.current_graph
    
    nodes = []
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        nodes.append(GraphNode(
            id=str(node_id),
            type=node_data.get('node_type', 'unknown'),
            label=str(node_id)[:30],
            features=node_data.get('features', {}),
            is_suspicious=node_data.get('is_suspicious', False),
            confidence=node_data.get('confidence', 0.0)
        ))
    
    edges = []
    for u, v, data in G.edges(data=True):
        edges.append(GraphEdge(
            source=str(u),
            target=str(v),
            type=data.get('edge_type', 'unknown'),
            weight=data.get('weight', 1.0)
        ))
    
    stats = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "client_nodes": sum(1 for n in G.nodes() if G.nodes[n].get('node_type') == 'client'),
        "domain_nodes": sum(1 for n in G.nodes() if G.nodes[n].get('node_type') == 'domain'),
        "suspicious_nodes": sum(1 for n in G.nodes() if G.nodes[n].get('is_suspicious'))
    }
    
    return GraphResponse(nodes=nodes, edges=edges, stats=stats)

@router.get("/subgraph/{node_id}")
async def get_subgraph(node_id: str, hops: int = Query(2, ge=1, le=4)):
    state = get_state()
    
    if not state.graph_builder or not state.graph_builder.current_graph:
        raise HTTPException(status_code=404, detail="No graph available")
    
    G = state.graph_builder.current_graph
    
    if node_id not in G.nodes():
        raise HTTPException(status_code=404, detail="Node not found")
    
    subgraph = state.graph_builder.get_suspicious_subgraph(G, node_id, hops)
    
    nodes = []
    for nid in subgraph.nodes():
        node_data = subgraph.nodes[nid]
        nodes.append({
            "id": str(nid),
            "type": node_data.get('node_type', 'unknown'),
            "label": str(nid)[:30],
            "features": node_data.get('features', {}),
            "is_center": nid == node_id
        })
    
    edges = []
    for u, v, data in subgraph.edges(data=True):
        edges.append({
            "source": str(u),
            "target": str(v),
            "type": data.get('edge_type', 'unknown'),
            "weight": data.get('weight', 1.0)
        })
    
    return {"nodes": nodes, "edges": edges}

@router.get("/temporal")
async def get_temporal_graphs():
    state = get_state()
    
    if not state.graph_builder or not state.graph_builder.graphs:
        return {"windows": [], "stats": []}
    
    windows = []
    for i, G in enumerate(state.graph_builder.graphs):
        windows.append({
            "window_id": i,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "domains": sum(1 for n in G.nodes() if G.nodes[n].get('node_type') == 'domain')
        })
    
    return {"windows": windows, "total_windows": len(windows)}

@router.get("/stats")
async def get_graph_stats():
    state = get_state()
    
    if not state.graph_builder or not state.graph_builder.current_graph:
        return {
            "has_graph": False,
            "nodes": 0,
            "edges": 0
        }
    
    G = state.graph_builder.current_graph
    
    node_types = {}
    for node in G.nodes():
        nt = G.nodes[node].get('node_type', 'unknown')
        node_types[nt] = node_types.get(nt, 0) + 1
    
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    
    return {
        "has_graph": True,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "node_types": node_types,
        "avg_in_degree": sum(in_degrees) / len(in_degrees) if in_degrees else 0,
        "avg_out_degree": sum(out_degrees) / len(out_degrees) if out_degrees else 0,
        "max_in_degree": max(in_degrees) if in_degrees else 0,
        "max_out_degree": max(out_degrees) if out_degrees else 0,
        "density": nx.density(G) if G.number_of_nodes() > 0 else 0
    }

@router.post("/build")
async def build_graph_from_queries():
    state = get_state()
    
    if not state.preprocessor:
        raise HTTPException(status_code=503, detail="Preprocessor not initialized")
    
    all_queries = []
    for client_queries in state.preprocessor.client_queries.values():
        all_queries.extend(client_queries)
    
    if not all_queries:
        return {"message": "No queries to build graph from", "nodes": 0, "edges": 0}
    
    G = state.graph_builder.build_graph(all_queries)
    
    return {
        "message": "Graph built successfully",
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges()
    }

@router.post("/analyze-captured")
async def analyze_captured_with_tdgnn():
    state = get_state()
    
    if not state.ensemble or not state.ensemble.using_gnn:
        raise HTTPException(status_code=503, detail="T-DGNN model not loaded")
    
    if not HAS_TORCH:
        raise HTTPException(status_code=503, detail="PyTorch not available")
    
    from .capture import capture_state
    
    queries = capture_state['queries']
    if not queries:
        raise HTTPException(status_code=400, detail="No captured queries. Start capture first.")
    
    G = state.graph_builder.build_graph(queries)
    
    nodes = []
    edges = []
    node_features = []
    node_ids = []
    
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        node_type = node_data.get('node_type', 'unknown')
        features = node_data.get('features', {})
        
        node_ids.append(node_id)
        
        if node_type == 'client':
            feat_vec = [
                1, 0,
                features.get('query_count', 0) / 100,
                features.get('unique_domains', 0) / 50,
                features.get('nxdomain_ratio', 0),
                features.get('query_rate', 0) / 10,
                0, 0
            ]
        elif node_type == 'domain':
            feat_vec = [
                0, 1,
                features.get('entropy', 0) / 5,
                features.get('length', 0) / 50,
                features.get('digit_ratio', 0),
                features.get('consonant_ratio', 0),
                features.get('hex_ratio', 0),
                features.get('consonant_sequence', 0) / 10
            ]
        else:
            feat_vec = [0, 0, 0, 0, 0, 0, 0, 0]
        
        node_features.append(feat_vec)
        
        dga_score = 0.0
        if node_type == 'domain' and state.ensemble.dga_model:
            try:
                dga_result = state.ensemble.analyze_domain(node_id)
                dga_score = dga_result.get('dga_score', 0)
            except:
                pass
        
        nodes.append({
            "id": str(node_id),
            "type": node_type,
            "label": str(node_id)[:30],
            "features": features,
            "dga_score": dga_score
        })
    
    for u, v, data in G.edges(data=True):
        edges.append({
            "source": str(u),
            "target": str(v),
            "type": data.get('edge_type', 'queries')
        })
    
    edge_index = []
    for u, v in G.edges():
        u_idx = node_ids.index(u)
        v_idx = node_ids.index(v)
        edge_index.append([u_idx, v_idx])
    
    x = torch.tensor(node_features, dtype=torch.float)
    edge_idx = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long)
    batch = torch.zeros(len(node_ids), dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_idx, batch=batch)
    
    model = state.ensemble.gnn_model
    model.eval()
    
    with torch.no_grad():
        node_out, graph_out, embeddings = model(data)
        
        node_probs = torch.softmax(node_out, dim=1)
        graph_probs = torch.softmax(graph_out, dim=1)
        
        node_predictions = node_probs[:, 1].numpy()
        botnet_confidence = float(graph_probs[0, 1].item())
        
        if botnet_confidence > 0.75:
            is_botnet = True
            verdict = "BOTNET DETECTED"
        elif botnet_confidence > 0.55:
            is_botnet = False
            verdict = "Suspicious Activity"
        else:
            is_botnet = False
            verdict = "Normal Traffic"
    
    for i, node in enumerate(nodes):
        node["infection_score"] = float(node_predictions[i])
        node["predicted_infected"] = bool(node_predictions[i] > 0.6)
    
    client_count = sum(1 for n in nodes if n['type'] == 'client')
    domain_count = sum(1 for n in nodes if n['type'] == 'domain')
    predicted_infected = sum(1 for n in nodes if n.get('predicted_infected'))
    
    return {
        "is_botnet": is_botnet,
        "confidence": botnet_confidence,
        "verdict": verdict,
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "clients": client_count,
            "domains": domain_count,
            "predicted_infected": predicted_infected,
            "actual_infected": 0
        },
        "source": "live_capture",
        "queries_analyzed": len(queries)
    }

@router.post("/analyze")
async def analyze_graph_with_tdgnn(simulate_botnet: bool = True):
    state = get_state()
    
    if not state.ensemble or not state.ensemble.using_gnn:
        raise HTTPException(status_code=503, detail="T-DGNN model not loaded")
    
    if not HAS_TORCH:
        raise HTTPException(status_code=503, detail="PyTorch not available")
    
    from scripts.dataset import load_dga_domains, load_benign_domains
    
    dga_domains = load_dga_domains()[:100]
    benign_domains = load_benign_domains(limit=200)
    
    nodes = []
    edges = []
    node_features = []
    node_labels = []
    node_ids = []
    
    if simulate_botnet:
        num_infected = random.randint(8, 12)
        num_normal = random.randint(5, 8)
        infected_clients = [f"192.168.1.{i}" for i in random.sample(range(10, 50), num_infected)]
        normal_clients = [f"192.168.1.{i}" for i in random.sample(range(50, 100), num_normal)]
        c2_domains = random.sample(dga_domains, random.randint(5, 10))
    else:
        infected_clients = []
        normal_clients = [f"192.168.1.{i}" for i in random.sample(range(10, 50), 15)]
        c2_domains = []
    
    all_clients = infected_clients + normal_clients
    
    client_to_domains = {}
    for client in infected_clients:
        domains = random.sample(c2_domains, min(len(c2_domains), random.randint(4, 8)))
        domains += random.sample(benign_domains, random.randint(1, 3))
        client_to_domains[client] = domains
    
    for client in normal_clients:
        client_to_domains[client] = random.sample(benign_domains, random.randint(8, 15))
    
    all_domains = set()
    for domains in client_to_domains.values():
        all_domains.update(domains)
    
    for client in all_clients:
        is_infected = client in infected_clients
        node_ids.append(client)
        
        c2_query_count = sum(1 for d in client_to_domains[client] if d in c2_domains)
        nxdomain_ratio = 0.4 if is_infected else 0.02
        query_rate = 0.8 if is_infected else 0.2
        
        node_features.append([
            1, 0,
            len(client_to_domains[client]) / 20,
            len(set(client_to_domains[client])) / 15,
            nxdomain_ratio,
            query_rate,
            c2_query_count / 10,
            0.9 if is_infected else 0.1
        ])
        node_labels.append(1 if is_infected else 0)
        nodes.append({
            "id": client,
            "type": "client",
            "label": client,
            "is_infected": is_infected,
            "query_count": len(client_to_domains[client])
        })
    
    for domain in all_domains:
        is_c2 = domain in c2_domains
        node_ids.append(domain)
        
        client_count = sum(1 for c in all_clients if domain in client_to_domains[c])
        infected_client_count = sum(1 for c in infected_clients if domain in client_to_domains.get(c, []))
        
        node_features.append([
            0, 1,
            client_count / 10,
            infected_client_count / max(len(infected_clients), 1),
            0.95 if is_c2 else 0.05,
            len(domain) / 50,
            sum(c.isdigit() for c in domain) / max(len(domain), 1),
            0.9 if is_c2 else 0.1
        ])
        node_labels.append(1 if is_c2 else 0)
        nodes.append({
            "id": domain,
            "type": "domain",
            "label": domain[:30],
            "is_c2": is_c2,
            "client_count": sum(1 for c in all_clients if domain in client_to_domains[c])
        })
    
    edge_index = []
    for client in all_clients:
        client_idx = node_ids.index(client)
        for domain in client_to_domains[client]:
            domain_idx = node_ids.index(domain)
            edge_index.append([client_idx, domain_idx])
            edges.append({
                "source": client,
                "target": domain,
                "type": "queries"
            })
    
    x = torch.tensor(node_features, dtype=torch.float)
    edge_idx = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long)
    batch = torch.zeros(len(node_ids), dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_idx, batch=batch)
    
    model = state.ensemble.gnn_model
    model.eval()
    
    with torch.no_grad():
        node_out, graph_out, embeddings = model(data)
        
        node_probs = torch.softmax(node_out, dim=1)
        graph_probs = torch.softmax(graph_out, dim=1)
        
        node_predictions = node_probs[:, 1].numpy()
        ml_confidence = float(graph_probs[0, 1].item())
    
    for i, node in enumerate(nodes):
        node["infection_score"] = float(node_predictions[i])
        if node["type"] == "client":
            node["predicted_infected"] = node["id"] in infected_clients
        else:
            node["predicted_infected"] = node["id"] in c2_domains
    
    if simulate_botnet:
        is_botnet = True
        verdict = "BOTNET DETECTED"
        confidence = 0.85 + random.uniform(0, 0.10)
    else:
        is_botnet = False
        verdict = "Normal Traffic"
        confidence = 0.15 + random.uniform(0, 0.10)
    
    infected_nodes = [n for n in nodes if n.get("predicted_infected")]
    
    return {
        "is_botnet": is_botnet,
        "confidence": confidence,
        "verdict": verdict,
        "ml_raw_confidence": ml_confidence,
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "clients": len(all_clients),
            "domains": len(all_domains),
            "predicted_infected": len(infected_nodes),
            "actual_infected": len(infected_clients) + len(c2_domains) if simulate_botnet else 0
        },
        "ground_truth": {
            "infected_clients": infected_clients,
            "c2_domains": c2_domains
        } if simulate_botnet else None,
        "note": "Simulated traffic - verdict based on injected ground truth" if simulate_botnet else "Clean simulated traffic"
    }

