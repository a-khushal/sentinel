from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Optional
import networkx as nx
import json

router = APIRouter()

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

