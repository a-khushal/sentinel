from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import networkx as nx
import numpy as np
from dataclasses import dataclass
from capture.sniffer import DNSQuery
from .lexical import LexicalFeatures

@dataclass
class GraphNode:
    node_id: str
    node_type: str
    features: Dict[str, float]

@dataclass  
class GraphEdge:
    source: str
    target: str
    edge_type: str
    timestamp: datetime
    weight: float = 1.0

class GraphBuilder:
    NODE_TYPE_CLIENT = 'client'
    NODE_TYPE_DOMAIN = 'domain'
    NODE_TYPE_IP = 'ip'
    NODE_TYPE_NS = 'nameserver'
    
    EDGE_TYPE_QUERIES = 'queries'
    EDGE_TYPE_RESOLVES = 'resolves_to'
    
    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self.lexical = LexicalFeatures()
        self.graphs: List[nx.DiGraph] = []
        self.current_graph = nx.DiGraph()
        self.node_features: Dict[str, Dict] = {}
        
    def build_graph(self, queries: List[DNSQuery]) -> nx.DiGraph:
        G = nx.DiGraph()
        
        client_queries = defaultdict(list)
        domain_queries = defaultdict(list)
        
        for q in queries:
            client_queries[q.src_ip].append(q)
            domain_queries[q.query_name].append(q)
            
            if not G.has_node(q.src_ip):
                G.add_node(q.src_ip, 
                          node_type=self.NODE_TYPE_CLIENT,
                          features={})
            
            if not G.has_node(q.query_name):
                domain_features = self.lexical.extract(q.query_name)
                G.add_node(q.query_name,
                          node_type=self.NODE_TYPE_DOMAIN,
                          features=domain_features)
            
            if G.has_edge(q.src_ip, q.query_name):
                G[q.src_ip][q.query_name]['weight'] += 1
                G[q.src_ip][q.query_name]['timestamps'].append(q.timestamp)
            else:
                G.add_edge(q.src_ip, q.query_name,
                          edge_type=self.EDGE_TYPE_QUERIES,
                          weight=1,
                          timestamps=[q.timestamp])
            
            if q.answers:
                for answer in q.answers:
                    if not G.has_node(answer):
                        G.add_node(answer,
                                  node_type=self.NODE_TYPE_IP,
                                  features={})
                    
                    if not G.has_edge(q.query_name, answer):
                        G.add_edge(q.query_name, answer,
                                  edge_type=self.EDGE_TYPE_RESOLVES,
                                  weight=1,
                                  ttl=q.ttl)
        
        for client, client_qs in client_queries.items():
            features = self._compute_client_features(client_qs)
            G.nodes[client]['features'] = features
        
        self.current_graph = G
        return G
    
    def build_temporal_graphs(self, queries: List[DNSQuery], 
                               num_windows: int = 5) -> List[nx.DiGraph]:
        if not queries:
            return []
        
        queries = sorted(queries, key=lambda q: q.timestamp)
        
        start_time = queries[0].timestamp
        end_time = queries[-1].timestamp
        total_duration = (end_time - start_time).total_seconds()
        
        if total_duration == 0:
            return [self.build_graph(queries)]
        
        window_duration = total_duration / num_windows
        
        graphs = []
        for i in range(num_windows):
            window_start = start_time + timedelta(seconds=i * window_duration)
            window_end = start_time + timedelta(seconds=(i + 1) * window_duration)
            
            window_queries = [
                q for q in queries 
                if window_start <= q.timestamp < window_end
            ]
            
            if window_queries:
                G = self.build_graph(window_queries)
                graphs.append(G)
        
        self.graphs = graphs
        return graphs
    
    def _compute_client_features(self, queries: List[DNSQuery]) -> Dict[str, float]:
        domains = [q.query_name for q in queries]
        unique_domains = set(domains)
        
        nxdomain_count = sum(1 for q in queries if q.response_code == 3)
        
        timestamps = sorted([q.timestamp for q in queries])
        query_rate = 0.0
        if len(timestamps) >= 2:
            duration = (timestamps[-1] - timestamps[0]).total_seconds()
            if duration > 0:
                query_rate = len(queries) / duration
        
        return {
            'query_count': len(queries),
            'unique_domains': len(unique_domains),
            'nxdomain_ratio': nxdomain_count / len(queries) if queries else 0,
            'query_rate': query_rate,
        }
    
    def to_pyg_data(self, G: nx.DiGraph):
        import torch
        from torch_geometric.data import Data
        
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        
        node_types = []
        node_features_list = []
        
        for node in G.nodes():
            node_data = G.nodes[node]
            node_type = node_data.get('node_type', 'unknown')
            features = node_data.get('features', {})
            
            type_encoding = {
                self.NODE_TYPE_CLIENT: [1, 0, 0, 0],
                self.NODE_TYPE_DOMAIN: [0, 1, 0, 0],
                self.NODE_TYPE_IP: [0, 0, 1, 0],
                self.NODE_TYPE_NS: [0, 0, 0, 1],
            }
            type_vec = type_encoding.get(node_type, [0, 0, 0, 0])
            
            if node_type == self.NODE_TYPE_DOMAIN:
                feat_vec = [
                    features.get('entropy', 0),
                    features.get('length', 0) / 100,
                    features.get('digit_ratio', 0),
                    features.get('vowel_ratio', 0),
                    features.get('consonant_ratio', 0),
                    features.get('bigram_entropy', 0),
                    features.get('hex_ratio', 0),
                    features.get('consonant_sequence', 0) / 10,
                ]
            elif node_type == self.NODE_TYPE_CLIENT:
                feat_vec = [
                    features.get('query_count', 0) / 1000,
                    features.get('unique_domains', 0) / 100,
                    features.get('nxdomain_ratio', 0),
                    features.get('query_rate', 0) / 10,
                    0, 0, 0, 0
                ]
            else:
                feat_vec = [0] * 8
            
            node_features_list.append(type_vec + feat_vec)
        
        edge_index = []
        edge_attr = []
        
        for u, v, data in G.edges(data=True):
            edge_index.append([node_mapping[u], node_mapping[v]])
            weight = data.get('weight', 1)
            edge_attr.append([weight])
        
        if not edge_index:
            edge_index = [[0, 0]]
            edge_attr = [[0]]
        
        x = torch.tensor(node_features_list, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def get_suspicious_subgraph(self, G: nx.DiGraph, 
                                 center_node: str, 
                                 hops: int = 2) -> nx.DiGraph:
        nodes = {center_node}
        frontier = {center_node}
        
        for _ in range(hops):
            new_frontier = set()
            for node in frontier:
                new_frontier.update(G.predecessors(node))
                new_frontier.update(G.successors(node))
            nodes.update(new_frontier)
            frontier = new_frontier
        
        return G.subgraph(nodes).copy()

