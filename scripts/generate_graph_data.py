#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import json
from datetime import datetime, timedelta
from collections import defaultdict
import networkx as nx

from scripts.dataset import load_dga_domains, load_benign_domains

def generate_botnet_scenario():
    dga_domains = load_dga_domains()[:5000]
    benign_domains = load_benign_domains(limit=10000)
    
    scenarios = []
    
    for scenario_id in range(100):
        num_infected = random.randint(3, 10)
        num_normal = random.randint(10, 30)
        
        infected_clients = [f"192.168.1.{i}" for i in random.sample(range(10, 100), num_infected)]
        normal_clients = [f"192.168.1.{i}" for i in random.sample(range(100, 200), num_normal)]
        
        c2_domains = random.sample(dga_domains, random.randint(5, 20))
        
        queries = []
        base_time = datetime.now() - timedelta(hours=random.randint(1, 24))
        
        for client in infected_clients:
            for c2_domain in c2_domains:
                if random.random() < 0.7:
                    for _ in range(random.randint(1, 5)):
                        queries.append({
                            'timestamp': (base_time + timedelta(seconds=random.randint(0, 3600))).isoformat(),
                            'src_ip': client,
                            'domain': c2_domain,
                            'query_type': 'A',
                            'label': 'botnet'
                        })
            
            for _ in range(random.randint(5, 20)):
                queries.append({
                    'timestamp': (base_time + timedelta(seconds=random.randint(0, 3600))).isoformat(),
                    'src_ip': client,
                    'domain': random.choice(benign_domains),
                    'query_type': 'A',
                    'label': 'normal'
                })
        
        for client in normal_clients:
            for _ in range(random.randint(10, 50)):
                queries.append({
                    'timestamp': (base_time + timedelta(seconds=random.randint(0, 3600))).isoformat(),
                    'src_ip': client,
                    'domain': random.choice(benign_domains),
                    'query_type': 'A',
                    'label': 'normal'
                })
        
        scenarios.append({
            'scenario_id': scenario_id,
            'infected_clients': infected_clients,
            'normal_clients': normal_clients,
            'c2_domains': c2_domains,
            'queries': queries,
            'label': 1
        })
    
    for scenario_id in range(100, 200):
        num_clients = random.randint(15, 40)
        clients = [f"192.168.1.{i}" for i in random.sample(range(10, 200), num_clients)]
        
        queries = []
        base_time = datetime.now() - timedelta(hours=random.randint(1, 24))
        
        for client in clients:
            for _ in range(random.randint(10, 50)):
                queries.append({
                    'timestamp': (base_time + timedelta(seconds=random.randint(0, 3600))).isoformat(),
                    'src_ip': client,
                    'domain': random.choice(benign_domains),
                    'query_type': 'A',
                    'label': 'normal'
                })
        
        scenarios.append({
            'scenario_id': scenario_id,
            'infected_clients': [],
            'normal_clients': clients,
            'c2_domains': [],
            'queries': queries,
            'label': 0
        })
    
    random.shuffle(scenarios)
    return scenarios

def build_graph_from_queries(queries):
    G = nx.DiGraph()
    
    client_stats = defaultdict(lambda: {'query_count': 0, 'unique_domains': set(), 'dga_count': 0})
    domain_stats = defaultdict(lambda: {'query_count': 0, 'unique_clients': set()})
    
    for q in queries:
        client = q['src_ip']
        domain = q['domain']
        
        client_stats[client]['query_count'] += 1
        client_stats[client]['unique_domains'].add(domain)
        
        domain_stats[domain]['query_count'] += 1
        domain_stats[domain]['unique_clients'].add(client)
        
        if G.has_edge(client, domain):
            G[client][domain]['weight'] += 1
        else:
            G.add_edge(client, domain, weight=1)
    
    for client, stats in client_stats.items():
        G.nodes[client]['type'] = 'client'
        G.nodes[client]['query_count'] = stats['query_count']
        G.nodes[client]['unique_domains'] = len(stats['unique_domains'])
    
    for domain, stats in domain_stats.items():
        G.nodes[domain]['type'] = 'domain'
        G.nodes[domain]['query_count'] = stats['query_count']
        G.nodes[domain]['unique_clients'] = len(stats['unique_clients'])
    
    return G

def graph_to_features(G, infected_clients, c2_domains):
    node_features = []
    node_labels = []
    node_ids = []
    
    for node in G.nodes():
        node_data = G.nodes[node]
        
        if node_data.get('type') == 'client':
            features = [
                1, 0,
                node_data.get('query_count', 0) / 100,
                node_data.get('unique_domains', 0) / 50,
                G.out_degree(node) / 20,
                0, 0, 0
            ]
            label = 1 if node in infected_clients else 0
        else:
            features = [
                0, 1,
                node_data.get('query_count', 0) / 100,
                node_data.get('unique_clients', 0) / 20,
                G.in_degree(node) / 10,
                len(node) / 50,
                sum(c.isdigit() for c in node) / max(len(node), 1),
                0
            ]
            label = 1 if node in c2_domains else 0
        
        node_features.append(features)
        node_labels.append(label)
        node_ids.append(node)
    
    edges = list(G.edges())
    edge_index = [[node_ids.index(e[0]), node_ids.index(e[1])] for e in edges]
    edge_weights = [G[e[0]][e[1]]['weight'] for e in edges]
    
    return {
        'node_features': node_features,
        'node_labels': node_labels,
        'node_ids': node_ids,
        'edge_index': edge_index,
        'edge_weights': edge_weights
    }

def main():
    print("Generating botnet scenarios...")
    scenarios = generate_botnet_scenario()
    
    print(f"Generated {len(scenarios)} scenarios")
    print(f"  Botnet: {sum(1 for s in scenarios if s['label'] == 1)}")
    print(f"  Normal: {sum(1 for s in scenarios if s['label'] == 0)}")
    
    dataset = []
    
    for i, scenario in enumerate(scenarios):
        G = build_graph_from_queries(scenario['queries'])
        
        graph_data = graph_to_features(
            G, 
            set(scenario['infected_clients']),
            set(scenario['c2_domains'])
        )
        
        graph_data['graph_label'] = scenario['label']
        graph_data['scenario_id'] = scenario['scenario_id']
        
        dataset.append(graph_data)
        
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(scenarios)} scenarios")
    
    output_path = "data/graph_dataset.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f)
    
    print(f"\nSaved to {output_path}")
    print(f"Total graphs: {len(dataset)}")
    
    sample = dataset[0]
    print(f"\nSample graph:")
    print(f"  Nodes: {len(sample['node_features'])}")
    print(f"  Edges: {len(sample['edge_index'])}")
    print(f"  Infected nodes: {sum(sample['node_labels'])}")
    print(f"  Graph label: {sample['graph_label']}")

if __name__ == "__main__":
    main()

