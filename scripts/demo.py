#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import string
from datetime import datetime, timedelta

from features.lexical import LexicalFeatures
from features.graph_builder import GraphBuilder
from models.ensemble import ThreatEnsemble
from blockchain.client import MockBlockchainClient, ThreatType
from capture.sniffer import DNSQuery

def generate_benign_domain():
    words = ['google', 'facebook', 'amazon', 'microsoft', 'github', 'stackoverflow', 
             'reddit', 'twitter', 'youtube', 'linkedin', 'netflix', 'spotify']
    tlds = ['com', 'org', 'net', 'io', 'co']
    return f"{random.choice(words)}.{random.choice(tlds)}"

def generate_dga_domain():
    length = random.randint(10, 25)
    chars = string.ascii_lowercase + string.digits
    domain = ''.join(random.choice(chars) for _ in range(length))
    tld = random.choice(['com', 'net', 'org', 'xyz', 'top'])
    return f"{domain}.{tld}"

def generate_mock_queries(num_queries: int = 100):
    queries = []
    clients = [f"192.168.1.{i}" for i in range(10, 20)]
    
    base_time = datetime.now() - timedelta(hours=1)
    
    for i in range(num_queries):
        is_malicious = random.random() < 0.2
        domain = generate_dga_domain() if is_malicious else generate_benign_domain()
        
        query = DNSQuery(
            timestamp=base_time + timedelta(seconds=i * 5),
            src_ip=random.choice(clients),
            dst_ip="8.8.8.8",
            query_name=domain,
            query_type=1,
            response_code=0 if random.random() > 0.1 else 3,
            answers=["1.2.3.4"] if random.random() > 0.3 else None,
            ttl=random.randint(60, 3600) if random.random() > 0.3 else None
        )
        queries.append(query)
    
    return queries

def main():
    print("=" * 60)
    print("SENTINEL - Botnet Detection System Demo")
    print("=" * 60)
    print()
    
    print("[1/5] Initializing components...")
    lexical = LexicalFeatures()
    graph_builder = GraphBuilder()
    ensemble = ThreatEnsemble()
    blockchain = MockBlockchainClient()
    blockchain.register_node()
    
    modes = []
    if ensemble.using_ml:
        modes.append("DGA-ML")
    if ensemble.using_gnn:
        modes.append("T-DGNN")
    mode = " + ".join(modes) if modes else "heuristic only"
    print(f"      Components initialized ({mode})")
    print()
    
    print("[2/5] Generating mock DNS traffic...")
    queries = generate_mock_queries(100)
    print(f"      Generated {len(queries)} DNS queries")
    print()
    
    print("[3/5] Building DNS traffic graph...")
    graph = graph_builder.build_graph(queries)
    print(f"      Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print()
    
    print("[4/5] Analyzing domains for threats...")
    detections = ensemble.analyze_traffic(queries)
    print(f"      Detected {len(detections)} threats")
    
    if detections:
        print()
        print("      Top threats:")
        for i, d in enumerate(detections[:5]):
            print(f"        {i+1}. {d.domain[:40]:<40} conf={d.confidence:.2f} type={d.threat_type}")
    print()
    
    print("[5/5] Reporting threats to blockchain...")
    reported = 0
    for detection in detections[:10]:
        tx_hash = blockchain.report_threat(
            domain=detection.domain,
            threat_type=ThreatType.DGA if detection.threat_type == 'dga' else ThreatType.UNKNOWN,
            confidence=int(detection.confidence * 100)
        )
        reported += 1
    print(f"      Reported {reported} threats to blockchain")
    print(f"      Total reports on chain: {blockchain.get_total_reports()}")
    print()
    
    print("=" * 60)
    print("Demo complete!")
    print()
    print("Summary:")
    print(f"  - Queries processed: {len(queries)}")
    print(f"  - Graph nodes: {graph.number_of_nodes()}")
    print(f"  - Threats detected: {len(detections)}")
    print(f"  - Blockchain reports: {blockchain.get_total_reports()}")
    print("=" * 60)
    
    print()
    print("Example domain analysis:")
    test_domains = [
        "google.com",
        "x7k9m2p4q8.evil.com",
        "asjhd7823hdkjashd.xyz",
        "github.io"
    ]
    
    for domain in test_domains:
        result = ensemble.analyze_domain(domain)
        status = "SUSPICIOUS" if result['is_suspicious'] else "CLEAN"
        ml_info = f"ml={result['dga_score']:.2f}" if result.get('using_ml') else "heuristic"
        print(f"  {domain:<35} [{status}] conf={result['confidence']:.2f} ({ml_info})")

if __name__ == "__main__":
    main()
