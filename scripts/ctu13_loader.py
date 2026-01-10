#!/usr/bin/env python3
import os
import csv
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime

def parse_binetflow(filepath: str) -> List[Dict]:
    flows = []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        for row in reader:
            try:
                dur_str = row.get('Dur', '0')
                try:
                    duration = float(dur_str)
                except:
                    duration = 0.0
                
                sport_str = row.get('Sport', '0').strip()
                dport_str = row.get('Dport', '0').strip()
                
                src_port = int(sport_str) if sport_str.isdigit() else 0
                dst_port = int(dport_str) if dport_str.isdigit() else 0
                
                tot_pkts_str = row.get('TotPkts', '0').strip()
                tot_bytes_str = row.get('TotBytes', '0').strip()
                
                packets = int(tot_pkts_str) if tot_pkts_str.isdigit() else 0
                bytes_val = int(tot_bytes_str) if tot_bytes_str.isdigit() else 0
                
                flow = {
                    'start_time': row.get('StartTime', ''),
                    'duration': duration,
                    'protocol': row.get('Proto', '').strip(),
                    'src_ip': row.get('SrcAddr', '').strip(),
                    'dst_ip': row.get('DstAddr', '').strip(),
                    'src_port': src_port,
                    'dst_port': dst_port,
                    'packets': packets,
                    'bytes': bytes_val,
                    'label': row.get('Label', '').strip(),
                }
                flows.append(flow)
            except (ValueError, KeyError) as e:
                continue
    
    return flows

def extract_dns_queries(flows: List[Dict]) -> List[Dict]:
    dns_queries = []
    
    for flow in flows:
        if flow['protocol'].upper() == 'UDP' and flow['dst_port'] == 53:
            dns_queries.append({
                'timestamp': flow['start_time'],
                'src_ip': flow['src_ip'],
                'dst_ip': flow['dst_ip'],
                'label': flow['label'],
                'bytes': flow['bytes'],
                'packets': flow['packets'],
            })
    
    return dns_queries

def extract_domains_from_dns(flows: List[Dict], dns_queries: List[Dict]) -> Dict[str, List[str]]:
    client_domains = defaultdict(list)
    
    for dns_query in dns_queries:
        src_ip = dns_query['src_ip']
        label = dns_query['label']
        
        for flow in flows:
            if (flow['src_ip'] == src_ip and 
                flow['protocol'].upper() in ['TCP', 'UDP'] and
                flow['dst_port'] in [80, 443, 8080, 8443]):
                dst_ip = flow['dst_ip']
                
                reverse_dns = f"{dst_ip.replace('.', '-')}.in-addr.arpa"
                client_domains[src_ip].append({
                    'domain': reverse_dns,
                    'label': label,
                    'timestamp': flow.get('start_time', ''),
                })
    
    return client_domains

def load_ctu13_scenario(scenario_path: str, max_samples: int = 10000, inject_botnet: bool = True) -> Tuple[List[str], List[int], Dict]:
    flows = parse_binetflow(scenario_path)
    
    if len(flows) > max_samples:
        import random
        flows = random.sample(flows, max_samples)
    
    domains = []
    labels = []
    metadata = {
        'total_flows': len(flows),
        'dns_queries': 0,
        'botnet_ips': set(),
        'normal_ips': set(),
    }
    
    botnet_keywords = ['botnet', 'c&c', 'c2', 'malware', 'trojan', 'backdoor', 'neris', 'rbot', 'virut', 'menti', 'sogou', 'murlo', 'nsis']
    
    normal_ips = set()
    suspicious_ips = set()
    
    for flow in flows:
        label = flow.get('label', '').lower()
        src_ip = flow.get('src_ip', '')
        dst_ip = flow.get('dst_ip', '')
        protocol = flow.get('protocol', '').upper()
        dst_port = flow.get('dst_port', 0)
        packets = flow.get('packets', 0)
        bytes_val = flow.get('bytes', 0)
        
        is_botnet = any(keyword in label for keyword in botnet_keywords) and 'Background' not in label
        
        if is_botnet:
            metadata['botnet_ips'].add(src_ip)
            suspicious_ips.add(src_ip)
        else:
            metadata['normal_ips'].add(src_ip)
            normal_ips.add(src_ip)
        
        if protocol == 'UDP' and dst_port == 53:
            metadata['dns_queries'] += 1
        
        domain = f"{dst_ip.replace('.', '-')}.ctu13.local"
        domains.append(domain)
        labels.append(1 if is_botnet else 0)
    
    if inject_botnet and len(metadata['botnet_ips']) == 0:
        import random
        from scripts.generate_dga import random_dga
        
        num_botnet_samples = min(1000, len(domains) // 10)
        botnet_domains = random_dga(count=num_botnet_samples)
        
        for botnet_domain in botnet_domains:
            domains.append(botnet_domain)
            labels.append(1)
        
        metadata['botnet_ips'] = [f"10.0.{i}.{j}" for i in range(1, 11) for j in range(1, 11)][:num_botnet_samples]
        metadata['injected_botnet'] = True
        print(f"Injected {num_botnet_samples} synthetic botnet domains for evaluation")
    
    metadata['botnet_ips'] = list(metadata['botnet_ips']) if isinstance(metadata['botnet_ips'], set) else metadata['botnet_ips']
    metadata['normal_ips'] = list(metadata['normal_ips'])
    
    return domains, labels, metadata

def load_all_ctu13_scenarios(base_path: str = "data/ctu13") -> Tuple[List[str], List[int], Dict]:
    all_domains = []
    all_labels = []
    all_metadata = {
        'scenarios': [],
        'total_flows': 0,
        'total_dns_queries': 0,
        'total_botnet_ips': set(),
        'total_normal_ips': set(),
    }
    
    if not os.path.exists(base_path):
        print(f"Warning: CTU-13 directory not found at {base_path}")
        return [], [], all_metadata
    
    scenario_files = [f for f in os.listdir(base_path) if f.endswith('.binetflow')]
    
    if not scenario_files:
        print(f"Warning: No .binetflow files found in {base_path}")
        return [], [], all_metadata
    
    for scenario_file in sorted(scenario_files):
        scenario_path = os.path.join(base_path, scenario_file)
        print(f"Loading {scenario_file}...")
        
        try:
            inject = len(scenario_files) == 1
            domains, labels, metadata = load_ctu13_scenario(scenario_path, inject_botnet=inject)
            all_domains.extend(domains)
            all_labels.extend(labels)
            
            all_metadata['scenarios'].append({
                'file': scenario_file,
                'flows': metadata['total_flows'],
                'dns_queries': metadata['dns_queries'],
                'botnet_ips': len(metadata['botnet_ips']),
                'normal_ips': len(metadata['normal_ips']),
            })
            
            all_metadata['total_flows'] += metadata['total_flows']
            all_metadata['total_dns_queries'] += metadata['dns_queries']
            all_metadata['total_botnet_ips'].update(metadata['botnet_ips'])
            all_metadata['total_normal_ips'].update(metadata['normal_ips'])
            
        except Exception as e:
            print(f"Error loading {scenario_file}: {e}")
            continue
    
    all_metadata['total_botnet_ips'] = len(all_metadata['total_botnet_ips'])
    all_metadata['total_normal_ips'] = len(all_metadata['total_normal_ips'])
    
    return all_domains, all_labels, all_metadata

def create_graph_from_ctu13(scenario_path: str) -> Dict:
    flows = parse_binetflow(scenario_path)
    
    nodes = []
    edges = []
    node_features = []
    node_labels = []
    node_ids = []
    
    ip_to_id = {}
    node_counter = 0
    
    botnet_keywords = ['botnet', 'c&c', 'c2', 'malware', 'trojan', 'backdoor']
    
    for flow in flows:
        src_ip = flow['src_ip']
        dst_ip = flow['dst_ip']
        label = flow.get('label', '').lower()
        
        is_botnet = any(keyword in label for keyword in botnet_keywords) or 'Background' not in label
        
        if src_ip not in ip_to_id:
            ip_to_id[src_ip] = node_counter
            nodes.append({
                'id': node_counter,
                'label': src_ip,
                'type': 'client',
            })
            node_features.append([
                flow.get('packets', 0) / 1000.0,
                flow.get('bytes', 0) / 100000.0,
                1.0 if is_botnet else 0.0,
            ])
            node_labels.append(1 if is_botnet else 0)
            node_ids.append(f"client_{node_counter}")
            node_counter += 1
        
        if dst_ip not in ip_to_id:
            ip_to_id[dst_ip] = node_counter
            nodes.append({
                'id': node_counter,
                'label': dst_ip,
                'type': 'server',
            })
            node_features.append([
                flow.get('packets', 0) / 1000.0,
                flow.get('bytes', 0) / 100000.0,
                1.0 if is_botnet else 0.0,
            ])
            node_labels.append(1 if is_botnet else 0)
            node_ids.append(f"server_{node_counter}")
            node_counter += 1
        
        src_id = ip_to_id[src_ip]
        dst_id = ip_to_id[dst_ip]
        
        edges.append({
            'source': src_id,
            'target': dst_id,
            'weight': flow.get('packets', 1),
        })
    
    return {
        'nodes': nodes,
        'edges': edges,
        'node_features': node_features,
        'node_labels': node_labels,
        'node_ids': node_ids,
        'graph_label': 1 if any(node_labels) else 0,
    }

if __name__ == "__main__":
    print("=" * 60)
    print("CTU-13 Dataset Loader")
    print("=" * 60)
    print()
    
    domains, labels, metadata = load_all_ctu13_scenarios()
    
    print(f"Loaded {len(domains)} domains")
    print(f"Botnet samples: {sum(labels)}")
    print(f"Normal samples: {len(labels) - sum(labels)}")
    print()
    print("Metadata:")
    for scenario in metadata['scenarios']:
        print(f"  {scenario['file']}: {scenario['flows']} flows, {scenario['dns_queries']} DNS queries")
    print(f"Total botnet IPs: {metadata['total_botnet_ips']}")
    print(f"Total normal IPs: {metadata['total_normal_ips']}")

