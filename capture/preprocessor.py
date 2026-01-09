from typing import List, Dict
from collections import defaultdict
from datetime import datetime, timedelta
import pandas as pd
from .sniffer import DNSQuery

class DNSPreprocessor:
    def __init__(self):
        self.client_queries: Dict[str, List[DNSQuery]] = defaultdict(list)
        self.domain_queries: Dict[str, List[DNSQuery]] = defaultdict(list)
    
    def process_queries(self, queries: List[DNSQuery]):
        for query in queries:
            self.client_queries[query.src_ip].append(query)
            self.domain_queries[query.query_name].append(query)
    
    def to_dataframe(self, queries: List[DNSQuery]) -> pd.DataFrame:
        data = []
        for q in queries:
            data.append({
                'timestamp': q.timestamp,
                'src_ip': q.src_ip,
                'dst_ip': q.dst_ip,
                'domain': q.query_name,
                'query_type': q.query_type,
                'response_code': q.response_code,
                'ttl': q.ttl,
                'num_answers': len(q.answers) if q.answers else 0
            })
        return pd.DataFrame(data)
    
    def get_client_stats(self, client_ip: str) -> Dict:
        queries = self.client_queries.get(client_ip, [])
        if not queries:
            return {}
        
        domains = [q.query_name for q in queries]
        unique_domains = set(domains)
        nxdomain_count = sum(1 for q in queries if q.response_code == 3)
        
        return {
            'total_queries': len(queries),
            'unique_domains': len(unique_domains),
            'nxdomain_ratio': nxdomain_count / len(queries) if queries else 0,
            'query_rate': self._calculate_query_rate(queries),
            'domains': list(unique_domains)[:100]
        }
    
    def get_domain_stats(self, domain: str) -> Dict:
        queries = self.domain_queries.get(domain, [])
        if not queries:
            return {}
        
        clients = set(q.src_ip for q in queries)
        ttls = [q.ttl for q in queries if q.ttl is not None]
        
        return {
            'total_queries': len(queries),
            'unique_clients': len(clients),
            'avg_ttl': sum(ttls) / len(ttls) if ttls else 0,
            'ttl_variance': self._variance(ttls) if ttls else 0,
            'clients': list(clients)[:100]
        }
    
    def _calculate_query_rate(self, queries: List[DNSQuery]) -> float:
        if len(queries) < 2:
            return 0.0
        
        timestamps = sorted(q.timestamp for q in queries)
        duration = (timestamps[-1] - timestamps[0]).total_seconds()
        
        if duration == 0:
            return float(len(queries))
        
        return len(queries) / duration
    
    def _variance(self, values: List) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def get_time_window(self, queries: List[DNSQuery], 
                        start: datetime, end: datetime) -> List[DNSQuery]:
        return [q for q in queries if start <= q.timestamp <= end]
    
    def clear(self):
        self.client_queries.clear()
        self.domain_queries.clear()

