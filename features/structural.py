from typing import Dict, List, Optional
from collections import defaultdict
import numpy as np
from capture.sniffer import DNSQuery

class StructuralFeatures:
    QUERY_TYPE_MAP = {
        1: 'A',
        2: 'NS',
        5: 'CNAME',
        6: 'SOA',
        12: 'PTR',
        15: 'MX',
        16: 'TXT',
        28: 'AAAA',
        33: 'SRV',
        255: 'ANY',
        10: 'NULL',
    }
    
    def __init__(self):
        self.domain_stats: Dict[str, Dict] = {}
    
    def extract(self, queries: List[DNSQuery]) -> Dict[str, float]:
        if not queries:
            return self._empty_features()
        
        return {
            'total_queries': len(queries),
            'unique_domains': len(set(q.query_name for q in queries)),
            'unique_clients': len(set(q.src_ip for q in queries)),
            'nxdomain_ratio': self._nxdomain_ratio(queries),
            'servfail_ratio': self._servfail_ratio(queries),
            'a_record_ratio': self._query_type_ratio(queries, 1),
            'aaaa_record_ratio': self._query_type_ratio(queries, 28),
            'txt_record_ratio': self._query_type_ratio(queries, 16),
            'null_record_ratio': self._query_type_ratio(queries, 10),
            'mx_record_ratio': self._query_type_ratio(queries, 15),
            'avg_response_size': self._avg_response_size(queries),
            'ttl_mean': self._ttl_stats(queries)['mean'],
            'ttl_std': self._ttl_stats(queries)['std'],
            'ttl_min': self._ttl_stats(queries)['min'],
            'avg_answers': self._avg_answers(queries),
            'query_type_entropy': self._query_type_entropy(queries),
        }
    
    def extract_domain(self, domain: str, queries: List[DNSQuery]) -> Dict[str, float]:
        domain_queries = [q for q in queries if q.query_name == domain]
        
        if not domain_queries:
            return self._empty_features()
        
        base = self.extract(domain_queries)
        
        base['client_count'] = len(set(q.src_ip for q in domain_queries))
        base['query_count'] = len(domain_queries)
        
        return base
    
    def _nxdomain_ratio(self, queries: List[DNSQuery]) -> float:
        if not queries:
            return 0.0
        responses = [q for q in queries if q.response_code is not None]
        if not responses:
            return 0.0
        nxdomain = sum(1 for q in responses if q.response_code == 3)
        return nxdomain / len(responses)
    
    def _servfail_ratio(self, queries: List[DNSQuery]) -> float:
        if not queries:
            return 0.0
        responses = [q for q in queries if q.response_code is not None]
        if not responses:
            return 0.0
        servfail = sum(1 for q in responses if q.response_code == 2)
        return servfail / len(responses)
    
    def _query_type_ratio(self, queries: List[DNSQuery], qtype: int) -> float:
        if not queries:
            return 0.0
        count = sum(1 for q in queries if q.query_type == qtype)
        return count / len(queries)
    
    def _avg_response_size(self, queries: List[DNSQuery]) -> float:
        sizes = []
        for q in queries:
            if q.answers:
                sizes.append(len(q.answers))
        return np.mean(sizes) if sizes else 0.0
    
    def _ttl_stats(self, queries: List[DNSQuery]) -> Dict[str, float]:
        ttls = [q.ttl for q in queries if q.ttl is not None]
        
        if not ttls:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': np.mean(ttls),
            'std': np.std(ttls),
            'min': min(ttls),
            'max': max(ttls),
        }
    
    def _avg_answers(self, queries: List[DNSQuery]) -> float:
        counts = [len(q.answers) if q.answers else 0 for q in queries]
        return np.mean(counts) if counts else 0.0
    
    def _query_type_entropy(self, queries: List[DNSQuery]) -> float:
        if not queries:
            return 0.0
        
        type_counts = defaultdict(int)
        for q in queries:
            type_counts[q.query_type] += 1
        
        total = len(queries)
        entropy = 0.0
        
        for count in type_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    def _empty_features(self) -> Dict[str, float]:
        return {
            'total_queries': 0,
            'unique_domains': 0,
            'unique_clients': 0,
            'nxdomain_ratio': 0.0,
            'servfail_ratio': 0.0,
            'a_record_ratio': 0.0,
            'aaaa_record_ratio': 0.0,
            'txt_record_ratio': 0.0,
            'null_record_ratio': 0.0,
            'mx_record_ratio': 0.0,
            'avg_response_size': 0.0,
            'ttl_mean': 0.0,
            'ttl_std': 0.0,
            'ttl_min': 0.0,
            'avg_answers': 0.0,
            'query_type_entropy': 0.0,
            'client_count': 0,
            'query_count': 0,
        }

