from typing import Dict, List
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
from capture.sniffer import DNSQuery

class TemporalFeatures:
    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self.query_history: Dict[str, List[datetime]] = defaultdict(list)
    
    def extract_client_features(self, client_ip: str, 
                                 queries: List[DNSQuery]) -> Dict[str, float]:
        if not queries:
            return self._empty_features()
        
        timestamps = sorted([q.timestamp for q in queries])
        domains = [q.query_name for q in queries]
        
        return {
            'query_count': len(queries),
            'unique_domains': len(set(domains)),
            'query_rate': self._query_rate(timestamps),
            'burst_score': self._burst_score(timestamps),
            'periodicity_score': self._periodicity_score(timestamps),
            'time_spread': self._time_spread(timestamps),
            'inter_arrival_mean': self._inter_arrival_mean(timestamps),
            'inter_arrival_std': self._inter_arrival_std(timestamps),
            'nxdomain_rate': self._nxdomain_rate(queries),
            'failed_ratio': self._failed_ratio(queries),
        }
    
    def extract_domain_features(self, domain: str,
                                 queries: List[DNSQuery]) -> Dict[str, float]:
        if not queries:
            return self._empty_features()
        
        timestamps = sorted([q.timestamp for q in queries])
        clients = [q.src_ip for q in queries]
        
        return {
            'query_count': len(queries),
            'unique_clients': len(set(clients)),
            'query_rate': self._query_rate(timestamps),
            'burst_score': self._burst_score(timestamps),
            'periodicity_score': self._periodicity_score(timestamps),
            'time_spread': self._time_spread(timestamps),
            'inter_arrival_mean': self._inter_arrival_mean(timestamps),
            'inter_arrival_std': self._inter_arrival_std(timestamps),
            'client_fanout': len(set(clients)) / max(len(queries), 1),
            'response_variance': self._response_variance(queries),
        }
    
    def _query_rate(self, timestamps: List[datetime]) -> float:
        if len(timestamps) < 2:
            return 0.0
        
        duration = (timestamps[-1] - timestamps[0]).total_seconds()
        if duration == 0:
            return float(len(timestamps))
        
        return len(timestamps) / duration
    
    def _burst_score(self, timestamps: List[datetime]) -> float:
        if len(timestamps) < 3:
            return 0.0
        
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals:
            return 0.0
        
        mean_interval = np.mean(intervals)
        if mean_interval == 0:
            return 1.0
        
        short_intervals = sum(1 for i in intervals if i < mean_interval * 0.1)
        return short_intervals / len(intervals)
    
    def _periodicity_score(self, timestamps: List[datetime]) -> float:
        if len(timestamps) < 5:
            return 0.0
        
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        if not intervals or len(intervals) < 3:
            return 0.0
        
        mean_interval = np.mean(intervals)
        if mean_interval == 0:
            return 0.0
        
        std_interval = np.std(intervals)
        cv = std_interval / mean_interval
        
        return max(0.0, 1.0 - cv)
    
    def _time_spread(self, timestamps: List[datetime]) -> float:
        if len(timestamps) < 2:
            return 0.0
        
        return (timestamps[-1] - timestamps[0]).total_seconds()
    
    def _inter_arrival_mean(self, timestamps: List[datetime]) -> float:
        if len(timestamps) < 2:
            return 0.0
        
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        return np.mean(intervals) if intervals else 0.0
    
    def _inter_arrival_std(self, timestamps: List[datetime]) -> float:
        if len(timestamps) < 3:
            return 0.0
        
        intervals = []
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i-1]).total_seconds()
            intervals.append(interval)
        
        return np.std(intervals) if intervals else 0.0
    
    def _nxdomain_rate(self, queries: List[DNSQuery]) -> float:
        if not queries:
            return 0.0
        
        nxdomain = sum(1 for q in queries if q.response_code == 3)
        return nxdomain / len(queries)
    
    def _failed_ratio(self, queries: List[DNSQuery]) -> float:
        if not queries:
            return 0.0
        
        failed = sum(1 for q in queries if q.response_code and q.response_code != 0)
        return failed / len(queries)
    
    def _response_variance(self, queries: List[DNSQuery]) -> float:
        ttls = [q.ttl for q in queries if q.ttl is not None]
        if len(ttls) < 2:
            return 0.0
        return np.var(ttls)
    
    def _empty_features(self) -> Dict[str, float]:
        return {
            'query_count': 0,
            'unique_domains': 0,
            'unique_clients': 0,
            'query_rate': 0.0,
            'burst_score': 0.0,
            'periodicity_score': 0.0,
            'time_spread': 0.0,
            'inter_arrival_mean': 0.0,
            'inter_arrival_std': 0.0,
            'nxdomain_rate': 0.0,
            'failed_ratio': 0.0,
            'client_fanout': 0.0,
            'response_variance': 0.0,
        }

