from typing import Dict, List, Optional
from dataclasses import dataclass, field
import threading

from models.ensemble import ThreatEnsemble
from features.graph_builder import GraphBuilder
from blockchain.client import MockBlockchainClient

try:
    from capture.sniffer import DNSSniffer
    from capture.preprocessor import DNSPreprocessor
    HAS_CAPTURE = True
except ImportError:
    HAS_CAPTURE = False
    DNSSniffer = None
    DNSPreprocessor = None

@dataclass
class AppState:
    capture_running: bool = False
    detection_active: bool = False
    blockchain_connected: bool = False
    model_loaded: bool = False
    
    total_queries: int = 0
    threats_detected: int = 0
    domains_analyzed: int = 0
    
    threat_queue: List[Dict] = field(default_factory=list)
    recent_threats: List[Dict] = field(default_factory=list)
    
    sniffer: Optional[object] = None
    preprocessor: Optional[object] = None
    graph_builder: Optional[GraphBuilder] = None
    ensemble: Optional[ThreatEnsemble] = None
    blockchain: Optional[MockBlockchainClient] = None
    
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def initialize(self):
        if HAS_CAPTURE:
            self.preprocessor = DNSPreprocessor()
        self.graph_builder = GraphBuilder()
        
        self.ensemble = ThreatEnsemble(auto_load=True)
        self.model_loaded = self.ensemble.using_ml
        
        self.blockchain = MockBlockchainClient()
        self.blockchain.register_node()
        self.blockchain_connected = True
        
        self.detection_active = True
    
    def cleanup(self):
        if self.sniffer:
            self.sniffer.stop()
    
    def add_threat(self, threat: Dict):
        with self._lock:
            self.threat_queue.append(threat)
            self.recent_threats.insert(0, threat)
            if len(self.recent_threats) > 100:
                self.recent_threats.pop()
            self.threats_detected += 1
    
    def increment_queries(self, count: int = 1):
        with self._lock:
            self.total_queries += count
    
    def increment_domains(self, count: int = 1):
        with self._lock:
            self.domains_analyzed += count
