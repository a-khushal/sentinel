from typing import Dict, List, Optional
from dataclasses import dataclass, field
import threading
import os

from models.ensemble import ThreatEnsemble
from features.graph_builder import GraphBuilder
from blockchain.client import get_blockchain_client, MockBlockchainClient, ThreatType

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
        
        self.blockchain = get_blockchain_client()
        self.blockchain_connected = self.blockchain.is_connected()
        
        is_mock = getattr(self.blockchain, 'mock_mode', False)
        if is_mock:
            self.blockchain.register_node()
        
        self.detection_active = True
    
    def cleanup(self):
        if self.sniffer:
            self.sniffer.stop()
    
    def add_threat(self, threat: Dict, auto_report: bool = True):
        with self._lock:
            self.threat_queue.append(threat)
            self.recent_threats.insert(0, threat)
            if len(self.recent_threats) > 100:
                self.recent_threats.pop()
            self.threats_detected += 1
        
        if auto_report and self.blockchain and self.blockchain_connected:
            confidence = threat.get('confidence', 0)
            if confidence >= 0.8:
                def _report():
                    try:
                        threat_type_map = {
                            'dga': ThreatType.DGA,
                            'c2': ThreatType.C2,
                            'tunnel': ThreatType.TUNNEL,
                        }
                        tx_hash = self.blockchain.report_threat(
                            domain=threat['domain'],
                            threat_type=threat_type_map.get(threat.get('threat_type', ''), ThreatType.UNKNOWN),
                            confidence=int(confidence * 100),
                            evidence=f"Auto-detected by SENTINEL ML at {threat.get('timestamp', '')}"
                        )
                        with self._lock:
                            threat['reported_to_blockchain'] = True
                            threat['tx_hash'] = tx_hash
                        print(f"Auto-reported {threat['domain']} to blockchain: {tx_hash}")
                    except Exception as e:
                        print(f"Auto-report failed for {threat['domain']}: {e}")
                threading.Thread(target=_report, daemon=True).start()
    
    def increment_queries(self, count: int = 1):
        with self._lock:
            self.total_queries += count
    
    def increment_domains(self, count: int = 1):
        with self._lock:
            self.domains_analyzed += count
