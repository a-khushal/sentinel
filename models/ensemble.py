from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    Data = Any

try:
    from .dga_detector import FeatureBasedDGA
except ImportError:
    FeatureBasedDGA = None

try:
    from .tdgnn import TDGNN
except ImportError:
    TDGNN = None

from features.lexical import LexicalFeatures
from features.graph_builder import GraphBuilder

@dataclass
class ThreatDetection:
    domain: str
    threat_type: str
    confidence: float
    dga_score: float
    gnn_score: float
    timestamp: datetime
    details: Dict

class ThreatEnsemble:
    THREAT_DGA = 'dga'
    THREAT_C2 = 'c2'
    THREAT_TUNNEL = 'tunnel'
    THREAT_UNKNOWN = 'unknown'
    
    def __init__(self,
                 dga_model: Optional[Any] = None,
                 gnn_model: Optional[Any] = None,
                 dga_weight: float = 0.6,
                 gnn_weight: float = 0.4,
                 threshold: float = 0.5,
                 auto_load: bool = True):
        self.dga_weight = dga_weight
        self.gnn_weight = gnn_weight
        self.threshold = threshold
        self.lexical = LexicalFeatures()
        self.graph_builder = GraphBuilder()
        
        self.dga_model = dga_model
        self.gnn_model = gnn_model
        self.using_ml = False
        
        if auto_load and dga_model is None and HAS_TORCH and FeatureBasedDGA:
            loaded = FeatureBasedDGA.load_trained()
            if loaded:
                self.dga_model = loaded
                self.using_ml = True
        
    def analyze_domain(self, domain: str) -> Dict:
        lexical_features = self.lexical.extract(domain)
        
        ml_score = 0.0
        if self.dga_model and HAS_TORCH:
            feature_vec = self._features_to_vector(lexical_features)
            ml_score = self.dga_model.predict_from_features(feature_vec)
        
        heuristic_score = self._heuristic_score(lexical_features)
        
        if self.using_ml:
            final_score = 0.7 * ml_score + 0.3 * heuristic_score
        else:
            final_score = heuristic_score
        
        return {
            'domain': domain,
            'is_suspicious': final_score > self.threshold,
            'confidence': final_score,
            'dga_score': ml_score,
            'heuristic_score': heuristic_score,
            'features': lexical_features,
            'threat_type': self.THREAT_DGA if final_score > self.threshold else None,
            'using_ml': self.using_ml,
        }
    
    def _features_to_vector(self, features: Dict) -> List[float]:
        return [
            features.get('entropy', 0) / 5.0,
            features.get('length', 0) / 50.0,
            features.get('sld_length', 0) / 30.0,
            features.get('digit_ratio', 0),
            features.get('vowel_ratio', 0),
            features.get('consonant_ratio', 0),
            features.get('special_ratio', 0),
            features.get('bigram_entropy', 0) / 5.0,
            features.get('trigram_entropy', 0) / 5.0,
            features.get('hex_ratio', 0),
            features.get('consonant_sequence', 0) / 10.0,
            features.get('numeric_sequence', 0) / 10.0,
        ]
    
    def analyze_graph(self, graphs: List, node_mapping: Dict[str, int]) -> Dict:
        if not self.gnn_model or not HAS_PYG:
            return {'error': 'GNN model not loaded'}
        
        graph_result = self.gnn_model.predict_graph(graphs)
        node_results = self.gnn_model.predict_nodes(graphs, node_mapping)
        
        suspicious_nodes = [n for n in node_results if n['is_infected']]
        
        return {
            'graph_malicious': graph_result['is_malicious'],
            'graph_confidence': graph_result['confidence'],
            'attention_weights': graph_result['attention_weights'].tolist(),
            'total_nodes': len(node_results),
            'suspicious_nodes': len(suspicious_nodes),
            'node_predictions': node_results,
            'threat_type': self.THREAT_C2 if graph_result['is_malicious'] else None,
        }
    
    def analyze_traffic(self, queries: List, graphs: Optional[List] = None) -> List[ThreatDetection]:
        detections = []
        
        domains = list(set(q.query_name for q in queries if q.query_name))
        
        for domain in domains:
            result = self.analyze_domain(domain)
            
            if result['is_suspicious']:
                detection = ThreatDetection(
                    domain=domain,
                    threat_type=result['threat_type'],
                    confidence=result['confidence'],
                    dga_score=result['dga_score'],
                    gnn_score=0.0,
                    timestamp=datetime.now(),
                    details=result
                )
                detections.append(detection)
        
        if graphs and self.gnn_model and HAS_PYG:
            node_mapping = {}
            if hasattr(graphs[-1], 'nodes'):
                node_mapping = {node: i for i, node in enumerate(graphs[-1].nodes())}
            
            graph_result = self.analyze_graph(graphs, node_mapping)
            
            if graph_result.get('graph_malicious'):
                for node in graph_result.get('node_predictions', []):
                    if node['is_infected'] and '.' in node['node_id']:
                        existing = next((d for d in detections if d.domain == node['node_id']), None)
                        
                        if existing:
                            existing.gnn_score = node['infected_prob']
                            existing.confidence = max(existing.confidence, 
                                                     self._combine_scores(existing.dga_score, node['infected_prob']))
                        else:
                            detection = ThreatDetection(
                                domain=node['node_id'],
                                threat_type=self.THREAT_C2,
                                confidence=node['infected_prob'],
                                dga_score=0.0,
                                gnn_score=node['infected_prob'],
                                timestamp=datetime.now(),
                                details={'node_prediction': node}
                            )
                            detections.append(detection)
        
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        return detections
    
    def _heuristic_score(self, features: Dict) -> float:
        score = 0.0
        
        entropy = features.get('entropy', 0)
        if entropy > 3.0:
            score += 0.35 * min(entropy / 4.0, 1.0)
        
        length = features.get('sld_length', 0)
        if length > 12:
            score += 0.25 * min((length - 12) / 15, 1.0)
        
        digit_ratio = features.get('digit_ratio', 0)
        if digit_ratio > 0.2:
            score += 0.2 * min(digit_ratio / 0.4, 1.0)
        
        consonant_seq = features.get('consonant_sequence', 0)
        if consonant_seq > 3:
            score += 0.1 * min((consonant_seq - 3) / 3, 1.0)
        
        hex_ratio = features.get('hex_ratio', 0)
        if hex_ratio > 0.6:
            score += 0.1 * min(hex_ratio, 1.0)
        
        return min(score, 1.0)
    
    def _combine_scores(self, dga_score: float, gnn_score: float) -> float:
        return self.dga_weight * dga_score + self.gnn_weight * gnn_score
    
    def get_threat_summary(self, detections: List[ThreatDetection]) -> Dict:
        if not detections:
            return {
                'total_threats': 0,
                'by_type': {},
                'high_confidence': 0,
                'domains': [],
            }
        
        by_type = {}
        for d in detections:
            by_type[d.threat_type] = by_type.get(d.threat_type, 0) + 1
        
        high_confidence = sum(1 for d in detections if d.confidence > 0.9)
        
        return {
            'total_threats': len(detections),
            'by_type': by_type,
            'high_confidence': high_confidence,
            'avg_confidence': float(np.mean([d.confidence for d in detections])),
            'domains': [d.domain for d in detections[:20]],
        }
