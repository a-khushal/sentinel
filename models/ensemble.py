from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from torch_geometric.data import Data

from .dga_detector import DGADetector
from .tdgnn import TDGNN
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
                 dga_model: Optional[DGADetector] = None,
                 gnn_model: Optional[TDGNN] = None,
                 dga_weight: float = 0.4,
                 gnn_weight: float = 0.6,
                 threshold: float = 0.7):
        self.dga_model = dga_model
        self.gnn_model = gnn_model
        self.dga_weight = dga_weight
        self.gnn_weight = gnn_weight
        self.threshold = threshold
        self.lexical = LexicalFeatures()
        self.graph_builder = GraphBuilder()
        
    def analyze_domain(self, domain: str) -> Dict:
        dga_result = None
        if self.dga_model:
            dga_results = self.dga_model.predict([domain])
            dga_result = dga_results[0] if dga_results else None
        
        lexical_features = self.lexical.extract(domain)
        
        dga_score = dga_result['dga_prob'] if dga_result else 0.0
        
        heuristic_score = self._heuristic_score(lexical_features)
        
        if dga_result:
            final_score = 0.7 * dga_score + 0.3 * heuristic_score
        else:
            final_score = heuristic_score
        
        return {
            'domain': domain,
            'is_suspicious': final_score > self.threshold,
            'confidence': final_score,
            'dga_score': dga_score,
            'heuristic_score': heuristic_score,
            'features': lexical_features,
            'threat_type': self.THREAT_DGA if final_score > self.threshold else None,
        }
    
    def analyze_graph(self, graphs: List[Data], node_mapping: Dict[str, int]) -> Dict:
        if not self.gnn_model:
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
    
    def analyze_traffic(self, queries: List, graphs: Optional[List[Data]] = None) -> List[ThreatDetection]:
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
        
        if graphs and self.gnn_model:
            node_mapping = {node: i for i, node in enumerate(graphs[-1].nodes()) 
                          if hasattr(graphs[-1], 'nodes') else {}}
            
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
        if entropy > 3.5:
            score += 0.3 * min(entropy / 4.5, 1.0)
        
        length = features.get('sld_length', 0)
        if length > 15:
            score += 0.2 * min((length - 15) / 20, 1.0)
        
        digit_ratio = features.get('digit_ratio', 0)
        if digit_ratio > 0.3:
            score += 0.2 * min(digit_ratio / 0.5, 1.0)
        
        consonant_seq = features.get('consonant_sequence', 0)
        if consonant_seq > 4:
            score += 0.15 * min((consonant_seq - 4) / 4, 1.0)
        
        hex_ratio = features.get('hex_ratio', 0)
        if hex_ratio > 0.8:
            score += 0.15
        
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
            'avg_confidence': np.mean([d.confidence for d in detections]),
            'domains': [d.domain for d in detections[:20]],
        }

