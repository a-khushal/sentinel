from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = APIRouter()

class ThreatResponse(BaseModel):
    id: str
    domain: str
    threat_type: str
    confidence: float
    dga_score: float
    gnn_score: float
    timestamp: str
    reported_to_blockchain: bool = False

class DomainAnalysisRequest(BaseModel):
    domain: str

class DomainAnalysisResponse(BaseModel):
    domain: str
    is_suspicious: bool
    confidence: float
    threat_type: Optional[str]
    dga_score: float
    heuristic_score: float
    features: dict

threats_db: List[dict] = []

def get_state():
    from ..main import state
    return state

@router.get("", response_model=List[ThreatResponse])
async def list_threats(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    threat_type: Optional[str] = None,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0)
):
    state = get_state()
    
    filtered = state.recent_threats
    
    if threat_type:
        filtered = [t for t in filtered if t.get('threat_type') == threat_type]
    
    filtered = [t for t in filtered if t.get('confidence', 0) >= min_confidence]
    
    return filtered[offset:offset + limit]

@router.get("/{threat_id}")
async def get_threat(threat_id: str):
    state = get_state()
    
    for threat in state.recent_threats:
        if threat.get('id') == threat_id:
            return threat
    
    raise HTTPException(status_code=404, detail="Threat not found")

@router.post("/analyze", response_model=DomainAnalysisResponse)
async def analyze_domain(request: DomainAnalysisRequest):
    state = get_state()
    
    if not state.ensemble:
        raise HTTPException(status_code=503, detail="Detection engine not initialized")
    
    state.increment_domains()
    result = state.ensemble.analyze_domain(request.domain)
    
    if result['is_suspicious']:
        import uuid
        threat = {
            'id': str(uuid.uuid4()),
            'domain': request.domain,
            'threat_type': result.get('threat_type', 'unknown'),
            'confidence': result['confidence'],
            'dga_score': result['dga_score'],
            'gnn_score': 0.0,
            'timestamp': datetime.now().isoformat(),
            'src_ip': 'manual',
            'reported_to_blockchain': False
        }
        state.add_threat(threat)
    
    return DomainAnalysisResponse(
        domain=result['domain'],
        is_suspicious=result['is_suspicious'],
        confidence=result['confidence'],
        threat_type=result.get('threat_type'),
        dga_score=result['dga_score'],
        heuristic_score=result['heuristic_score'],
        features=result['features']
    )

@router.post("/analyze/batch")
async def analyze_domains_batch(domains: List[str]):
    state = get_state()
    
    if not state.ensemble:
        raise HTTPException(status_code=503, detail="Detection engine not initialized")
    
    results = []
    for domain in domains[:100]:
        result = state.ensemble.analyze_domain(domain)
        results.append(result)
    
    state.increment_domains(len(domains))
    
    suspicious = [r for r in results if r['is_suspicious']]
    
    return {
        "total_analyzed": len(results),
        "suspicious_count": len(suspicious),
        "results": results
    }

@router.post("/{threat_id}/report")
async def report_to_blockchain(threat_id: str):
    state = get_state()
    
    threat = None
    for t in state.recent_threats:
        if t.get('id') == threat_id:
            threat = t
            break
    
    if not threat:
        raise HTTPException(status_code=404, detail="Threat not found")
    
    if not state.blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not connected")
    
    from blockchain.client import ThreatType
    
    threat_type_map = {
        'dga': ThreatType.DGA,
        'c2': ThreatType.C2,
        'tunnel': ThreatType.TUNNEL,
    }
    
    tx_hash = state.blockchain.report_threat(
        domain=threat['domain'],
        threat_type=threat_type_map.get(threat.get('threat_type', ''), ThreatType.UNKNOWN),
        confidence=int(threat['confidence'] * 100)
    )
    
    threat['reported_to_blockchain'] = True
    threat['tx_hash'] = tx_hash
    
    return {
        "success": True,
        "tx_hash": tx_hash,
        "domain": threat['domain']
    }

@router.get("/stats/summary")
async def get_threat_stats():
    state = get_state()
    
    threats = state.recent_threats
    
    by_type = {}
    for t in threats:
        tt = t.get('threat_type', 'unknown')
        by_type[tt] = by_type.get(tt, 0) + 1
    
    high_confidence = sum(1 for t in threats if t.get('confidence', 0) > 0.9)
    
    return {
        "total_threats": len(threats),
        "by_type": by_type,
        "high_confidence": high_confidence,
        "reported_to_blockchain": sum(1 for t in threats if t.get('reported_to_blockchain'))
    }

