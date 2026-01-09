from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

class ReputationResponse(BaseModel):
    domain: str
    total_reports: int
    malicious_score: int
    first_seen: int
    last_seen: int
    reporter_count: int
    is_malicious: bool

class ReportRequest(BaseModel):
    domain: str
    threat_type: str
    confidence: int
    evidence: Optional[str] = ""

def get_state():
    from ..main import state
    return state

@router.get("/status")
async def get_blockchain_status():
    state = get_state()
    
    if not state.blockchain:
        return {
            "connected": False,
            "node_registered": False
        }
    
    return {
        "connected": state.blockchain.is_connected(),
        "node_registered": state.blockchain.is_node_registered(),
        "total_reports": state.blockchain.get_total_reports()
    }

@router.get("/reputation/{domain}")
async def get_domain_reputation(domain: str):
    state = get_state()
    
    if not state.blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not connected")
    
    rep = state.blockchain.query_reputation(domain)
    
    return ReputationResponse(
        domain=domain,
        total_reports=rep.total_reports,
        malicious_score=rep.malicious_score,
        first_seen=rep.first_seen,
        last_seen=rep.last_seen,
        reporter_count=rep.reporter_count,
        is_malicious=rep.malicious_score > 50
    )

@router.post("/report")
async def report_threat(request: ReportRequest):
    state = get_state()
    
    if not state.blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not connected")
    
    if not state.blockchain.is_node_registered():
        raise HTTPException(status_code=403, detail="Node not registered")
    
    from blockchain.client import ThreatType
    
    threat_type_map = {
        'dga': ThreatType.DGA,
        'c2': ThreatType.C2,
        'tunnel': ThreatType.TUNNEL,
        'unknown': ThreatType.UNKNOWN
    }
    
    threat_type = threat_type_map.get(request.threat_type.lower(), ThreatType.UNKNOWN)
    
    tx_hash = state.blockchain.report_threat(
        domain=request.domain,
        threat_type=threat_type,
        confidence=request.confidence,
        evidence=request.evidence or ""
    )
    
    return {
        "success": True,
        "tx_hash": tx_hash,
        "domain": request.domain,
        "threat_type": request.threat_type
    }

@router.get("/reports")
async def get_recent_reports(
    start: int = Query(0, ge=0),
    count: int = Query(50, ge=1, le=200)
):
    state = get_state()
    
    if not state.blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not connected")
    
    total = state.blockchain.get_total_reports()
    
    return {
        "total_reports": total,
        "start": start,
        "count": min(count, total - start) if total > start else 0
    }

@router.post("/register")
async def register_node(stake: float = Query(0.01, ge=0.01)):
    state = get_state()
    
    if not state.blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not connected")
    
    if state.blockchain.is_node_registered():
        return {"message": "Node already registered"}
    
    tx_hash = state.blockchain.register_node(stake)
    
    return {
        "success": True,
        "tx_hash": tx_hash,
        "stake": stake
    }

@router.get("/node/stats")
async def get_node_stats():
    state = get_state()
    
    if not state.blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not connected")
    
    return state.blockchain.get_node_stats() if hasattr(state.blockchain, 'get_node_stats') else {
        "registered": state.blockchain.is_node_registered(),
        "reports": state.blockchain.get_total_reports()
    }

