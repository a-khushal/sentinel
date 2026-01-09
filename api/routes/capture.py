from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os

router = APIRouter()

class CaptureConfig(BaseModel):
    interface: str = "eth0"

class CaptureStatus(BaseModel):
    running: bool
    interface: Optional[str]
    packets_captured: int
    queries_processed: int

def get_state():
    from ..main import state
    return state

@router.get("/status", response_model=CaptureStatus)
async def get_capture_status():
    state = get_state()
    
    return CaptureStatus(
        running=state.capture_running,
        interface=state.sniffer.interface if state.sniffer else None,
        packets_captured=state.total_queries,
        queries_processed=state.total_queries
    )

@router.post("/start")
async def start_capture(config: CaptureConfig):
    state = get_state()
    
    if state.capture_running:
        return {"message": "Capture already running"}
    
    from capture.sniffer import DNSSniffer
    
    def on_query(query):
        state.increment_queries()
        
        if state.ensemble:
            result = state.ensemble.analyze_domain(query.query_name)
            
            if result['is_suspicious']:
                import uuid
                threat = {
                    'id': str(uuid.uuid4()),
                    'domain': query.query_name,
                    'threat_type': result.get('threat_type', 'unknown'),
                    'confidence': result['confidence'],
                    'dga_score': result['dga_score'],
                    'gnn_score': 0.0,
                    'timestamp': query.timestamp.isoformat(),
                    'src_ip': query.src_ip,
                    'reported_to_blockchain': False
                }
                state.add_threat(threat)
    
    state.sniffer = DNSSniffer(interface=config.interface, callback=on_query)
    
    try:
        state.sniffer.start()
        state.capture_running = True
        return {"message": f"Capture started on {config.interface}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_capture():
    state = get_state()
    
    if not state.capture_running:
        return {"message": "Capture not running"}
    
    if state.sniffer:
        state.sniffer.stop()
    
    state.capture_running = False
    return {"message": "Capture stopped"}

@router.post("/upload")
async def upload_pcap(file: UploadFile = File(...)):
    state = get_state()
    
    if not file.filename.endswith('.pcap') and not file.filename.endswith('.pcapng'):
        raise HTTPException(status_code=400, detail="File must be .pcap or .pcapng")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pcap') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        from capture.parser import DNSParser
        
        parser = DNSParser()
        queries = parser.parse_pcap(tmp_path)
        
        state.preprocessor.process_queries(queries)
        state.increment_queries(len(queries))
        
        if state.graph_builder:
            state.graph_builder.build_graph(queries)
        
        threats_found = []
        if state.ensemble:
            for query in queries:
                result = state.ensemble.analyze_domain(query.query_name)
                if result['is_suspicious']:
                    import uuid
                    threat = {
                        'id': str(uuid.uuid4()),
                        'domain': query.query_name,
                        'threat_type': result.get('threat_type', 'unknown'),
                        'confidence': result['confidence'],
                        'dga_score': result['dga_score'],
                        'gnn_score': 0.0,
                        'timestamp': query.timestamp.isoformat(),
                        'src_ip': query.src_ip,
                        'reported_to_blockchain': False
                    }
                    state.add_threat(threat)
                    threats_found.append(threat)
        
        return {
            "message": "PCAP processed successfully",
            "queries_parsed": len(queries),
            "unique_domains": len(parser.get_unique_domains()),
            "unique_clients": len(parser.get_unique_clients()),
            "threats_detected": len(threats_found)
        }
    
    finally:
        os.unlink(tmp_path)

@router.get("/queries")
async def get_recent_queries(limit: int = 100):
    state = get_state()
    
    if not state.sniffer:
        return {"queries": []}
    
    queries = state.sniffer.get_queries(limit)
    
    return {
        "queries": [
            {
                "timestamp": q.timestamp.isoformat(),
                "src_ip": q.src_ip,
                "domain": q.query_name,
                "query_type": q.query_type,
                "response_code": q.response_code
            }
            for q in queries
        ]
    }

@router.get("/clients")
async def get_active_clients():
    state = get_state()
    
    if not state.preprocessor:
        return {"clients": []}
    
    clients = []
    for client_ip, queries in state.preprocessor.client_queries.items():
        clients.append({
            "ip": client_ip,
            "query_count": len(queries),
            "unique_domains": len(set(q.query_name for q in queries))
        })
    
    clients.sort(key=lambda x: x['query_count'], reverse=True)
    
    return {"clients": clients[:100]}

