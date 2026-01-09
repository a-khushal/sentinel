from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import os
import time
import threading

router = APIRouter()

capture_state = {
    'queries': [],
    'start_time': None,
    'duration': 0,
    'interface': 'any',
    'max_duration': 60,
    'timer': None
}

class CaptureConfig(BaseModel):
    interface: str = "any"
    duration: int = 10

class CaptureStatus(BaseModel):
    capturing: bool
    queries_captured: int
    duration_seconds: float
    interface: str

def get_state():
    from ..main import state
    return state

@router.get("/status")
async def get_capture_status():
    state = get_state()
    
    duration = 0
    if capture_state['start_time'] and state.capture_running:
        duration = time.time() - capture_state['start_time']
    
    return {
        "capturing": state.capture_running,
        "queries_captured": len(capture_state['queries']),
        "duration_seconds": round(duration, 1),
        "interface": capture_state['interface']
    }

@router.post("/start")
async def start_capture(config: CaptureConfig):
    state = get_state()
    
    if state.capture_running:
        return {"message": "Capture already running", "status": "running"}
    
    from capture.sniffer import DNSSniffer
    
    capture_state['queries'] = []
    capture_state['start_time'] = time.time()
    capture_state['interface'] = config.interface
    capture_state['max_duration'] = config.duration
    
    def on_query(query):
        state.increment_queries()
        capture_state['queries'].append(query)
        
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
    
    def auto_stop():
        time.sleep(config.duration)
        if state.capture_running:
            if state.sniffer:
                state.sniffer.stop()
            state.capture_running = False
    
    state.sniffer = DNSSniffer(interface=config.interface, callback=on_query)
    
    try:
        state.sniffer.start()
        state.capture_running = True
        
        capture_state['timer'] = threading.Thread(target=auto_stop, daemon=True)
        capture_state['timer'].start()
        
        return {"message": f"Capture started on {config.interface} for {config.duration}s", "status": "started"}
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied. Run with sudo for packet capture.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_capture():
    state = get_state()
    
    if not state.capture_running:
        return {"message": "Capture not running", "queries_captured": len(capture_state['queries'])}
    
    if state.sniffer:
        state.sniffer.stop()
    
    state.capture_running = False
    capture_state['duration'] = time.time() - (capture_state['start_time'] or time.time())
    
    return {
        "message": "Capture stopped",
        "queries_captured": len(capture_state['queries']),
        "duration": round(capture_state['duration'], 1)
    }

@router.get("/queries")
async def get_captured_queries(limit: int = 100):
    queries = capture_state['queries'][-limit:]
    
    return {
        "total": len(capture_state['queries']),
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

@router.get("/graph-data")
async def get_graph_from_captured():
    state = get_state()
    queries = capture_state['queries']
    
    if not queries:
        return {"nodes": [], "edges": [], "stats": {"total": 0}}
    
    if state.graph_builder:
        G = state.graph_builder.build_graph(queries)
        
        nodes = []
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            nodes.append({
                "id": str(node_id),
                "type": node_data.get('node_type', 'unknown'),
                "label": str(node_id)[:30],
                "features": node_data.get('features', {})
            })
        
        edges = []
        for u, v, data in G.edges(data=True):
            edges.append({
                "source": str(u),
                "target": str(v),
                "type": data.get('edge_type', 'queries'),
                "weight": data.get('weight', 1)
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_queries": len(queries),
                "unique_clients": len(set(q.src_ip for q in queries)),
                "unique_domains": len(set(q.query_name for q in queries)),
                "nodes": len(nodes),
                "edges": len(edges)
            }
        }
    
    return {"nodes": [], "edges": [], "stats": {}}

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
        
        capture_state['queries'] = queries
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
            "unique_domains": len(set(q.query_name for q in queries)),
            "unique_clients": len(set(q.src_ip for q in queries)),
            "threats_detected": len(threats_found)
        }
    
    finally:
        os.unlink(tmp_path)

@router.get("/clients")
async def get_active_clients():
    queries = capture_state['queries']
    
    if not queries:
        return {"clients": []}
    
    from collections import defaultdict
    client_stats = defaultdict(lambda: {'queries': [], 'domains': set()})
    
    for q in queries:
        client_stats[q.src_ip]['queries'].append(q)
        client_stats[q.src_ip]['domains'].add(q.query_name)
    
    clients = []
    for client_ip, stats in client_stats.items():
        clients.append({
            "ip": client_ip,
            "query_count": len(stats['queries']),
            "unique_domains": len(stats['domains'])
        })
    
    clients.sort(key=lambda x: x['query_count'], reverse=True)
    
    return {"clients": clients[:100]}
