from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from typing import List
import json

from .routes import threats, graph, blockchain, capture, federation, model
from .state import AppState

state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    state.initialize()
    yield
    state.cleanup()

app = FastAPI(
    title="SENTINEL API",
    description="Decentralized Botnet Detection System",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(threats.router, prefix="/api/v1/threats", tags=["Threats"])
app.include_router(graph.router, prefix="/api/v1/graph", tags=["Graph"])
app.include_router(blockchain.router, prefix="/api/v1/blockchain", tags=["Blockchain"])
app.include_router(capture.router, prefix="/api/v1/capture", tags=["Capture"])
app.include_router(federation.router, prefix="/api/v1/federation", tags=["Federation"])
app.include_router(model.router, prefix="/api/v1/model", tags=["Model"])

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws/threats")
async def websocket_threats(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            if state.threat_queue:
                threat = state.threat_queue.pop(0)
                await websocket.send_json({
                    "type": "threat",
                    "data": threat
                })
            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/")
async def root():
    return {
        "name": "SENTINEL",
        "version": "1.0.0",
        "status": "running",
        "components": {
            "capture": state.capture_running,
            "detection": state.detection_active,
            "blockchain": state.blockchain_connected
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/v1/stats")
async def get_stats():
    return {
        "total_queries_processed": state.total_queries,
        "threats_detected": state.threats_detected,
        "domains_analyzed": state.domains_analyzed,
        "capture_running": state.capture_running,
        "model_loaded": state.model_loaded
    }

