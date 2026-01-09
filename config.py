import os
from dataclasses import dataclass

@dataclass
class Config:
    polygon_rpc_url: str = os.getenv("POLYGON_RPC_URL", "https://rpc-mumbai.maticvigil.com")
    private_key: str = os.getenv("PRIVATE_KEY", "")
    contract_address: str = os.getenv("CONTRACT_ADDRESS", "")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    capture_interface: str = os.getenv("CAPTURE_INTERFACE", "eth0")
    
    model_path: str = "models/weights/"
    dga_model_path: str = "models/weights/dga_detector.pt"
    gnn_model_path: str = "models/weights/tdgnn.pt"
    
    graph_window_seconds: int = 300
    detection_threshold: float = 0.7
    max_domains_cache: int = 100000

config = Config()

