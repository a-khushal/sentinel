# SENTINEL

Privacy-preserving botnet detection via federated graph learning and blockchain-anchored threat intelligence.

## What This Is

A decentralized framework for DNS-based botnet detection combining:
- Temporal Dynamic Graph Neural Network (T-DGNN) for detecting C2 patterns
- Federated learning with differential privacy for collaborative training
- Blockchain-based threat intelligence ledger

Target: IEEE S&P, NDSS, ACM CCS level publication + deployable application.

---

## Implementation Status

### Core Detection (Working)

| Component | File | Status |
|-----------|------|--------|
| DNS Sniffer | `capture/sniffer.py` | Done - Scapy-based packet capture |
| DNS Parser | `capture/parser.py` | Done - Extracts query/response data |
| Preprocessor | `capture/preprocessor.py` | Done - Cleans and normalizes |
| Lexical Features | `features/lexical.py` | Done - Entropy, n-grams, char ratios |
| Temporal Features | `features/temporal.py` | Done - Query rate, periodicity |
| Structural Features | `features/structural.py` | Done - NXDOMAIN ratio, TTL variance |
| Graph Builder | `features/graph_builder.py` | Done - NetworkX + PyG conversion |
| DGA Detector | `models/dga_detector.py` | Done - Transformer/LSTM classifier |
| T-DGNN | `models/tdgnn.py` | Done - GraphSAGE + GRU + temporal attention |
| Ensemble | `models/ensemble.py` | Done - Combines DGA + GNN scores |

### Federated Learning (Scaffolded)

| Component | File | Status |
|-----------|------|--------|
| FL Client | `federated/client.py` | Done - Flower client + DP noise |
| FL Server | `federated/server.py` | Done - FedAvg aggregation |
| Aggregation | - | Merged into server.py |
| Privacy | - | Merged into client.py (Gaussian mechanism) |

### Blockchain (Scaffolded)

| Component | File | Status |
|-----------|------|--------|
| ThreatLedger | `blockchain/contracts/ThreatLedger.sol` | Done - Commit-reveal, reputation |
| FederatedGovernance | `blockchain/contracts/FederatedGovernance.sol` | Done - Model proposals, voting |
| NodeRegistry | - | Merged into ThreatLedger.sol |
| Python Client | `blockchain/client.py` | Done - Web3.py + mock mode |
| Hardhat Config | `blockchain/hardhat.config.js` | Done - Polygon Mumbai setup |
| Deploy Script | `blockchain/scripts/deploy.js` | Done |

### API (Working)

| Endpoint | File | Status |
|----------|------|--------|
| `GET /` | `api/main.py` | Done - System status |
| `GET /api/v1/threats` | `api/routes/threats.py` | Done |
| `POST /api/v1/threats/analyze` | `api/routes/threats.py` | Done |
| `GET /api/v1/graph` | `api/routes/graph.py` | Done |
| `POST /api/v1/capture/start` | `api/routes/capture.py` | Done |
| `GET /api/v1/blockchain/status` | `api/routes/blockchain.py` | Done |
| `/api/v1/model/*` | - | Not implemented |
| `/api/v1/federation/*` | - | Not implemented |

### Dashboard (Working)

| Page | File | Status |
|------|------|--------|
| Dashboard | `dashboard/src/pages/Dashboard.tsx` | Done - Overview stats |
| Threat Monitor | `dashboard/src/pages/ThreatMonitor.tsx` | Done - Domain analysis, threat table |
| Graph View | `dashboard/src/pages/GraphView.tsx` | Done - Placeholder for force-graph |
| Blockchain | `dashboard/src/pages/Blockchain.tsx` | Done - Reports, reputation lookup |
| Model Metrics | - | Not implemented |
| Federation | - | Not implemented |

### Not Yet Implemented

| Component | Planned Location | Notes |
|-----------|------------------|-------|
| Tunnel Detector | `models/tunnel_detector.py` | DNS tunneling detection |
| Model Routes | `api/routes/model.py` | Metrics, training status |
| Federation Routes | `api/routes/federation.py` | Join network, status |
| ModelMetrics Page | `dashboard/src/pages/ModelMetrics.tsx` | ROC curves, confusion matrix |
| Federation Page | `dashboard/src/pages/Federation.tsx` | Peer nodes, training rounds |
| Evaluation Suite | `evaluation/` | Experiments for RQ1-RQ5 |
| Paper LaTeX | `paper/` | Main.tex, figures |
| TimescaleDB | - | Currently in-memory only |
| Redis | - | Currently no pub/sub |
| PCAP Upload | UI button exists | Handler incomplete |
| WebSocket Threats | Stubbed | Not pushing real-time |

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For ML models (optional, large download):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
```

For blockchain contracts:
```bash
cd blockchain && npm install
```

---

## Run

### Demo (No servers, mock data)
```bash
python scripts/demo.py
```

### Full Application
```bash
# Terminal 1: Backend
source .venv/bin/activate
python -m uvicorn api.main:app --port 8000

# Terminal 2: Frontend
cd dashboard && npm install && npm run dev
```

- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

---

## Detection Modes

| Mode | Requirements | What Works |
|------|--------------|------------|
| Heuristic | Base deps only | Lexical analysis, entropy, patterns |
| ML | + torch | DGA detection via trained model |
| Full | + torch-geometric | T-DGNN graph-based detection |

System auto-falls back to heuristic mode if torch not installed.

---

## Architecture

```
DNS Traffic → Sniffer → Parser → Feature Extraction → Graph Builder
                                        ↓
                              [Lexical][Temporal][Structural]
                                        ↓
                              T-DGNN / DGA Detector / Ensemble
                                        ↓
                              Threat Detection + Confidence Score
                                        ↓
                              Blockchain Report (optional)
```

---

## Project Structure

```
sentinel/
├── capture/          DNS packet capture (Scapy)
├── features/         Feature extraction + graph building
├── models/           Detection models (DGA, T-DGNN, ensemble)
├── federated/        FL client/server (Flower + DP)
├── blockchain/       Solidity contracts + Python client
├── api/              FastAPI backend
├── dashboard/        React frontend
└── scripts/          Demo, training, utilities
```

---

## Key Files

| Purpose | File |
|---------|------|
| Domain analysis | `models/ensemble.py` → `analyze_domain()` |
| Graph construction | `features/graph_builder.py` → `build_graph()` |
| DGA scoring | `features/lexical.py` → `extract()` |
| API entry | `api/main.py` |
| Mock demo | `scripts/demo.py` |

---

## Datasets (For Evaluation)

| Dataset | Use |
|---------|-----|
| CTU-13 | Primary botnet evaluation |
| ISOT | Cross-dataset validation |
| ISCX-Bot-2014 | Additional families |
| Custom Multi-Site | Federated evaluation (to create) |

---

## Research Questions (Planned)

- RQ1: T-DGNN vs baselines (BotGraph, DeepDGA, Kitsune)
- RQ2: Federated vs centralized accuracy gap
- RQ3: Differential privacy impact on detection
- RQ4: Blockchain latency and throughput
- RQ5: Adversarial robustness
