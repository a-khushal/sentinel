# Evaluation Experiments

This directory contains evaluation scripts for the research questions (RQ1-RQ5) outlined in the paper.

## Structure

```
evaluation/
├── experiments/
│   ├── rq1_detection.py      # RQ1: T-DGNN vs baselines
│   ├── rq2_federated.py     # RQ2: Federated vs centralized
│   ├── rq3_privacy.py       # RQ3: Privacy-utility trade-off
│   ├── rq4_blockchain.py    # RQ4: Blockchain performance
│   └── rq5_adversarial.py   # RQ5: Adversarial robustness
├── results/                 # Output JSON files with results
└── datasets/                # Dataset files (CTU-13, ISOT, etc.)
```

## Running Experiments

### RQ1: Detection Accuracy

Compare T-DGNN against state-of-the-art baselines (BotGraph, DeepDGA, Kitsune):

```bash
python evaluation/experiments/rq1_detection.py
```

This will:
- Load CTU-13 dataset
- Evaluate DGA Detector and Ensemble models
- Compare against baseline results
- Save results to `evaluation/results/rq1_detection.json`

### Quick CTU-13 Test

Test CTU-13 dataset loading and basic evaluation:

```bash
python scripts/evaluate_ctu13.py
```

## CTU-13 Dataset

The CTU-13 dataset contains 13 botnet scenarios with network flow data in bi-netflow format.

**Location:** `data/ctu13/*.binetflow`

**Format:** Tab-separated CSV with columns:
- StartTime, Dur, Proto, SrcAddr, Sport, DstAddr, Dport, Label, etc.

**Usage:**
```python
from scripts.ctu13_loader import load_all_ctu13_scenarios

domains, labels, metadata = load_all_ctu13_scenarios()
```

## Results Format

Each experiment outputs a JSON file with:
- Experiment metadata (timestamp, dataset info)
- Model performance metrics (accuracy, precision, recall, F1, AUC)
- Comparison with baselines
- Confusion matrix (TP, FP, TN, FN)

## Requirements

All experiments require:
- PyTorch
- scikit-learn (for ROC-AUC calculation)
- Trained models in `models/weights/`

