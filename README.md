# MNAR-Net 
Official implementation of **"An Adaptive Noise-Robust Incremental Learning Framework for Long-Term Condition Monitoring of Shipboard Pump Drive Systems"**.

- **MDCT**: a multi-scale dilated convolutional transformer backbone for noise-robust feature extraction.
- **NAER**: a noise-adaptive exemplar replay mechanism for class-incremental learning.
- **MNAR-Net**: the integration of MDCT and NAER.


## Project Structure

```text
MNAR-Net/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── data/
│   └── dataset_reference.txt
├── examples/
│   ├── minimal_forward.py
│   └── minimal_incremental.py
└── src/
    └── mnar_net/
        ├── __init__.py
        ├── data/
        │   ├── __init__.py
        │   └── preprocessing.py
        ├── models/
        │   ├── __init__.py
        │   ├── mdct.py
        │   └── mnar_net.py
        ├── modules/
        │   ├── __init__.py
        │   ├── blocks.py
        │   ├── mdrc.py
        │   └── transformer.py
        └── replay/
            ├── __init__.py
            └── naer.py
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### 1. Forward pass with MDCT

```python
import torch
from mnar_net import MDCTBackbone

model = MDCTBackbone(num_classes=9, signal_length=1024)
x = torch.randn(8, 3, 1024)
logits = model(x)
print(logits.shape)
```

### 2. Incremental wrapper with NAER

```python
import torch
from mnar_net import MNARNet

model = MNARNet(num_classes=3, signal_length=1024, memory_per_class=20)
x_clean = torch.randn(12, 3, 1024)
x_noisy = torch.randn(12, 3, 1024)
y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

model.update_memory(x_clean=x_clean, x_noisy=x_noisy, labels=y)
replay_x, replay_y = model.sample_replay(snr_db=-10.0, seed=42)
print(replay_x.shape, replay_y.shape)
```

## Citation and Dataset

See `data/dataset_reference.txt` for the RNNEP/NLN-EMP dataset note used by the manuscript.
