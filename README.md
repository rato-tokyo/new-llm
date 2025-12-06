# Infini-Pythia: Infinite Context with Compressive Memory

Pythia-70M with Infini-Attention for efficient long-context processing using compressive memory.

## Overview

Infini-Pythia extends Pythia-70M with Infini-Attention, enabling infinite context processing through learned compressive memory. The architecture uses a layer-based design for maximum flexibility.

```
Infini-Pythia Architecture:
  Token Embedding (512-dim)
         ↓
  Layer 0: InfiniLayer (NoPE, Memory Only)
    └─ Compressive Memory (linear attention)
         ↓
  Layer 1-5: PythiaLayer (RoPE)
    ├─ Multi-Head Attention
    └─ MLP
         ↓
  Output Head (512 → vocab)
```

## Key Features

- **Layer-based Design**: Compose models from reusable layer types
- **Compressive Memory**: O(1) memory for infinite context via linear attention
- **Delta Rule**: Efficient memory update preventing information overwrite
- **Memory Transfer**: Save/load memory state across devices
- **Multiple Memory Variants**: Infini, Multi-Memory, Hierarchical

## Quick Start

### Installation

```bash
pip install torch transformers datasets
```

### Create Models

```python
from src.models import create_model

# Standard Pythia
model = create_model("pythia")

# Infini-Pythia (Layer 0: Infini + 5 Pythia layers)
model = create_model("infini")

# Multi-Memory with 8 memories
model = create_model("multi_memory", num_memories=8)

# Hierarchical Memory
model = create_model("hierarchical", num_memories=4)
```

### Custom Layer Composition

```python
from src.models import TransformerLM
from src.models.layers import InfiniLayer, PythiaLayer

# Custom: 2 Infini layers + 4 Pythia layers
layers = [
    InfiniLayer(hidden_size=512, num_heads=8, intermediate_size=2048),
    InfiniLayer(hidden_size=512, num_heads=8, intermediate_size=2048),
    *[PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048) for _ in range(4)]
]
model = TransformerLM(layers=layers)

# Forward pass with memory update
output = model(input_ids, update_memory=True)

# Reset memory
model.reset_memory()
```

### Memory Transfer

```python
import torch
from src.models import create_model

# On PC A: Save memory state
model = create_model("infini")
# ... training or processing ...
state = model.get_memory_state()
torch.save(state, "memory.pt")

# On PC B: Load memory state
state = torch.load("memory.pt")
model = create_model("infini")
model.set_memory_state(state)
# Memory is now restored!
```

### Run Experiments

```bash
# Compare all models
python3 scripts/experiment.py --models pythia infini multi_memory hierarchical

# Infini only
python3 scripts/experiment.py --models infini

# Multi-Memory vs Hierarchical
python3 scripts/experiment.py --models multi_memory hierarchical --num-memories 4
```

## Model Variants

| Model | Description |
|-------|-------------|
| `pythia` | Standard Pythia-70M with RoPE |
| `infini` | Infini-Pythia (Layer 0: Infini + RoPE) |
| `multi_memory` | Multiple independent memories with attention-based selection |
| `hierarchical` | Hierarchical memory with coarse-to-fine retrieval |

## Layer Types

| Layer | Description |
|-------|-------------|
| `PythiaLayer` | Standard Pythia (RoPE + Softmax Attention) |
| `InfiniLayer` | Infini-Attention (Memory + Linear Attention, NoPE) |
| `MultiMemoryLayer` | Multiple independent memories |
| `HierarchicalLayer` | Hierarchical memory with expansion gate |

## Architecture Details

### Pythia-70M Base

| Component | Value |
|-----------|-------|
| Hidden Size | 512 |
| Layers | 6 |
| Attention Heads | 8 |
| Intermediate Size | 2048 |
| Position Encoding | Rotary (RoPE, 25%) |
| Vocab Size | 50,304 |

### Infini-Attention

```
Memory Update (Delta Rule):
  M_s = M_{s-1} + σ(K)^T @ (V - retrieved_V)

Memory Retrieval:
  A_mem = σ(Q) @ M / (σ(Q) @ z)

σ(x) = ELU(x) + 1
```

### Memory Sizes

| Model | Memory Size |
|-------|-------------|
| Infini | ~135 KB |
| Multi-Memory (4) | ~540 KB |
| Hierarchical (4) | ~540 KB |

## Project Structure

```
new-llm/
├── config/
│   └── pythia.py                   # PythiaConfig
├── scripts/
│   └── experiment.py               # Unified experiment script
├── src/
│   ├── data/
│   │   └── reversal_pairs.py       # Reversal Curse evaluation data
│   ├── models/
│   │   ├── __init__.py             # create_model() factory
│   │   ├── layers/                 # Layer package
│   │   │   ├── pythia.py           # PythiaLayer (RoPE + Softmax)
│   │   │   ├── infini.py           # InfiniLayer (Memory + Linear)
│   │   │   ├── multi_memory.py     # MultiMemoryLayer
│   │   │   └── hierarchical.py     # HierarchicalLayer
│   │   ├── model.py                # TransformerLM (generic model)
│   │   ├── base_components.py      # PythiaMLP, init_weights
│   │   ├── memory_utils.py         # Linear attention utilities
│   │   └── position_encoding.py    # RoPE
│   └── utils/
│       ├── experiment_runner.py    # Unified experiment runner
│       ├── training.py             # Training utilities
│       ├── evaluation.py           # Evaluation functions
│       └── ...
├── docs/
│   └── experiments/                # Experiment results
├── CLAUDE.md                       # Development guidelines
└── README.md
```

## Evaluation

### Reversal Curse

Measures if a model trained on "A is B" can also infer "B is A".

| Metric | Description |
|--------|-------------|
| Forward PPL | PPL on training direction |
| Backward PPL | PPL on reverse direction |
| Reversal Gap | Backward - Forward (lower is better) |

### Position-wise PPL

Evaluates model performance at different sequence positions.

## Development

See `CLAUDE.md` for development guidelines including:
- Layer-based architecture
- Model factory pattern
- Memory transfer API
- Code quality rules
- Past bugs and lessons

## License

MIT
