# Senri: Japanese LLM with Compressive Memory

Senri is a Japanese LLM with Infini-Attention for efficient long-context processing.

## Overview

Senri uses OpenCALM tokenizer with Infini-Attention, enabling infinite context processing through learned compressive memory. The architecture uses a layer-based design for maximum flexibility.

```
Architecture:
  Token Embedding (512-dim, vocab=52,000)
         |
  Layer 0: InfiniLayer (NoPE, Memory Only)
    - Compressive Memory (linear attention)
         |
  Layer 1-5: PythiaLayer (RoPE)
    - Multi-Head Attention
    - MLP
         |
  Output Head (512 -> vocab)
```

## Key Features

- **Japanese LLM**: OpenCALM tokenizer (UNK-free, byte_fallback support)
- **Layer-based Design**: Compose models from reusable layer types
- **Compressive Memory**: O(1) memory for infinite context via linear attention
- **Delta Rule**: Efficient memory update preventing information overwrite
- **Memory Transfer**: Save/load memory state across devices

## Quick Start

### Installation

```bash
pip install torch transformers datasets
```

### Create Models

```python
from src.models import create_model
from src.config import SenriConfig

# Standard model (Senri config, 52K vocab)
model = create_model("pythia")

# Infini model (Layer 0: Infini + 5 Pythia layers)
model = create_model("infini")

# Multi-Memory with 8 memories
model = create_model("multi_memory", num_memories=8)

# Custom config
config = SenriConfig()
model = create_model("pythia", base_config=config)
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
model = TransformerLM(layers=layers, vocab_size=52000, hidden_size=512)

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
# Senri integration verification
python3 scripts/verify_senri.py

# Context Separation Training (Reversal Curse)
python3 scripts/experiment_context_reasoning.py

# Baseline PPL evaluation
python3 scripts/evaluate_baseline.py
```

## Tokenizer

Using **CyberAgent OpenCALM** tokenizer for Japanese:

| Feature | Value |
|---------|-------|
| Vocab Size | 52,000 |
| UNK Token | None (byte_fallback) |
| Japanese | Full support |
| English | Full support (AI, API, GPU, etc.) |
| Emoji | Full support |

```python
from src.utils.tokenizer_utils import get_open_calm_tokenizer

tokenizer = get_open_calm_tokenizer()
# No UNK tokens for any input!
```

## Model Variants

| Model | Description |
|-------|-------------|
| `pythia` | Standard Transformer with RoPE |
| `infini` | Infini model (Layer 0: Infini + RoPE) |
| `multi_memory` | Multiple independent memories with Landmark selection |

## Layer Types

| Layer | Description |
|-------|-------------|
| `PythiaLayer` | Standard Pythia (RoPE + Softmax Attention) |
| `InfiniLayer` | Infini-Attention (Memory + Linear Attention, NoPE) |
| `MultiMemoryLayer` | Multiple independent memories with attention-based selection |

## Architecture Details

### Senri Config

| Component | Value |
|-----------|-------|
| Hidden Size | 512 |
| Layers | 6 |
| Attention Heads | 8 |
| Intermediate Size | 2048 |
| Position Encoding | Rotary (RoPE, 25%) |
| Vocab Size | 52,000 |
| Tokenizer | cyberagent/open-calm-small |

### Infini-Attention

```
Memory Update (Delta Rule):
  M_s = M_{s-1} + sigma(K)^T @ (V - retrieved_V)

Memory Retrieval:
  A_mem = sigma(Q) @ M / (sigma(Q) @ z)

sigma(x) = ELU(x) + 1
```

### Memory Sizes

| Model | Memory Size |
|-------|-------------|
| Infini | ~135 KB |
| Multi-Memory (4) | ~540 KB |

## Project Structure

```
senri/
├── src/
│   ├── config/
│   │   ├── senri.py              # SenriConfig (default)
│   │   ├── open_calm.py          # OpenCALM tokenizer constants
│   │   ├── pythia.py             # PythiaConfig (legacy)
│   │   ├── models.py             # InfiniConfig, MultiMemoryConfig
│   │   └── experiment.py         # ExperimentConfig
│   ├── models/
│   │   ├── __init__.py           # create_model() factory
│   │   ├── layers/               # Layer package
│   │   │   ├── base.py           # BaseLayer base class
│   │   │   ├── pythia.py         # PythiaLayer (RoPE + Softmax)
│   │   │   ├── infini.py         # InfiniLayer (Memory + Linear)
│   │   │   └── multi_memory.py   # MultiMemoryLayer
│   │   ├── model.py              # TransformerLM (generic model)
│   │   ├── base_components.py    # PythiaMLP, init_weights
│   │   ├── memory_utils.py       # Linear attention utilities
│   │   └── position_encoding.py  # RoPE
│   └── utils/
│       ├── tokenizer_utils.py    # OpenCALM tokenizer
│       ├── training.py           # Training utilities
│       └── ...
├── scripts/
│   ├── verify_senri.py           # Senri integration test
│   ├── experiment_context_reasoning.py  # Reversal Curse experiment
│   └── evaluate_baseline.py      # Baseline PPL evaluation
├── tests/
│   └── test_pythia_pretrained.py # Pretrained validation
├── docs/
│   └── experiments/              # Experiment results
├── CLAUDE.md                     # Development guidelines
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
