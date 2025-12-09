# Senri: Japanese LLM with Compressive Memory

Senri is a Japanese LLM with unified compressive memory for efficient long-context processing.

## Overview

Senri uses OpenCALM tokenizer with SenriLayer (unified compressive memory), enabling infinite context processing through learned memory. The architecture uses a simple layer-based design with just 2 layer types.

```
Architecture:
  Token Embedding (512-dim, vocab=52,000)
         |
  Layer 0: SenriLayer (NoPE, Memory Only)
    - Compressive Memory (linear attention)
    - Configurable: num_memories, memory_head_dim
         |
  Layer 1-5: PythiaLayer (RoPE)
    - Multi-Head Attention
    - MLP
         |
  Output Head (512 -> vocab)
```

## Key Features

- **Japanese LLM**: OpenCALM tokenizer (UNK-free, byte_fallback support)
- **Simple Design**: Only 2 layer types (PythiaLayer, SenriLayer)
- **Compressive Memory**: O(1) memory for infinite context via linear attention
- **Flexible Configuration**: num_memories, memory_head_dim parameters
- **Delta Rule**: Efficient memory update preventing information overwrite
- **Memory Transfer**: Save/load memory state across devices

## Quick Start

### Installation

```bash
pip install torch transformers datasets
```

### Create Models

```python
from src.models import TransformerLM, senri_layers, pythia_layers

# Senri model (1 SenriLayer + 5 PythiaLayers)
model = TransformerLM(layers=senri_layers(1) + pythia_layers(5), vocab_size=52000)

# Pythia-only baseline
model = TransformerLM(layers=pythia_layers(6), vocab_size=52000)

# Multiple memories
model = TransformerLM(
    layers=senri_layers(1, num_memories=4) + pythia_layers(5),
    vocab_size=52000,
)

# Custom memory head dimension
model = TransformerLM(
    layers=senri_layers(1, memory_head_dim=256) + pythia_layers(5),
    vocab_size=52000,
)

# Forward pass with memory update
output = model(input_ids, update_memory=True)

# Reset memory
model.reset_memory()
```

### Memory Transfer

```python
import torch
from src.models import TransformerLM, senri_layers, pythia_layers

# On PC A: Save memory state
model = TransformerLM(layers=senri_layers(1) + pythia_layers(5), vocab_size=52000)
# ... training or processing ...
state = model.get_memory_state()
torch.save(state, "memory.pt")

# On PC B: Load memory state
state = torch.load("memory.pt")
model = TransformerLM(layers=senri_layers(1) + pythia_layers(5), vocab_size=52000)
model.set_memory_state(state)
# Memory is now restored!
```

### Run Experiments

```bash
# Senri integration verification
python3 scripts/verify_senri.py

# Context Separation Training (Reversal Curse)
python3 scripts/experiment_context_reasoning.py
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

## Layer Types

| Layer | Description |
|-------|-------------|
| `PythiaLayer` | Standard Pythia (RoPE + Softmax Attention) |
| `SenriLayer` | Unified memory layer (Linear Attention, NoPE). Configurable: num_memories, memory_head_dim |

### SenriLayer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_memories` | 1 | Number of memory slots |
| `memory_head_dim` | None (=hidden_size) | Memory head dimension. None = single-head (512) |
| `use_delta_rule` | True | Use delta rule for memory update |

## Architecture Details

### Default Configuration

| Component | Value |
|-----------|-------|
| Hidden Size | 512 |
| Layers | 6 |
| Attention Heads | 8 |
| Intermediate Size | 2048 |
| Position Encoding | Rotary (RoPE, 25%) |
| Vocab Size | 52,000 |
| Tokenizer | cyberagent/open-calm-small |

### SenriAttention (Compressive Memory)

```
Memory Update (Delta Rule):
  M_s = M_{s-1} + sigma(K)^T @ (V - retrieved_V)

Memory Retrieval:
  A_mem = sigma(Q) @ M / (sigma(Q) @ z)

sigma(x) = ELU(x) + 1
```

### Memory Sizes

| Configuration | Memory Size |
|---------------|-------------|
| SenriLayer (num_memories=1) | ~135 KB |
| SenriLayer (num_memories=4) | ~540 KB |

## Project Structure

```
senri/
├── src/
│   ├── config/
│   │   ├── __init__.py           # Constants + ExperimentConfig
│   │   ├── constants.py          # OPEN_CALM_TOKENIZER, OPEN_CALM_VOCAB_SIZE
│   │   └── experiments/
│   │       └── base.py           # ExperimentConfig
│   ├── models/
│   │   ├── __init__.py           # Layer factories (senri_layers, pythia_layers)
│   │   ├── layers/
│   │   │   ├── base.py           # BaseLayer
│   │   │   ├── pythia.py         # PythiaLayer
│   │   │   └── senri.py          # SenriLayer (unified memory layer)
│   │   ├── model.py              # TransformerLM
│   │   ├── base_components.py    # PythiaMLP, init_weights
│   │   ├── memory_utils.py       # Linear attention utilities
│   │   └── position_encoding.py  # RoPE
│   └── utils/
│       ├── tokenizer_utils.py    # OpenCALM tokenizer
│       ├── training.py           # Training utilities
│       └── evaluation.py         # Evaluation utilities
├── scripts/
│   ├── verify_senri.py           # Senri integration test
│   └── experiment_context_reasoning.py  # Reversal Curse experiment
├── tests/
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

## Development

See `CLAUDE.md` for development guidelines including:
- Layer-based architecture (2 layer types only)
- SenriLayer configuration (num_memories, memory_head_dim)
- Memory transfer API
- Code quality rules
- Past bugs and lessons

## License

MIT
