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
from src.models import TransformerLM, senri_layers, pythia_layers, multi_memory_layers

# Senri model (1 Infini + 5 Pythia layers)
model = TransformerLM(layers=senri_layers(), vocab_size=52000)

# Pythia-only baseline
model = TransformerLM(layers=pythia_layers(6), vocab_size=52000)

# Custom configuration
model = TransformerLM(
    layers=multi_memory_layers(1, num_memories=8) + pythia_layers(5),
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
from src.models import TransformerLM, senri_layers

# On PC A: Save memory state
model = TransformerLM(layers=senri_layers(), vocab_size=52000)
# ... training or processing ...
state = model.get_memory_state()
torch.save(state, "memory.pt")

# On PC B: Load memory state
state = torch.load("memory.pt")
model = TransformerLM(layers=senri_layers(), vocab_size=52000)
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

## Layer Types

| Layer | Description |
|-------|-------------|
| `PythiaLayer` | Standard Pythia (RoPE + Softmax Attention) |
| `InfiniLayer` | Infini-Attention (Memory + Linear Attention, NoPE) |
| `MultiMemoryLayer` | Multiple independent memories with attention-based selection |

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
│   │   ├── __init__.py           # Constants + ExperimentConfig
│   │   ├── constants.py          # OPEN_CALM_TOKENIZER, OPEN_CALM_VOCAB_SIZE
│   │   └── experiments/
│   │       └── base.py           # ExperimentConfig
│   ├── models/
│   │   ├── __init__.py           # Layer factories (senri_layers, pythia_layers)
│   │   ├── layers/
│   │   │   ├── base.py           # BaseLayer
│   │   │   ├── pythia.py         # PythiaLayer
│   │   │   ├── infini.py         # InfiniLayer
│   │   │   └── multi_memory.py   # MultiMemoryLayer
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
│   ├── experiment_context_reasoning.py  # Reversal Curse experiment
│   └── evaluate_baseline.py      # Baseline PPL evaluation
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
- Layer-based architecture
- SenriModelConfig usage
- Memory transfer API
- Code quality rules
- Past bugs and lessons

## License

MIT
