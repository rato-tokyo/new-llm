# Context-Pythia: KV Cache Compression for Pythia-70M

Context-based KV cache compression achieving **50% memory reduction** while maintaining model quality.

## Overview

Context-Pythia compresses Pythia-70M's KV cache from 512-dim to 256-dim using a learned ContextBlock, reducing memory usage by 50% during inference.

```
Context-Pythia Architecture:
  Token Embedding (512-dim) ← Pythia-70M
       ↓
  ContextBlock: 512 → 256 (50% compression)
       ↓
  Layer 0-5: All use context (256-dim) as input
       ↓
  Output Head (vocab_size=50304)
```

## Key Features

- **50% KV Cache Reduction**: context_dim (256) vs hidden_size (512)
- **OACD Algorithm**: Origin-Anchored Centroid Dispersion for Phase 1 diversity learning
- **Two-Phase Training**: Phase 1 (OACD) → Phase 2 (Cross-entropy)
- **Pythia-70M Compatible**: Same tokenizer and data format

## Quick Start

### Installation

```bash
pip install torch transformers datasets
```

### Phase 1: ContextBlock Training (OACD)

```bash
# Train ContextBlock with OACD loss
python3 scripts/train_phase1_pythia.py --tokens 100000
```

Expected output:
```
Phase 1 Training:
  Iter  1: random init
  Iter  2: loss=5.37, conv=0%
  ...
  Iter 18: loss=1.26, conv=92%
  → Early stop: conv 92% >= 90%
```

### Phase 2: Comparison Experiment

```bash
# Compare Pythia-70M (baseline) vs Context-Pythia (50% KV compression)
python3 scripts/experiment_pythia_comparison.py --samples 10000 --epochs 10
```

Expected output:
```
| Model | Best PPL | KV Cache | Reduction |
|-------|----------|----------|-----------|
| Pythia-70M | XXX.X | 96.0 KB | - |
| Context-Pythia | XXX.X | 48.0 KB | 50% |
```

## Architecture Details

### Pythia-70M Specifications

| Component | Value |
|-----------|-------|
| Hidden Size | 512 |
| Layers | 6 |
| Attention Heads | 8 |
| Intermediate Size | 2048 |
| Position Encoding | Rotary (RoPE, 25%) |
| Vocab Size | 50,304 |

### Context-Pythia Modifications

| Component | Original | Context-Pythia |
|-----------|----------|----------------|
| KV dim | 512 | 256 |
| KV Cache | 100% | 50% |
| Attention Input | hidden_states | context |

### Training Pipeline

```
Phase 1: OACD (ContextBlock diversity learning)
  ├─ Train ContextBlock only
  ├─ OACD loss for diverse context vectors
  └─ Target: 90%+ convergence rate
       ↓
Phase 2: Full Training (ContextBlock frozen)
  ├─ Freeze ContextBlock
  ├─ Train Transformer Layers + Output Head
  └─ Cross-entropy loss
```

## OACD Algorithm

Origin-Anchored Centroid Dispersion for learning diverse context representations:

```python
def oacd_loss(contexts, centroid_weight=0.1):
    context_mean = contexts.mean(dim=0)
    deviation = contexts - context_mean

    # Term 1: Maximize dispersion from centroid
    dispersion_loss = -torch.norm(deviation, p=2) / len(contexts)

    # Term 2: Anchor centroid to origin
    centroid_loss = torch.norm(context_mean, p=2) ** 2

    return dispersion_loss + centroid_weight * centroid_loss
```

## Project Structure

```
new-llm/
├── config/
│   ├── phase1.py              # Phase 1 settings
│   └── pythia.py              # Pythia model config
├── scripts/
│   ├── train_phase1_pythia.py         # Phase 1 training
│   └── experiment_pythia_comparison.py # Phase 2 comparison
├── src/
│   ├── models/
│   │   ├── pythia.py          # Pythia-70M baseline
│   │   ├── context_pythia.py  # Context-Pythia (ours)
│   │   ├── blocks.py          # ContextBlock
│   │   └── layers.py          # ContextLayer
│   ├── losses/
│   │   └── diversity.py       # OACD algorithm
│   └── utils/
│       └── data_pythia.py     # Pile data loader
├── CLAUDE.md                  # Development guidelines
└── README.md
```

## Configuration

### Phase 1 Settings (config/phase1.py)

```python
max_iterations = 100         # Max training iterations
convergence_threshold = 0.03 # MSE threshold
learning_rate = 0.003        # Learning rate
batch_size = 5000            # Batch size
gradient_clip = 2.0          # Gradient clipping
context_noise = 0.05         # Gaussian noise for convergence
early_stopping_threshold = 0.9  # Stop at 90% convergence
```

### Pythia Settings (config/pythia.py)

```python
vocab_size = 50304
hidden_size = 512
context_dim = 256            # 50% compression
num_layers = 6
num_attention_heads = 8
```

## Development

See `CLAUDE.md` for detailed guidelines including:
- Phase 1 specifications (DO NOT DELETE)
- ContextBlock implementation details
- Weight initialization (normal_, NOT Xavier)
- Memory management patterns

## License

MIT
