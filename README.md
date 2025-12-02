# New-LLM: Cascade Context Architecture

A novel language model architecture using OACD (Origin-Anchored Centroid Dispersion) algorithm with cascade context concatenation for learning diverse context representations.

## Core Concept: Cascade Context

New-LLM uses a cascade context architecture where two ContextBlocks are trained sequentially, then their outputs are concatenated for token prediction. This provides the expressiveness of wider context dimensions while maintaining efficient training.

```
Phase 1A: ContextBlock A (cd=500) → context_a cache
Phase 1B: ContextBlock B (cd=500, input=context_a) → context_b cache
Phase 2:  TokenBlock (input=concat(context_a, context_b)=1000) → predictions
```

## Key Results

| Configuration | Val PPL | Val Acc | Notes |
|---------------|---------|---------|-------|
| **Cascade (500×2=1000)** | **111.9** | **25.6%** | Best performance |
| C1T1-500 | 127.2 | 24.7% | Standard single context |
| C2T2-500 | 132.2 | 24.4% | 2-layer (worse than 1-layer) |
| C1T1-1000 | 134.0 | 23.6% | Wide context (inefficient) |

**Key Discovery**: Cascade concatenation outperforms both multi-layer and wider single-context approaches.

## Features

- **Cascade Context Architecture**: Two ContextBlocks with cascade concatenation
- **1-Layer Fixed**: Each block is single layer (no multi-layer complexity)
- **OACD Algorithm**: Origin-Anchored Centroid Dispersion for diversity learning
- **Two-Phase Training**: Separate diversity learning and token prediction
- **GPT-2 Embeddings**: Frozen pretrained embeddings (768-dim)
- **Weight Tying**: Output head shares weights with embedding layer
- **GPU Optimized**: Efficient memory management for Colab (22GB VRAM)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running Experiments

```bash
# Local test (CPU, minimal samples)
python3 scripts/experiment_cascade_context.py -s 2

# Full experiment (GPU/Colab)
python3 scripts/experiment_cascade_context.py -s 2000

# Custom context dimension
python3 scripts/experiment_cascade_context.py -s 2000 -c 500
```

## Architecture

### 1-Layer Fixed Design (2025-12-02)

Based on experiments showing that multi-layer architectures provide no benefit:

- **ContextBlock**: 1 layer (Phase 1 training, Phase 2 frozen)
- **TokenBlock**: 1 layer (Phase 2 training)
- **Cascade Concatenation**: `concat(context_a, context_b)` for expressiveness

### OACD Algorithm

```python
def oacd_loss(contexts, centroid_weight=0.1):
    """Origin-Anchored Centroid Dispersion"""
    centroid = contexts.mean(dim=0)
    deviations = contexts - centroid
    dispersion_loss = -torch.norm(deviations, p=2) / len(contexts)
    centroid_loss = torch.norm(centroid) ** 2
    return dispersion_loss + centroid_weight * centroid_loss
```

**Benefits**:
- Stable convergence with origin anchoring
- High Effective Rank (80%+)
- Simple loss function

### Training Pipeline

**Phase 1A**: Train ContextBlock A on full data
- Input: zero-initialized context + token embeddings
- Output: context_a (cached for Phase 1B and Phase 2)

**Phase 1B**: Train ContextBlock B on full data
- Input: context_a (fixed) + token embeddings
- Output: context_b (cached for Phase 2)

**Phase 2**: Train TokenBlock
- Input: `concat(context_a, context_b)` + token embeddings
- Both ContextBlocks frozen
- Cross-entropy loss for next-token prediction

## Project Structure

```
new-llm/
├── scripts/
│   └── experiment_cascade_context.py  # Main experiment script
├── src/
│   ├── models/
│   │   ├── llm.py                     # Base LLM model
│   │   ├── blocks.py                  # ContextBlock, TokenBlock (1-layer)
│   │   └── layers.py                  # ContextLayer, TokenLayer
│   ├── trainers/
│   │   └── phase1/
│   │       ├── base.py                # Phase 1 abstract base
│   │       └── memory.py              # Memory-based Phase 1 trainer
│   ├── losses/
│   │   └── diversity.py               # OACD algorithm
│   ├── providers/
│   │   └── data/
│   │       └── memory.py              # Data provider
│   └── utils/
│       ├── device.py                  # GPU utilities
│       ├── initialization.py          # Weight init, parameter counting
│       └── memory.py                  # Memory management
├── config/
│   ├── base.py                        # Base configuration
│   ├── phase1.py                      # Phase 1 config
│   └── phase2.py                      # Phase 2 config
├── CLAUDE.md                          # Development guidelines
└── README.md                          # This file
```

## Configuration

Default configuration in `config/base.py`:

```python
embed_dim = 768           # GPT-2 embedding dimension
context_dim = 500         # Context dimension per block
vocab_size = 50257        # GPT-2 vocabulary
num_input_tokens = 1      # Input tokens per step
```

## Development Guidelines

See `CLAUDE.md` for:
- Cascade context architecture details
- OACD algorithm specification
- 1-layer fixed architecture rationale
- Code quality standards
- GPU/CPU memory management

## Current Status (2025-12-02)

**Working**:
- Cascade context architecture (best performance)
- 1-layer fixed design (simplified codebase)
- OACD algorithm with high Effective Rank
- GPU-optimized training pipeline
- Type-safe configuration with Protocol

**Architecture Decision**:
- Multi-layer logic removed (C2T2 performed worse than C1T1)
- Cascade concatenation provides sufficient expressiveness
- Code significantly simplified

## License

MIT
