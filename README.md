# New-LLM: Context-KV Attention for Efficient LLM Inference

A research project to improve pretrained LLMs (Pythia-70M) with Context-KV Attention, achieving **50% KV cache memory reduction** while maintaining or improving performance.

## 🎯 Project Goal

**Replace early attention layers in Pythia-70M with Context-KV Attention to reduce KV cache memory usage by 50%.**

### Target Model: Pythia-70M

| Parameter | Value |
|-----------|-------|
| Layers | 6 |
| Hidden Size | 512 |
| Attention Heads | 8 |
| Total Parameters | 70M |

### Approach

1. **Replace Layer 0** of Pythia with Context-KV Attention
2. **Keep Layers 1-5** as original Pythia attention
3. **Two-phase training**:
   - Phase 1: Train ContextBlock (OACD algorithm for diversity)
   - Phase 2: Fine-tune entire model

### Expected Benefits

- **50% KV cache reduction** (conservative target)
- **Maintained or improved PPL** on evaluation benchmarks
- **Minimal architectural changes** to pretrained model

## Core Concept: Context-KV Attention

Instead of storing all token KV pairs, we compress context into fixed-size vectors:

```
Standard Attention (Layer 0):
  KV Cache = [kv₀, kv₁, kv₂, ..., kv₁₀₂₃]  // 1024 KV pairs

Context-KV Attention (Layer 0):
  KV Cache = [ctx₀, ctx₃₂, ctx₆₄, ...]      // ~32 context vectors
  → 32x compression at interval=32
```

### How It Works

```
Position 350, interval=32, max_contexts=32:
  Context KV = [ctx[350], ctx[318], ctx[286], ..., ctx[30]]
               ↑current   ↑-32      ↑-64

  Query: from current token embedding
  Key/Value: projected from context vectors
```

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Pythia-70M + Context-KV                  │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Token Embedding (Pythia pretrained, 512-dim)        │ │
│  └──────────────────────────────────────────────────────┘ │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Layer 0: Context-KV Attention (REPLACED)            │ │
│  │  - ContextBlock: learns compressed representations    │ │
│  │  - Context-KV Attention: uses context as KV cache    │ │
│  │  - context_dim=256 (compression ratio ~2x)           │ │
│  └──────────────────────────────────────────────────────┘ │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Layers 1-5: Original Pythia Attention (PRESERVED)   │ │
│  │  - Standard self-attention                           │ │
│  │  - Pretrained weights maintained                     │ │
│  └──────────────────────────────────────────────────────┘ │
│                          ↓                                  │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  Output Head (Pythia pretrained)                     │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

## Training Pipeline

### Phase 1: Context Diversity Learning (OACD)

Train only the ContextBlock to produce diverse context representations:

```python
def oacd_loss(contexts, centroid_weight=0.1):
    """Origin-Anchored Centroid Dispersion"""
    centroid = contexts.mean(dim=0)
    dispersion_loss = -torch.norm(contexts - centroid) / len(contexts)
    centroid_loss = torch.norm(centroid) ** 2
    return dispersion_loss + centroid_weight * centroid_loss
```

### Phase 2: Full Model Fine-tuning

- Freeze ContextBlock
- Fine-tune Context-KV Attention layer + remaining Pythia layers
- Cross-entropy loss for next-token prediction

## Evaluation

### Primary Metrics

| Metric | Purpose |
|--------|---------|
| **PPL (Perplexity)** | Language modeling quality |
| **LAMBADA Accuracy** | Long-range dependency (final word prediction) |
| **KV Cache Memory** | Actual memory usage measurement |

### Comparison

```
Baseline: Pythia-70M (original)
Ours:     Pythia-70M + Context-KV (Layer 0 replaced)

Evaluate on:
- WikiText-2 PPL
- Pile test set PPL
- LAMBADA accuracy
- torch.cuda.max_memory_allocated()
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Development Experiments (Limited Data)

```bash
# Quick test with minimal samples
python3 scripts/experiment_context_kv.py -s 100

# Medium-scale test
python3 scripts/experiment_context_kv.py -s 1600
```

### Full Experiments (Pile Dataset)

```bash
# Full training (requires significant compute)
python3 scripts/experiment_pythia_context_kv.py  # TBD
```

## Project Structure

```
new-llm/
├── scripts/
│   ├── experiment_context_kv.py      # Current Context-KV experiments
│   └── experiment_pythia_context_kv.py  # Pythia integration (TBD)
├── src/
│   ├── models/
│   │   ├── context_kv.py             # ContextKVAttentionLLM
│   │   ├── blocks.py                 # ContextBlock (1-layer)
│   │   └── layers.py                 # ContextLayer
│   ├── trainers/
│   │   └── phase1/
│   │       └── memory.py             # Phase 1 trainer (OACD)
│   ├── losses/
│   │   └── diversity.py              # OACD algorithm
│   └── utils/
├── config/
│   ├── base.py                       # Model architecture config
│   ├── phase1.py                     # Phase 1 training config
│   └── phase2.py                     # Phase 2 training config
├── CLAUDE.md                         # Development guidelines
└── README.md                         # This file
```

## Configuration

Key parameters in `config/base.py`:

```python
# Context-KV Attention
context_dim = 256           # Context vector dimension
context_interval = 32       # Interval between contexts
max_contexts = 32           # Maximum contexts (context window)
num_heads = 8               # Attention heads

# Data (development)
num_samples = 1600          # Limited samples for development
```

## Current Status (2025-12-03)

### Completed
- ✅ Context-KV Attention implementation
- ✅ OACD algorithm for Phase 1
- ✅ Two-phase training pipeline
- ✅ Config-driven architecture (no hardcoding)

### In Progress
- 🔄 Pythia-70M integration
- 🔄 Pile dataset support
- 🔄 LAMBADA evaluation

### Planned
- ⬚ Layer 0 replacement experiments
- ⬚ Memory usage benchmarking
- ⬚ Comparison with original Pythia-70M

## References

- [Pythia: A Suite for Analyzing Large Language Models](https://arxiv.org/abs/2304.01373)
- [DeepSeek MLA (Multi-Head Latent Attention)](https://arxiv.org/abs/2401.02954)
- [EleutherAI/pythia-70m](https://huggingface.co/EleutherAI/pythia-70m)

## License

MIT
