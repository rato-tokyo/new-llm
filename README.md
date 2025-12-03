# Context-Pythia: 50% KV Cache Reduction

A research project to reduce KV cache memory by 50% through context-based dimension compression, while maintaining model performance.

## 🎯 Goal

Replace Pythia-70M's attention input with compressed context vectors (512→256 dim), achieving **50% KV cache reduction**.

## Architecture

```
Context-Pythia:
  Token Embedding (512-dim)
       ↓
  ContextBlock: 512 → 256 (compression)
       ↓
  Layer 0-5: All use context (256-dim) as input
       ↓
  Output Head (vocab_size)

KV Cache: 256-dim × seq_len × 6 layers
         (vs. 512-dim × seq_len × 6 in original)
```

## Key Innovation

Instead of storing full-dimensional KV pairs, we:
1. Compress token embeddings to context vectors via ContextBlock
2. Use compressed context as attention input for all layers
3. Maintain model capacity while reducing memory

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run comparison experiment (dev mode)
python3 scripts/experiment_pythia_comparison.py --dev

# Full experiment
python3 scripts/experiment_pythia_comparison.py --samples 10000
```

## Results

| Model | Params | KV Cache | Val PPL | Val Acc |
|-------|--------|----------|---------|---------|
| PythiaModel | 70M | 100% | TBD | TBD |
| ContextPythiaModel | 67M | **50%** | TBD | TBD |

## Project Structure

```
new-llm/
├── config/
│   └── pythia.py              # Configuration
├── scripts/
│   └── experiment_pythia_comparison.py
├── src/
│   ├── models/
│   │   ├── pythia.py          # Baseline (Pythia reproduction)
│   │   └── context_pythia.py  # Ours (50% KV reduction)
│   └── losses/
│       └── diversity.py       # OACD algorithm
└── CLAUDE.md                  # Development guidelines
```

## Training Pipeline

### Phase 1: Context Diversity Learning (OACD)

Train ContextBlock to produce diverse representations:

```python
def oacd_loss(contexts, centroid_weight=0.1):
    centroid = contexts.mean(dim=0)
    dispersion_loss = -torch.norm(contexts - centroid) / len(contexts)
    centroid_loss = torch.norm(centroid) ** 2
    return dispersion_loss + centroid_weight * centroid_loss
```

### Phase 2: Full Model Training

- Freeze ContextBlock
- Train all layers with cross-entropy loss

## Baseline: Pythia-70M

| Parameter | Value |
|-----------|-------|
| Layers | 6 |
| Hidden Size | 512 |
| Attention Heads | 8 |
| Parameters | 70M |
| Training Data | Pile (~300B tokens) |

## References

- [Pythia Paper](https://arxiv.org/abs/2304.01373)
- [EleutherAI/pythia-70m](https://huggingface.co/EleutherAI/pythia-70m)
- [DeepSeek MLA](https://arxiv.org/abs/2401.02954)

## License

MIT
