# New-LLM: Context Vector Fixed-Point Property

A novel language model architecture based on the hypothesis that context vectors converge to fixed points with high dimensional diversity.

## Core Concept: CVFP (Context Vector Fixed-Point Property)

New-LLM explores the idea that meaningful context representations emerge through iterative refinement to fixed points, rather than traditional recurrent or transformer-based approaches.

## Features

- **Two-Phase Training**: Separate fixed-point learning and token prediction
- **Phase 2 Cache Reuse**: Pass cache from Phase 1 to Phase 2, saving 627s (40% faster)
- **Parallel Processing**: **23x speedup** (265s → 11s) with parallel batch processing
- **High Effective Rank**: Achieves **55.9% (val) / ~60% (train) Effective Rank** with parallel optimization
- **Auto Batch Size**: GPU memory-based batch size calculation with OOM prevention
- **Optimized Loss Weight**: `dist_reg_weight = 0.9` compensates information delay with diversity enhancement
- **Diversity Regularization**: Global mean-based tracking for parallel processing
- **Function-Based Architecture**: Clean, efficient implementation in [src/trainers/phase1/memory.py](src/trainers/phase1/memory.py)
- **Flexible Data Loading**: Supports UltraChat, text files, and custom datasets
- **Full Reproducibility**: Fixed random seed (42) for deterministic training
- **GPU-Ready**: Optimized for Colab (22GB VRAM)

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Training

```bash
# Quick test (100 tokens)
python3 train.py --test

# Standard test with fixed train/val data (6400 train + 1280 val tokens)
python3 test.py

# Full training with configuration (skips Phase 1 if checkpoint exists)
python3 train.py
```

### Configuration

Edit `config.py` to adjust:
- Model architecture (layers, dimensions)
- Training parameters (learning rates, iterations)
- Data sources and preprocessing
- Distribution regularization settings

## Project Structure

```
new-llm/
├── train.py                       # Main training script
├── test.py                        # Standard test script
├── config.py                      # Configuration
├── CLAUDE.md                      # Design guidelines and architecture decisions
├── README.md                      # This file
├── src/
│   ├── models/
│   │   └── new_llm_residual.py    # Main model architecture
│   ├── trainers/
│   │   ├── phase1/
│   │   │   ├── base.py            # Phase 1 abstract base class
│   │   │   └── memory.py          # Memory-based Phase 1 trainer
│   │   └── phase2.py              # Phase 2: Token prediction
│   ├── data/
│   │   └── loader.py              # Data loading utilities
│   ├── utils/
│   │   └── memory.py              # GPU memory management
│   └── evaluation/
│       ├── metrics.py             # Analysis and metrics
│       └── diagnostics.py         # Identity mapping check
├── scripts/
│   ├── unified_scaling_experiment.py  # Scaling law experiments
│   └── create_val_from_train.py   # Generate validation data
├── data/
│   └── ultrachat_*samples_val.txt # Validation data files
└── importants/
    └── *.md                       # Experimental reports
```

## Architecture Highlights

### Parallel Processing with Diversity Optimization

Our implementation achieves **23x speedup** through parallel batch processing while maintaining high diversity:

**Implementation in phase1_train() (src/trainers/phase1.py):**
```python
def compute_diversity_loss(contexts):
    """
    多様性損失: 全トークンの平均からの偏差（負の損失で最大化）

    Args:
        contexts: 現在のコンテキスト [num_tokens, context_dim]

    Returns:
        diversity_loss: 多様性損失（スカラー）
    """
    context_mean = contexts.mean(dim=0)  # [context_dim]
    deviation = contexts - context_mean  # [num_tokens, context_dim]
    diversity_loss = -torch.norm(deviation, p=2) / len(contexts)
    return diversity_loss

# Combined loss with parallel optimization
total_loss = (1 - dist_reg_weight) * cvfp_loss + dist_reg_weight * diversity_loss
```

**Parallel Processing Design:**
- **Iteration 0**: Sequential processing (establishes fixed-point target)
- **Iteration 1+**: Parallel batch processing (uses previous iteration's contexts)
- **1-token shift**: Token i uses previous_contexts[i-1] from prior iteration
- **Information delay**: Compensated by `dist_reg_weight = 0.9` (90% diversity)

**Key Benefits:**
- **23x speedup**: 265s → 11s (vs sequential version)
- **High Effective Rank**: 55.9% (val) / ~60% (train) with parallel optimization
- **Diversity-first optimization**: `dist_reg_weight = 0.9` compensates information delay
- **Stable training**: Gradient clipping, deterministic results

### Two-Phase Training

**Phase 1: Parallel Fixed-Point Learning**
- **Parallel batch processing**: 23x speedup (265s → 11s)
- **Iteration 0**: Sequential processing to establish fixed-point target
- **Iteration 1+**: Parallel processing with context propagation
- **Global mean-based diversity**: Enforces high dimensional spread (55.9% Effective Rank)
- **Diversity-first optimization**: `dist_reg_weight = 0.9` (90% diversity, 10% CVFP)
- Gradient clipping ensures training stability
- Early stopping based on convergence rate (95% of tokens)

**Phase 2: Token Prediction**
- Context propagation across tokens (matches Phase 1 behavior)
- Prediction from concatenated context + token embeddings (both utilized)
- Context provides文脈information, token_embed provides local representation
- Next-token prediction with CrossEntropyLoss
- Full model fine-tuning with small learning rate (0.002)
- CVFP layers remain trainable for end-to-end optimization

## Development Guidelines

See `CLAUDE.md` for:
- Design principles and architecture decisions
- Critical bug fixes and lessons learned
- Mandatory numerical reporting rules
- Code quality standards

## Current Status

**Recent Achievements (2025-11-29):**
- ✅ **Phase 2 cache reuse**: Pass cache from Phase 1, skip 627s rebuild (40% faster)
- ✅ **Auto batch size**: GPU memory-based calculation with OOM prevention
- ✅ **Memory optimization**: Unified memory management in `src/utils/memory.py`
- ✅ **Parallel processing**: 23x speedup (265s → 11s)
- ✅ **Scaling experiments**: Confirmed α=-0.29 to -0.41 scaling law

**Recent Achievements (2025-11-27):**
- ✅ **Embedding freeze**: 71.9% PPL reduction with frozen GPT-2 embeddings
- ✅ **Weight tying**: 42% parameter reduction (91M → 53M)
- ✅ **Equal decrease design**: `token_input_all_layers=False` for parameter efficiency

**Design Decisions:**
- **Phase 2 cache reuse**: `return_all_layers=True` in Phase 1, pass to Phase 2
- **dist_reg_weight = 0.8**: Diversity optimization for effective rank
- **Parallel processing**: Iteration 0 sequential + Iteration 1+ parallel
- **Validation data**: Must be subset of training data (auto_split forbidden)

**Working:**
- ✅ Phase 2 cache reuse (skip 627s rebuild)
- ✅ Auto batch size with OOM prevention
- ✅ Parallel batch processing (23x speedup)
- ✅ High Effective Rank (55.9% val / ~60% train)
- ✅ Two-phase training pipeline
- ✅ GPT-2 pre-trained embeddings (768-dim, frozen in Phase 2)
- ✅ Weight tying (embedding = output head)
- ✅ Deterministic training (seed=42)

**Next Steps:**
- 🎯 Test cache reuse on Colab
- 🎯 Scale to 1000+ samples
- 🎯 Improve scaling efficiency (target α < -0.5)

## License

MIT
