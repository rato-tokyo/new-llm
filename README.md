# New-LLM: Context Vector Fixed-Point Property

A novel language model architecture based on the hypothesis that context vectors converge to fixed points with high dimensional diversity.

## Core Concept: CVFP (Context Vector Fixed-Point Property)

New-LLM explores the idea that meaningful context representations emerge through iterative refinement to fixed points, rather than traditional recurrent or transformer-based approaches.

## Features

- **Two-Phase Training**: Separate fixed-point learning and token prediction
- **High Dimensional Diversity**: Achieves **89.7% (train) / 89.4% (val) Effective Rank** using LayerNorm + Per-Dimension Variance Tracking
- **Balanced Loss Weight**: `dist_reg_weight = 0.5` enables both diversity and CVFP learning
- **Diversity Regularization**: Per-dimension usage tracking with EMA-based weighting
- **Clean Architecture**: Object-oriented design with CVFPLayer encapsulation
- **Flexible Data Loading**: Supports UltraChat, text files, and custom datasets
- **Full Reproducibility**: Fixed random seed (42) for deterministic training
- **GPU-Ready**: 10-20x speedup available with CUDA

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
├── test.py                        # Standard test script (6400 train + 1280 val)
├── config.py                      # Configuration
├── CLAUDE.md                      # Design guidelines and architecture decisions
├── README.md                      # This file
├── src/
│   ├── models/
│   │   ├── cvfp/
│   │   │   ├── __init__.py        # CVFP module exports
│   │   │   ├── layer.py           # CVFPLayer (basic unit)
│   │   │   └── block.py           # CVFPBlock (multi-layer)
│   │   └── new_llm_residual.py    # Main model architecture
│   ├── training/
│   │   ├── phase1_trainer.py      # Fixed-point learning
│   │   └── phase2_trainer.py      # Token prediction
│   ├── data/
│   │   └── loader.py              # Data loading utilities
│   └── evaluation/
│       ├── metrics.py             # Analysis and metrics
│       └── diagnostics.py         # Identity mapping check
├── scripts/
│   └── create_val_from_train.py   # Generate validation data from training data
├── data/
│   ├── example_train.txt          # Training data (auto-generated)
│   └── example_val.txt            # Validation data (from training data)
└── docs/
    └── experiment_*.md            # Experimental reports
```

## Architecture Highlights

### Diversity Regularization: Per-Dimension Usage Tracking

Our breakthrough approach uses dimension usage statistics to enforce diversity:

**Implementation in Phase1Trainer:**
```python
# Each iteration starts fresh
dim_stats = torch.zeros(context_dim, device=device)

# For each token
for token_id in token_ids:
    # Calculate dimension weights (inverse of usage frequency)
    dim_weights = 1.0 / (dim_stats + 1.0)  # detached

    # Diversity loss: activate less-used dimensions
    diversity_loss = -(dim_weights * context.abs().squeeze(0)).mean()

    # Update usage statistics (no gradient)
    with torch.no_grad():
        dim_stats += context.abs().squeeze(0)

    # Combined loss
    total_loss = (1 - dist_reg_weight) * cvfp_loss + dist_reg_weight * diversity_loss
```

**Key Design:**
- **LayerNorm**: Prevents value explosion in residual connections (`layernorm_mix = 1.0`)
- **Dimension weights**: Detached (no gradient), guide optimization only
- **Context gradients**: Flow through for actual learning
- **Per-iteration reset**: Each iteration starts with zero statistics

Benefits:
- **High Effective Rank**: Achieves **89.7% (train) / 89.4% (val)** with `dist_reg_weight = 0.5`
- **Balanced learning**: 50% CVFP loss + 50% diversity loss
- **CVFP convergence**: Passes convergence check (final_diff < 0.001)
- **Stable training**: No value explosion, deterministic results

### Two-Phase Training

**Phase 1: Fixed-Point Learning with Diversity Regularization**
- Contexts converge through iterative refinement (carries context between iterations)
- LayerNorm prevents value explosion in residual connections (`layernorm_mix = 1.0`)
- **Per-dimension usage tracking** enforces high dimensional diversity (89.7%/89.4% Effective Rank)
- **Balanced loss weight** (`dist_reg_weight = 0.5`) enables both CVFP and diversity learning
- Gradient clipping ensures training stability
- Early stopping based on convergence rate (95% of tokens)

**Phase 2: Token Prediction**
- **Zero-vector initialization**: Each token starts from 0-vector (matches real inference)
- Next-token prediction with CrossEntropyLoss
- Full model fine-tuning with small learning rate (0.0001)
- CVFP layers remain trainable for end-to-end optimization

## Development Guidelines

See `CLAUDE.md` for:
- Design principles and architecture decisions
- Critical bug fixes and lessons learned
- Mandatory numerical reporting rules
- Code quality standards

## Current Status

**Recent Achievements (2025-11-24):**
- ✅ **Phase 1**: 89.7% (train) / 89.4% (val) Effective Rank with `dist_reg_weight = 0.5`
- ✅ **Balanced loss weight** enables both diversity and CVFP learning
- ✅ **CVFP convergence verified** (final_diff = 0.000745 < 0.001)
- ✅ **Phase 2 implementation** with zero-vector initialization
- ✅ **Full reproducibility** with fixed random seed (42)
- ✅ **Critical bug fix**: Context carryover between iterations (Phase 1)
- ✅ **Architecture fix**: CVFPBlock tuple handling in Phase 2

**Design Decisions:**
- **dist_reg_weight = 0.5**: Balances CVFP loss and diversity loss (50/50)
- **LayerNorm enabled**: Prevents value explosion (`layernorm_mix = 1.0`)
- **Validation data**: Must be subset of training data (auto_split forbidden)
- **Phase 2 initialization**: Zero-vector per token (matches real inference)

**Working:**
- ✅ High dimensional diversity (89.7% train / 89.4% val)
- ✅ Stable CVFP convergence (passes convergence check)
- ✅ Two-phase training pipeline
- ✅ Phase 1 skip functionality (checkpoint resume)
- ✅ Full model fine-tuning in Phase 2
- ✅ GPT-2 pre-trained embeddings (768-dim)
- ✅ Deterministic training (seed=42)

**In Progress:**
- 🔄 Phase 2 experiment running (full fine-tuning with zero-vector initialization)

**Next Steps:**
- 🎯 Evaluate Phase 2 performance (loss, perplexity, accuracy)
- 🎯 Scale to larger datasets (10k+ tokens)
- 🎯 GPU acceleration for faster training

## License

MIT
