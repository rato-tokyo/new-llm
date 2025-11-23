# New-LLM: Context Vector Fixed-Point Property

A novel language model architecture based on the hypothesis that context vectors converge to fixed points with high dimensional diversity.

## Core Concept: CVFP (Context Vector Fixed-Point Property)

New-LLM explores the idea that meaningful context representations emerge through iterative refinement to fixed points, rather than traditional recurrent or transformer-based approaches.

## Features

- **Two-Phase Training**: Separate fixed-point learning and token prediction
- **High Dimensional Diversity**: Achieves 80%+ Effective Rank using LayerNorm + fixed dimension assignment
- **Diversity Regularization**: LayerNorm prevents value explosion, fixed dimension assignment forces diversity
- **Clean Architecture**: Object-oriented design with CVFPLayer encapsulation
- **Flexible Data Loading**: Supports UltraChat, text files, and custom datasets

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Training

```bash
# Quick test with 10 tokens (for development)
python3 tests/test_refactored.py

# Full training with configuration
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
├── train.py              # Main training script
├── config.py             # Configuration
├── CLAUDE.md             # Design guidelines and architecture decisions
├── CONTEXT.md            # Development history and insights
├── README.md             # This file
├── src/
│   ├── models/
│   │   ├── cvfp/
│   │   │   ├── __init__.py        # CVFP module exports
│   │   │   ├── layer.py           # CVFPLayer (basic unit)
│   │   │   └── block.py           # CVFPBlock (multi-layer)
│   │   └── new_llm_residual.py    # Main model architecture
│   ├── training/
│   │   ├── phase1.py              # Fixed-point learning
│   │   └── phase2.py              # Token prediction
│   ├── data/
│   │   └── loader.py              # Data loading utilities
│   └── evaluation/
│       └── metrics.py             # Analysis and metrics
├── tests/
│   └── test_refactored.py         # Quick development test (10 tokens)
├── scripts/
│   └── (experimental scripts)     # Future experiments and utilities
└── data/
    ├── example_train.txt          # Example training data
    └── example_val.txt            # Example validation data
```

## Architecture Highlights

### Diversity Regularization: LayerNorm + Fixed Dimension Assignment

Our breakthrough approach combines two complementary techniques:

**1. LayerNorm (Value Explosion Prevention)**
```python
# Prevents residual connection value explosion
if layernorm_mix > 0:
    new_context = (1 - mix) * new_context + mix * layer_norm(new_context)
```

**2. Fixed Dimension Assignment (Diversity Enforcement)**
```python
# Each token assigned to specific dimensions via hash
token_hash = hash(token_idx) % context_dim
assigned_dims = [(token_hash + i) % context_dim for i in range(dims_per_token)]
```

Benefits:
- **High Effective Rank**: Achieves 12.84/16 (80.3%) on training data
- **Stable Training**: No value explosion (norms stay controlled)
- **Forced Diversity**: Each token uses different dimension subsets
- **Simple & Effective**: No complex covariance or orthogonality constraints

### Two-Phase Training

**Phase 1: Fixed-Point Learning with Diversity Regularization**
- Contexts converge through iterative refinement
- LayerNorm prevents value explosion in residual connections
- Fixed dimension assignment forces high dimensional diversity
- Gradient clipping ensures training stability
- Early stopping based on convergence rate (95% of tokens)

**Phase 2: Token Prediction** (Optional)
- Standard next-token prediction
- Uses fixed contexts from Phase 1
- Optional context freezing

## Development Guidelines

See `CLAUDE.md` for:
- Design principles and philosophy
- Token-wise vs batch normalization rationale
- Object-oriented architecture patterns
- Code quality standards

## Current Status

**Recent Breakthrough (2025-11-23):**
- ✅ **80.3% Effective Rank achieved** using LayerNorm + fixed dimension assignment
- ✅ Stable training with no value explosion
- ✅ Validation data contamination issue identified and fixed
- ✅ Unified architecture (removed obsolete EMA/covariance/contrastive methods)

**Working:**
- ✅ High dimensional diversity (Effective Rank: 12.84/16 = 80.3%)
- ✅ Clean CVFPLayer architecture with LayerNorm
- ✅ Fixed dimension assignment for diversity enforcement
- ✅ Two-phase training pipeline
- ✅ Flexible data loading
- ✅ Gradient clipping for stability

**Next Steps:**
- 🎯 Scale to larger datasets (UltraChat)
- 🎯 Phase 2 token prediction evaluation
- 🎯 Perplexity and generation quality assessment

## License

MIT
