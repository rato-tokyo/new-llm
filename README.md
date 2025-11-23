# New-LLM: Context Vector Fixed-Point Property

A novel language model architecture based on the hypothesis that context vectors converge to fixed points.

## Core Concept: CVFP (Context Vector Fixed-Point Property)

New-LLM explores the idea that meaningful context representations emerge through iterative refinement to fixed points, rather than traditional recurrent or transformer-based approaches.

## Features

- **Two-Phase Training**: Separate fixed-point learning and token prediction
- **Distribution Regularization**: Token-wise normalization using Exponential Moving Average (EMA)
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
│   │   ├── layers.py              # CVFPLayer and CVFPBlock
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

### CVFPLayer (Token-wise Normalization)

Unlike traditional batch normalization, our approach uses Exponential Moving Average (EMA) to track statistics per token:

```python
# Running statistics updated automatically during forward pass
running_mean = 0.99 * running_mean + 0.01 * current_mean
running_var = 0.99 * running_var + 0.01 * current_var
```

Benefits:
- Prevents trivial identity mapping solutions
- Works with any sequence length
- Theoretically correct for sequential processing
- Better gradient flow

### Two-Phase Training

**Phase 1: Fixed-Point Learning**
- Contexts converge through iterative refinement
- Distribution regularization ensures N(0,1) per dimension
- Early stopping based on convergence rate

**Phase 2: Token Prediction**
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

**Working:**
- ✅ Refactored CVFPLayer architecture
- ✅ Token-wise distribution regularization
- ✅ Two-phase training pipeline
- ✅ Flexible data loading

**Under Investigation:**
- ⚠️ Identity mapping tendency (model preserves input too much)
- ⚠️ Rapid convergence (2 iterations) - may indicate trivial solutions
- 🔬 CVFP loss function design

## License

MIT
