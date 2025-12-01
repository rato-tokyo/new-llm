# New-LLM: Origin-Anchored Centroid Dispersion

A novel language model architecture using OACD (Origin-Anchored Centroid Dispersion) algorithm for learning diverse context representations.

## Core Concept: OACD Algorithm

New-LLM uses diversity-based learning where context vectors are trained to maximize dispersion while anchoring their centroid to the origin. This creates meaningful context representations through diversity optimization rather than traditional recurrent or transformer-based approaches.

## Features

- **Two-Phase Training**: Separate diversity learning (OACD) and token prediction
- **Shallow & Wide Architecture**: 3 layers, 1536 context_dim, 2 input tokens (best performance)
- **Best Scaling Law**: α = **-0.5402** (R² = 0.977), PPL 197.0, Acc 22.9%
- **Token Input All Layers**: `token_input_all_layers=True` is essential for performance
- **Parallel Cache Collection**: 51s → few seconds with batch processing (context similarity 99.7%)
- **Phase 2 Cache Reuse**: Pass cache from Phase 1 to Phase 2, saving 627s (40% faster)
- **Parallel Processing**: **23x speedup** (265s → 11s) with parallel batch processing
- **Auto Batch Size**: GPU memory-based batch size calculation with OOM prevention
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

### Scaling Experiments

```bash
# Standard scaling law experiment
python3 scripts/scaling_experiment.py --input-tokens 1 --layers 1 --context-dim 768

# 9-config matrix (input_tokens × layers)
python3 scripts/scaling_experiment.py --matrix

# Alpha progression analysis: measure how α changes with more data
# Generates sample sizes: [50, 100, 200, 400, 800]
# Window 1: [50-400] → α₁, Window 2: [100-800] → α₂
python3 scripts/scaling_experiment.py --alpha-scaling \
  --init-samples 50 --multiplier 2 --window-size 4 --num-windows 2
```

**Alpha Scaling Mode**: Measures how scaling efficiency (α) changes as data amount increases. Uses sliding window analysis to track α progression.

### Diversity Algorithm Experiments

```bash
# Phase 1 only: Compare diversity algorithms on Effective Rank
python3 scripts/diversity_algorithm_experiment.py -a MCDL ODCM SDL NUC -s 50 100

# Phase 1 + Phase 2: Full experiment with α analysis
# Uses 4 algorithms, samples=[50,100,200], context_dim=1000
python3 scripts/diversity_full_experiment.py
```

**Available Algorithms**:
- **MCDL**: Mean-Centered Dispersion Loss (fastest baseline)
- **ODCM**: Off-Diagonal Covariance Minimization (VICReg-style, recommended)
- **SDL**: Spectral Diversity Loss (direct ER maximization, highest ER)
- **NUC**: Nuclear Norm Maximization (high ER, high cost)

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
│   │   └── llm.py                 # Main model architecture (LLM)
│   ├── trainers/
│   │   ├── phase1/
│   │   │   ├── base.py            # Phase 1 abstract base class
│   │   │   └── memory.py          # Memory-based Phase 1 trainer
│   │   └── phase2.py              # Phase 2: Token prediction
│   ├── experiments/
│   │   ├── config.py              # Shared config classes (DataConfig, Phase1Config, Phase2Config)
│   │   └── runner.py              # ExperimentRunner
│   ├── losses/
│   │   └── diversity.py           # Diversity loss algorithms (MCDL, ODCM, SDL, NUC)
│   ├── providers/
│   │   └── data/
│   │       └── memory.py          # Memory-based data provider
│   ├── utils/
│   │   └── memory.py              # GPU memory management
│   └── evaluation/
│       ├── metrics.py             # Analysis and metrics
│       └── diagnostics.py         # Identity mapping check
├── scripts/
│   ├── scaling_experiment.py      # Scaling law experiments (with alpha progression)
│   ├── diversity_algorithm_experiment.py  # Phase 1 diversity algorithm comparison
│   ├── diversity_full_experiment.py       # Phase 1+2 with α analysis
│   └── create_val_from_train.py   # Generate validation data
├── data/
│   └── example_val.txt            # Validation data file (auto-generated)
└── importants/
    └── *.md                       # Experimental reports
```

## Architecture Highlights

### OACD Algorithm

Our implementation uses the Origin-Anchored Centroid Dispersion (OACD) algorithm:

**Implementation in src/losses/diversity.py:**
```python
def oacd_loss(contexts, centroid_weight=0.1):
    """
    OACD: Origin-Anchored Centroid Dispersion

    Term 1: 重心からの分散を最大化
    Term 2: 重心を原点に引き寄せる
    """
    centroid = contexts.mean(dim=0)
    deviations = contexts - centroid
    dispersion_loss = -torch.norm(deviations, p=2) / len(contexts)
    centroid_loss = torch.norm(centroid) ** 2
    return dispersion_loss + centroid_weight * centroid_loss
```

**Key Benefits:**
- **Stable convergence**: Origin anchoring provides consistent equilibrium
- **High Effective Rank**: 80%+ dimensional utilization
- **No complex logic**: Simple loss function without convergence checking
- **Parallel processing**: 23x speedup with batch processing

### Two-Phase Training

**Phase 1: Diversity Learning (OACD)**
- **Parallel batch processing**: 23x speedup (265s → 11s)
- **OACD algorithm**: Origin-anchored centroid dispersion
- **High diversity**: 80%+ Effective Rank through dispersion maximization
- Gradient clipping ensures training stability
- Early stopping based on validation Effective Rank

**Phase 2: Token Prediction**
- Context propagation across tokens (matches Phase 1 behavior)
- Prediction from concatenated context + token embeddings (both utilized)
- Context provides 文脈 information, token_embed provides local representation
- Next-token prediction with CrossEntropyLoss
- ContextBlock frozen, TokenBlock trained

## Development Guidelines

See `CLAUDE.md` for:
- Design principles and architecture decisions
- Critical bug fixes and lessons learned
- Mandatory numerical reporting rules
- Code quality standards

## Current Status

**Architecture Comparison Results (2025-11-29):**

| Config | Layers | context_dim | input_tokens | α | Best PPL | Best Acc |
|--------|--------|-------------|--------------|------|----------|----------|
| baseline | 6 | 768 | 1 | -0.4860 | 249.3 | 21.3% |
| input_tokens_2 | 6 | 768 | 2 | -0.4702 | 198.1 | 22.5% |
| context_dim_1152 | 6 | 1152 | 1 | -0.4988 | 246.9 | 21.4% |
| layers_9 | 9 | 768 | 1 | -0.4818 | 256.8 | 21.1% |
| **shallow_wide** | **3** | **1536** | **2** | **-0.5402** | **197.0** | **22.9%** |

**Key Discovery: Model benefits from width over depth**
- shallow_wide achieves best α (-0.5402) with only 3 layers
- Doubling context_dim (768→1536) + 2 input tokens is optimal
- 9 layers provides no benefit over 6 layers (unlike Transformers)

See [importants/experiment-results-20251129-architecture-comparison.md](importants/experiment-results-20251129-architecture-comparison.md) for full analysis.

**Recommended Configuration:**
```python
num_layers = 3
context_dim = 1536
num_input_tokens = 2
embed_dim = 768
```

**Working:**
- ✅ Shallow & wide architecture (3L/1536d/2tok)
- ✅ Best scaling law α = -0.5402
- ✅ Parallel cache collection (51s → few seconds)
- ✅ Phase 2 cache reuse (skip 627s rebuild)
- ✅ Auto batch size with OOM prevention
- ✅ Parallel batch processing (23x speedup)
- ✅ Two-phase training pipeline
- ✅ GPT-2 pre-trained embeddings (768-dim, frozen in Phase 2)
- ✅ Weight tying (embedding = output head)
- ✅ Deterministic training (seed=42)

**Current Research Focus (2025-12-01):**
- 🔬 OACD algorithm optimization
- 🔬 Diversity algorithm comparison (MCDL, ODCM, SDL, NUC)
- 🔬 α value comparison across algorithms

**Next Steps:**
- 🎯 Scale to 1000+ samples with shallow_wide config
- 🎯 Test even wider architectures (context_dim=2048+)
- 🎯 Explore num_input_tokens=3

## License

MIT
