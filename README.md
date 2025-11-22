# New-LLM: Context Vector Fixed-Point Property Language Model

An experimental language model that replaces attention mechanisms with **context vector fixed-point property (CVFP)** for **O(1) memory usage**.

**Powered by CVFP Training** - a two-phase learning approach where context vectors converge to fixed points before learning token prediction.

---

## 🚀 Quick Start

### Local Training

```bash
# Install dependencies
pip install torch tokenizers datasets tqdm

# Quick test - 16-dim model (次元崩壊テスト)
python3 tests/phase2_experiments/test_residual.py \
    --context-dim 16 \
    --embed-dim 16 \
    --hidden-dim 32 \
    --num-samples 10

# Standard model - 256-dim, 4 layers (default)
python3 tests/phase2_experiments/test_residual.py

# Custom configuration - 3 layers
python3 tests/phase2_experiments/test_residual.py \
    --context-dim 128 \
    --num-layers 3 \
    --num-samples 50 \
    --dist-reg-weight 0.2
```

---

## 🎯 What Makes New-LLM Different?

### 1. O(1) Memory Usage

| Architecture | Memory Complexity | Max Sequence Length |
|--------------|-------------------|---------------------|
| **Transformer** | O(n²) | Limited by memory |
| **New-LLM** | **O(1)** | **Unlimited** ✨ |

**No attention mechanism, no positional embeddings** - position information emerges naturally from sequential processing (like RNN/LSTM).

### 2. Context Vector Fixed-Point Property (CVFP)

**Core Hypothesis**: After sufficient iterations, context vectors converge to fixed points:
- `context(n) ≈ context(n+1)` for large n
- Enables stable, long-term information compression
- O(1) memory regardless of sequence length

### 3. Two-Phase Training

**Phase 1: Fixed-Point Context Learning**
- Learn context generation layers
- Iterate until contexts converge to fixed points
- **Critical**: Must achieve high Effective Rank (no dimension collapse)

**Phase 2: Token Prediction**
- Use fixed-point contexts from Phase 1
- Train token output layer only
- Standard cross-entropy loss

### 4. Distribution Regularization

**Problem**: Context vectors can collapse to low dimensions (Effective Rank 1/16 = 6%)

**Solution**: Constrain each dimension (across all tokens) to follow N(0,1)
```python
dim_mean = all_contexts.mean(dim=0)  # [context_dim]
dim_var = all_contexts.var(dim=0)    # [context_dim]
dist_loss = (dim_mean ** 2).mean() + ((dim_var - 1.0) ** 2).mean()
total_loss = 0.8 * cvfp_loss + 0.2 * dist_loss
```

**Result**: Effective Rank improved from 1.01/16 (6%) to 7.54/16 (47%) - **7.5x improvement**

---

## 🏗️ Architecture Overview

### Residual Standard Architecture

```python
# Simplified pseudocode - Phase 1 (Fixed-Point Learning)
context = torch.zeros(context_dim)

for iteration in range(max_iterations):
    for token in sequence:
        # 1. Update context
        context_new = model._update_context_one_step(token_embed, context)

        # 2. Learn to match previous iteration's context (fixed-point)
        loss = mse_loss(context_new, fixed_contexts[token_idx])
        loss.backward()
        optimizer.step()

        # 3. Pass context to next token (but cut gradient)
        context = context_new.detach()
        context.requires_grad = True

    # Check convergence
    if converged_ratio > 0.95:
        break

# Phase 2: Token Prediction (using fixed contexts)
for epoch in range(epochs):
    logits = model.token_output(context)
    loss = cross_entropy(logits, target)
```

**Key Innovation**:
- Fixed-size context vector (16-256 dims) instead of O(n²) attention
- Distribution regularization prevents dimension collapse
- Two-phase training ensures stable fixed points before token prediction

---

## 📊 Command-Line Arguments

All settings can be controlled via command-line arguments:

### Model Architecture
```
--context-dim INT       Context vector dimension (default: 256)
--embed-dim INT         Token embedding dimension (default: 256)
--hidden-dim INT        Hidden layer dimension (default: 512)
--num-layers INT        Number of single-layer blocks (default: 4)
                        Creates [1,1,1,1] for 4, [1,1,1] for 3, etc.
```

### Phase 1 Settings
```
--phase1-max-iter INT        Max iterations (default: 10)
--phase1-lr-warmup FLOAT     Warmup LR (default: 0.002)
--phase1-lr-medium FLOAT     Medium LR (default: 0.0005)
--phase1-lr-finetune FLOAT   Finetune LR (default: 0.0001)
```

### Distribution Regularization
```
--dist-reg-weight FLOAT  Regularization weight (default: 0.2)
--no-dist-reg            Disable distribution regularization
```

### Phase 2 Settings
```
--phase2-lr FLOAT         Learning rate (default: 0.0001)
--phase2-epochs INT       Epochs (default: 10)
--phase2-batch-size INT   Batch size (default: 32)
```

### Data Settings
```
--num-samples INT         Number of training samples (default: 10)
--train-val-split FLOAT   Train/Val split ratio (default: 0.8)
```

### Other
```
--device STR           Device (cpu/cuda, default: cpu)
--skip-phase2          Skip Phase 2 (only run Phase 1)
--freeze-context       Freeze context in Phase 2
```

---

## 📊 Project Structure

```
new-llm/
├── config.py                          # Default configuration
├── src/
│   ├── models/
│   │   └── new_llm_residual.py        # Residual Standard architecture
│   └── utils/
│       ├── early_stopping.py          # Phase 1/2 early stopping
│       └── cache_manager.py           # Fixed-context cache
├── tests/
│   ├── phase1_experiments/            # Phase 1 experiments
│   └── phase2_experiments/            # Phase 2 experiments
│       └── test_residual.py           # Main experiment script
├── docs/
│   └── experiments/                   # Experiment result reports
├── cache/                             # Cache (DO NOT DELETE)
│   ├── tokenizer/                     # Tokenizer cache
│   └── manual_val_tokens.pt           # Manual validation data
├── CLAUDE.md                          # Development guidelines
└── README.md                          # This file
```

---

## 📖 Training Output Example

```
======================================================================
Residual Standard Architecture Test
======================================================================

📋 Configuration: Command-line arguments
   Edit defaults in: config.py (project root)

🏗️  Model Architecture:
   Layer structure: [1, 1, 1, 1]
   Context dim: 16
   Embed dim: 16
   Hidden dim: 32

⚙️  Phase 1 Settings (CVFP):
   Max iterations: 10
   Convergence threshold: 0.02
   Min converged ratio: 0.95
   LR schedule: 0.001 → 0.001 → 0.001
   Distribution Reg: weight=0.2 (80% CVFP, 20% Dist)

📊 Data Settings:
   Num samples: 10
   Train/Val split: 0.8
   Device: cpu

======================================================================
PHASE 1: FIXED-POINT CONTEXT LEARNING
======================================================================

  Iteration 1: Loss=4.110549 (CVFP=4.110549, Dist=0.000000), Converged=0.0%
  Iteration 2: Loss=0.103548 (CVFP=0.000000, Dist=0.517741), Converged=100.0%

✅ Phase 1 converged in 2 iterations (100.0% converged)

======================================================================
FIXED-POINT ANALYSIS (Train)
======================================================================

1. Global Attractor Detection:
   Average pairwise L2 distance: 4.199
   Average pairwise cosine similarity: 0.37698
   Status: ✅ No global attractor (diverse fixed points)

2. Zero Solution Detection:
   Average norm: 4.000
   Status: ✅ Non-zero solution

3. Distribution Statistics:
   Norm range: [3.999978, 3.999982]
   Norm std: 0.000001

4. Information Content:
   Effective Rank: 8.33 / 16 (52.0%)
   Top 5 Singular Values: [156.30, 102.49, 73.90, 42.95, 36.61]

======================================================================
✅ PHASE 1 SUCCESSFUL
======================================================================

  Train Effective Rank: 8.33/16 (52%)
  Val Effective Rank:   7.54/16 (47%)

  Proceeding to Phase 2...
```

**Key Metrics**:
- **Converged**: % of tokens converged to fixed points
- **CVFP Loss**: Fixed-point matching loss
- **Dist Loss**: Distribution regularization loss
- **Effective Rank**: Measure of dimensional diversity (higher = better)
- **L2 Distance**: Diversity of fixed points (higher = no global attractor)

---

## 🧪 Development Guidelines

See `CLAUDE.md` for:

- **CVFP Property** - Core principle (DO NOT DELETE)
- **Distribution Regularization** - Dimension collapse solution
- **Phase 1/2 Execution Policy** - When to skip Phase 2
- **Experiment Result Verification** - Must check all metrics
- **Code Quality Policies** - DRY principle, file naming, cleanup

---

## 🎓 Key Research Findings

### 1. Fixed Memory Complexity

New-LLM maintains **O(1) memory** regardless of sequence length, unlike Transformers' O(n²).

### 2. Dimension Collapse Problem

**Problem**: Context vectors can collapse to 1-2 effective dimensions
- Val Effective Rank: 1.01/16 (6%) - Global Attractor
- All tokens converge to nearly identical vectors

**Solution**: Distribution Regularization (force each dimension to follow N(0,1))
- Val Effective Rank: 7.54/16 (47%) - **7.5x improvement**
- Global Attractor completely eliminated

### 3. Two-Phase Training Necessity

**Phase 1 must succeed before Phase 2**:
- Minimum Train Effective Rank: 50/256 (20%)
- Minimum Val Effective Rank: 20/256 (8%)

Running Phase 2 with failed Phase 1 wastes hours of compute time.

---

## 🚀 Future Work

1. **Scale to 256-dim models** - Test with full-size context vectors
2. **Longer sequences** - Validate O(1) memory advantage
3. **Multi-sample training** - 100+ samples for better generalization
4. **Phase 2 optimization** - Improve token prediction performance

---

## 📝 Citation

```bibtex
@misc{newllm2025,
  title={New-LLM: Context Vector Fixed-Point Property for Language Modeling},
  author={New-LLM Project},
  year={2025},
  url={https://github.com/rato-tokyo/new-llm}
}
```

---

## 📄 License

MIT

---

**Status**: Active research project - Phase 1 dimension collapse solved via Distribution Regularization (2025-01-22).
