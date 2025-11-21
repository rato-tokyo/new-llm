# New-LLM Dialogue Training: Simplified Architecture

## Overview

This is a **completely refactored** New-LLM implementation focused on:
- ✅ **UltraChat dialogue dataset** training
- ✅ **Flexible architecture** (easy to change layers and context_dim)
- ✅ **Two-phase training** (context learning → token prediction)
- ✅ **Local GPU training** (no Colab dependency)
- ✅ **Comprehensive caching** for efficiency

---

## Architecture

### Model: `new_llm_flexible.py`

**Key Features**:
- Variable number of layers (configurable)
- Variable context vector dimensions (configurable)
- Gated context updater (prevents global attractor)
- Fixed-point context computation

**Easy to Change**:
```python
# In dialogue_config.py
num_layers = 2           # Change to 1, 2, 3, 4, ...
context_dim = 256        # Change to 128, 256, 512, 1024, ...
hidden_dim = 512         # Change to 256, 512, 1024, ...
```

---

## Two-Phase Training

### Phase 1: Context Vector Learning

**Goal**: Find stable fixed-point context for each token

**Method**:
1. **First pass**: Forward through dialogue, just observe contexts
2. **Second pass onwards**: Use previous context as teacher signal
3. **Converge**: Repeat until contexts stabilize (L2 distance < threshold)

**Success Criteria**:
- At least 95% of tokens converge
- If not → increase `num_layers` or `context_dim`

### Phase 2: Token Prediction Learning

**Goal**: Train to predict next tokens (like standard LLM)

**Method**:
1. **Freeze contexts** at fixed points from Phase 1
2. **Train token output head** to predict next tokens
3. **Only train on assistant responses** (not user messages)

**Training Details**:
- Batch size: 1 (for debugging, can increase later)
- Per-step logging: Loss, Perplexity, Accuracy
- Only backprop through assistant tokens (using mask)

---

## Dataset: UltraChat with ChatML Format

### UltraChat Structure

```json
{
  "messages": [
    {"role": "user", "content": "How do I...?"},
    {"role": "assistant", "content": "You can..."},
    {"role": "user", "content": "Thanks!"},
    {"role": "assistant", "content": "You're welcome!"}
  ]
}
```

### ChatML Formatting

We convert to ChatML format to distinguish speakers:

```
<|im_start|>user
How do I...?<|im_end|>
<|im_start|>assistant
You can...<|im_end|>
```

This allows the model to learn:
- User asks questions
- Assistant provides answers
- Turn-taking structure

---

## Quick Start

### 1. Test Basic Functionality

```bash
python3 test_training.py
```

This verifies:
- Data loading works
- Model creation works
- Forward pass works
- Fixed-point context computation works

### 2. Run Full Training

```bash
python3 train_dialogue.py
```

This runs:
- Phase 1: Context learning (first N samples)
- Phase 2: Token prediction (same N samples)

**First Run**: Use `SmallDialogueConfig` (1 layer, 128 dim, 10 samples)

---

## Configuration

### Pre-defined Configs

| Config | Layers | Context Dim | Hidden Dim | Samples |
|--------|--------|-------------|------------|---------|
| **Small** | 1 | 128 | 256 | 10 |
| **Medium** | 2 | 256 | 512 | 100 |
| **Large** | 4 | 512 | 1024 | 1000 |

### Modify Config

```python
# src/utils/dialogue_config.py
class MyConfig(DialogueConfig):
    num_layers = 3           # Your choice
    context_dim = 384        # Your choice
    hidden_dim = 768         # Your choice
    phase1_max_samples = 50  # Your choice
```

---

## Caching System

### What is Cached?

1. **Tokenizer** (`./cache/tokenizer/`)
2. **UltraChat dataset** (`./cache/` - auto by HuggingFace)
3. **Fixed-point contexts** (`./cache/contexts/context_cache.pt`)

### Why Caching?

- **Tokenizer**: Loaded once, reused forever
- **Dataset**: Downloaded once, reused forever
- **Contexts**: Phase 1 results reused in Phase 2

**Result**: Second run is much faster (no recomputation)

---

## Monitoring Training

### Phase 1 Output

```
Sample 0: 512 tokens
Dialogue: 8 messages
  Converged: 487/512 tokens (95.1%)
  Avg iterations: 12.3

Phase 1 Results:
  Total samples: 10
  Convergence rate: 94.8%
  Avg iterations: 13.1
```

**Interpretation**:
- ✅ 95%+ convergence → Proceed to Phase 2
- ❌ <95% convergence → Increase layers or context_dim

### Phase 2 Output

```
Sample 0: Training token prediction
  Epoch 1/10: Loss=5.2341, PPL=187.23, Acc=12.3%
  Epoch 5/10: Loss=3.8912, PPL=48.95, Acc=28.7%
  Epoch 10/10: Loss=2.1543, PPL=8.62, Acc=51.2%
```

**Interpretation**:
- Loss decreases → Model is learning
- PPL decreases → Better prediction quality
- Accuracy increases → More tokens predicted correctly

---

## File Structure

```
new-llm/
├── train_dialogue.py              # Main training script
├── test_training.py               # Quick test script
├── src/
│   ├── models/
│   │   └── new_llm_flexible.py    # Flexible model
│   ├── data/
│   │   └── ultrachat_loader.py    # UltraChat + ChatML
│   └── utils/
│       └── dialogue_config.py     # Configurations
└── cache/                         # All cached data
    ├── tokenizer/                 # Tokenizer cache
    ├── contexts/                  # Fixed-point contexts
    └── checkpoints/               # Model checkpoints
```

---

## Design Philosophy

### 1. Simplicity

**No complex inheritance**, **no abstract base classes**, **no over-engineering**

Each file is **self-contained** and **easy to understand**:
- `new_llm_flexible.py`: Just the model
- `ultrachat_loader.py`: Just data loading
- `dialogue_config.py`: Just configurations
- `train_dialogue.py`: Just training logic

### 2. Flexibility

**Easy to modify key parameters**:
- Change `num_layers` in config → model automatically adjusts
- Change `context_dim` in config → model automatically adjusts
- No need to modify model code

### 3. Efficiency

**Aggressive caching**:
- Tokenizer loaded once
- Dataset downloaded once
- Fixed-point contexts computed once

**Result**: Fast iteration speed

---

## Next Steps

### Immediate

1. **Run test**: `python3 test_training.py`
2. **Run small training**: `python3 train_dialogue.py` (SmallDialogueConfig)
3. **Check Phase 1 convergence**: >95%?
   - ✅ Yes → Proceed to Phase 2
   - ❌ No → Increase layers/context_dim

### Experiments

1. **Vary architecture**:
   - Try 1, 2, 3, 4 layers
   - Try 128, 256, 512, 1024 context_dim
   - Find minimum for 95% convergence

2. **Scale up**:
   - Start with 10 samples
   - Then 100 samples
   - Then 1000 samples

3. **Multi-sample strategies**:
   - Option A: Concatenate all samples → train as single sequence
   - Option B: Train sample-by-sample → accumulate gradients
   - Need to experiment which works better

---

## Troubleshooting

### Low Phase 1 Convergence (<95%)

**Problem**: Contexts don't converge to fixed points

**Solutions**:
1. Increase `num_layers` (1 → 2 → 3)
2. Increase `context_dim` (128 → 256 → 512)
3. Increase `hidden_dim` (256 → 512 → 1024)
4. Increase `phase1_max_iterations` (100 → 200)

### High Phase 2 Loss (>5.0 after 10 epochs)

**Problem**: Token prediction not learning

**Solutions**:
1. Check if Phase 1 converged properly
2. Increase Phase 2 epochs (10 → 50)
3. Adjust learning rate (0.0001 → 0.0002)
4. Check if assistant_mask is correct (use debug prints)

### GPU Out of Memory

**Problem**: Model too large for GPU

**Solutions**:
1. Reduce `batch_size` (already 1, can't reduce)
2. Reduce `max_seq_length` (512 → 256)
3. Reduce `context_dim` (512 → 256)
4. Reduce `num_layers` (4 → 2)

---

## Design Decisions

### Why No Colab?

**Reason**: GPU parallelization is difficult for sequential models

New-LLM is inherently sequential (unlike Transformer), so:
- No benefit from Colab's multi-GPU
- Local GPU is simpler and more flexible
- Long-term training is expected (we're prepared)

### Why Batch Size = 1?

**Reason**: Start simple, debug thoroughly

- Easier to debug per-step metrics
- Easier to verify loss/accuracy computation
- Can increase later after validation

### Why ChatML Format?

**Reason**: Clear speaker distinction

- Standard format used by many dialogue models
- Easy to parse (special tokens)
- Model can learn turn-taking structure

### Why Two-Phase Training?

**Reason**: Context vectors are fundamental to New-LLM

- Phase 1 ensures contexts are meaningful (fixed points)
- Phase 2 builds on stable foundations
- Separates concerns (context vs prediction)

---

## Performance Expectations

### Phase 1 (Context Learning)

- **Small (1 layer, 128 dim)**: May not converge (60-80%)
- **Medium (2 layers, 256 dim)**: Likely converges (90-95%)
- **Large (4 layers, 512 dim)**: Should converge (>95%)

### Phase 2 (Token Prediction)

- **Initial PPL**: 100-200 (random prediction)
- **After 10 epochs**: 10-50 (learning progress)
- **Target PPL**: <10 (good dialogue model)

**Note**: These are rough estimates, actual results depend on data

---

## Comparison with Old System

| Aspect | Old System | New System |
|--------|-----------|------------|
| **Dataset** | WikiText, CVFPT | UltraChat (dialogue) |
| **Platform** | Colab (one-line scripts) | Local GPU |
| **Architecture** | Fixed (hard to change) | Flexible (easy to change) |
| **Training** | Single-phase (token only) | Two-phase (context + token) |
| **Complexity** | Many files, many configs | Few files, simple configs |
| **Caching** | Minimal | Comprehensive |

---

## Summary

**This is a complete rewrite focused on simplicity and flexibility.**

Key improvements:
- ✅ Easy to change architecture (layers, context_dim)
- ✅ Two-phase training (context then tokens)
- ✅ UltraChat dialogue dataset
- ✅ Comprehensive caching (efficiency)
- ✅ Clear code structure (readability)

**Start small, iterate fast, scale up gradually.**
