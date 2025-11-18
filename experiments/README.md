# New-LLM Experiments

This directory contains all experimental results and findings for the New-LLM project.

## 📊 Latest Experiments (2025-11-18)

### Layer Optimization Experiments

**File**: `layer_optimization_experiment_2025-11-18.md`

Comprehensive layer optimization study testing 1, 4, 5, 6, 7 layers on WikiText-2:

| Layers | Val PPL | Val Acc | Status | Notes |
|--------|---------|---------|--------|-------|
| **4** | **20.1-20.2** | **38.3-38.4%** | Partial (135/150) | **Most promising** |
| **5** | **20.50** | **38.3%** | Complete | Beats baseline |
| **6** | 20.60 | 38.5% | Complete (150 epochs) | Baseline |
| **7** | 21.77 | 37.9% | Complete | Worse than baseline |
| **1** | 20.4-20.5 | 38.0-38.1% | Partial (75/150) | Good performance |

**Key Findings**:
- Layer 4-5 range optimal for WikiText-2
- "Fewer layers = better" hypothesis partially confirmed
- Layer 7 clearly excessive for task complexity

---

### FP16 Mixed Precision Training

**Files**:
- `l4_fp16_experiment_2025-11-18.md` - Initial FP16 training
- `l4_fp16_baseline_50epochs_2025-11-18.md` - 50-epoch baseline
- `l4_advanced_experiment_2025-11-18.md` - Advanced model (4.84M params)

**Configuration**:
- GPU: L4 (24GB VRAM)
- Batch size: 2048 (L4 optimized)
- Learning rate: 0.0008 (Square Root Scaling Rule)
- Precision: FP16 mixed precision

**Results**:
- Baseline (2.74M params): PPL 20.5-20.6
- Advanced (4.84M params): PPL 40.93 (learning rate needs adjustment)

---

## 📂 Experiment Archive

Older experiments and comprehensive summaries:

- `COMPREHENSIVE_EXPERIMENT_SUMMARY.md` - Complete experiment history
- `baseline_wikitext_experiment.md` - WikiText-2 baseline experiments
- `colab_advanced_experiment.md` - Early Colab experiments

---

## 🎯 Current Focus (2025-11-19)

**Dolly-15k Dialog Training**

Training New-LLM on Dolly-15k instruction-response pairs for dialog capabilities.

**Status**: In progress
**Configuration**:
- Layer 1 baseline
- Context vector: 256 dimensions
- Max seq length: 128 (longer for dialog)
- GPU: L4, batch_size=2048

**Goal**: Achieve dialog/instruction-following capability

---

## 📈 Performance Summary

### WikiText-2 Language Modeling

| Model | Layers | Params | Val PPL | Val Acc | Training |
|-------|--------|--------|---------|---------|----------|
| **Layer 4** | 4 | ~2.5M | **20.1** | **38.3%** | Partial (most promising) |
| **Layer 5** | 5 | ~2.6M | **20.5** | **38.3%** | Complete |
| **Layer 6 (Baseline)** | 6 | 2.74M | 20.6 | 38.5% | Complete |
| Layer 1 | 1 | ~1.4M | 20.4 | 38.0% | Partial |
| Layer 7 | 7 | ~3.0M | 21.8 | 37.9% | Complete |

---

## 🔬 Experimental Findings

### Scaling Rules Discovered

1. **Batch Size Scaling (Square Root Rule)**:
   ```
   batch_size 32→2048 (64x) → learning_rate 0.0001→0.0008 (√64 = 8x)
   ```

2. **Model Size Scaling**:
   ```
   Larger model (4.84M vs 2.74M) → Lower learning rate (0.0008→0.0004)
   ```

3. **Layer Optimization**:
   - Optimal range: 4-5 layers for WikiText-2
   - Too few (<3): Underfitting
   - Too many (>6): Overfitting or inefficiency

### GPU Optimization

| GPU | VRAM | Batch Size | Learning Rate | Performance |
|-----|------|------------|---------------|-------------|
| **T4** | 16GB | 512 | 0.0001 | Baseline |
| **L4** | 24GB | **2048** | **0.0008** | **4x speedup** |
| **A100** | 40GB | 4096 | 0.0011 | 8x speedup (estimated) |

---

## 📝 Experiment Guidelines

When adding new experiments:

1. **File naming**: `{experiment_type}_{date}.md`
2. **Include**:
   - Configuration (model, hyperparameters, GPU)
   - Results (PPL, accuracy, training time)
   - Analysis and findings
   - Comparisons with baselines

3. **Update this README** with key findings

---

## 🚀 Future Experiments

Planned experiments:

1. **Context Expansion**:
   - Expand context vector 256→512 dimensions
   - Two modes: freeze base dims vs fine-tune all
   - Progressive neural growth approach

2. **Japanese Dialog**:
   - Japanese Alpaca dataset
   - Multilingual capability

3. **Longer Sequences**:
   - Test with max_seq_length > 128
   - Memory efficiency validation

---

## 📖 Related Documentation

- `/DOLLY_TRAINING.md` - Dolly-15k training guide
- `/CLAUDE.md` - Development guidelines
- `/README.md` - Project overview
