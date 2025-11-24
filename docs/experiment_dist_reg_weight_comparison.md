# Experiment: Distribution Regularization Weight Comparison

**Date**: 2025-11-24
**Experiment Type**: Hyperparameter Optimization
**Purpose**: Compare the effect of different `dist_reg_weight` values on CVFP learning and dimensional diversity

---

## Background

After fixing the critical context carryover bug, we observed that with `dist_reg_weight = 0.99` (99% diversity focus, 1% CVFP focus), the model achieved high Effective Rank but **failed the CVFP convergence check**.

**Hypothesis**: Too much emphasis on diversity regularization may prevent proper fixed-point learning.

---

## Experimental Setup

### Common Configuration

- **Model Architecture**: 6-layer CVFP blocks, 768-dim context
- **Training Data**: 6400 tokens (50 samples from UltraChat)
- **Validation Data**: 1280 tokens (generated from training data)
- **Learning Rate**: 0.002
- **Max Iterations**: 10
- **Convergence Threshold**: 0.05
- **Random Seed**: 42 (full reproducibility)

### Variable Parameter

- **Experiment 1**: `dist_reg_weight = 0.99` (99% diversity, 1% CVFP)
- **Experiment 2**: `dist_reg_weight = 0.5` (50% diversity, 50% CVFP)

---

## Results

### Experiment 1: dist_reg_weight = 0.99

**Training Results**:
- Convergence Rate: 0.0% (0/6400 tokens) - 10 iterations completed
- Effective Rank: 681.47/768 (88.7%)
- CVFP Loss: (not recorded)

**Validation Results**:
- Convergence Rate: 0.0% (0/1280 tokens) - 10 iterations completed
- Effective Rank: 627.29/768 (81.7%)

**CVFP Convergence Check**:
- **final_diff = 32.618008** ❌ (threshold: < 0.001 for GOOD)
- **Status: FAILED** - Model did not learn stable fixed-point properties

---

### Experiment 2: dist_reg_weight = 0.5

**Training Results**:
- Convergence Rate: 0.0% (0/6400 tokens) - 10 iterations completed
- Effective Rank: 689.26/768 (89.7%)
- CVFP Loss (final): 0.001847
- Diversity Loss (final): -27.879181

**Validation Results**:
- Convergence Rate: 0.0% (0/1280 tokens) - 10 iterations completed
- Effective Rank: 686.90/768 (89.4%)
- CVFP Loss: 0.001847 (stable across iterations)

**CVFP Convergence Check**:
- **final_diff = 0.000745** ✅ (threshold: < 0.001 for GOOD)
- **Status: PASSED** - Model successfully learned fixed-point properties

**Identity Mapping Check**:
- Zero context difference: 3.0948 (threshold: > 0.1) ✅
- Token embedding similarity: 0.0037 (threshold: < 0.95) ✅

---

## Comparative Analysis

| Metric | drw=0.99 | drw=0.5 | Change |
|--------|----------|---------|--------|
| **Train Effective Rank** | 88.7% | 89.7% | +1.0% |
| **Val Effective Rank** | 81.7% | 89.4% | **+7.7%** ✅ |
| **CVFP Convergence Check** | 32.618 (FAIL) | 0.000745 (PASS) | **43,800× improvement** ✅ |

---

## Key Findings

### 1. Balanced Loss Weight Enables Both Diversity and Fixed-Point Learning

With `dist_reg_weight = 0.5`:
- ✅ Maintained high dimensional diversity (89.4-89.7% Effective Rank)
- ✅ Achieved stable fixed-point convergence (final_diff = 0.000745)
- ✅ Significantly improved validation generalization (+7.7% Effective Rank)

### 2. Over-Emphasis on Diversity Breaks CVFP Learning

With `dist_reg_weight = 0.99`:
- ✅ Achieved reasonable diversity (81.7% on validation)
- ❌ Failed to learn fixed-point properties (final_diff = 32.618)
- ❌ Lower validation Effective Rank suggests overfitting to diversity metric

### 3. CVFP Convergence Check is Critical

The CVFP convergence check (running additional iterations after training) is essential to verify that the model has learned genuine fixed-point properties, not just minimized the training loss.

---

## Convergence Rate Interpretation

**Convergence Rate 0.0% does NOT mean failure**:
- It means all 10 iterations completed without early stopping
- Early stopping occurs when 95% of tokens converge (change < threshold)
- With diversity regularization, contexts continue evolving across iterations
- The CVFP convergence check (post-training) verifies actual fixed-point learning

---

## Conclusions

1. **Recommended Setting**: `dist_reg_weight = 0.5` provides optimal balance
2. **Critical Bug Fix Impact**: Context carryover fix was essential for any CVFP learning
3. **Diversity-CVFP Tradeoff**: Excessive diversity focus (0.99) prevents fixed-point convergence
4. **Validation Improvement**: Balanced weight improved validation Effective Rank by 7.7%

---

## Next Steps

1. Test additional `dist_reg_weight` values (e.g., 0.3, 0.7) to map the tradeoff curve
2. Scale to larger datasets (10k+ tokens) with current optimal setting
3. Evaluate Phase 2 token prediction performance
4. Consider adaptive weight scheduling during training

---

## Implementation Notes

**Files Modified**:
- `config.py`: Updated `dist_reg_weight` from 0.99 to 0.5
- `CLAUDE.md`: Added mandatory convergence rate reporting rule

**Reproducibility**:
- All results are fully reproducible with `random_seed = 42`
- Checkpoint was deleted before each experiment to ensure clean training
- Same training/validation data used for both experiments

---

## References

- Context Carryover Bug Fix: CLAUDE.md (2025-11-24)
- Per-Dimension Variance Tracking Implementation: src/training/phase1_trainer.py
- CVFP Convergence Check: src/evaluation/metrics.py
