# 10,000 Token Experiment Status

**Started**: 2025-11-24 00:55 JST
**Status**: Running in background
**Expected completion**: ~10-11 hours (around 11:00-12:00 JST)

## Experiment Overview

### Purpose
1. **Performance Measurement**: Measure actual training time at scale
2. **Convergence Analysis**: Evaluate convergence behavior with 10k tokens
3. **Diversity Metrics**: Analyze Effective Rank and context diversity
4. **Parameter Optimization**: Generate data for config.py tuning decisions

### Configuration
- **Training tokens**: 10,000
- **Validation tokens**: 2,000
- **Architecture**: 6 layers, 768-dim embeddings & context
- **Learning rate**: 0.002
- **Diversity weight**: 0.99
- **Max iterations**: 10
- **Device**: CPU

### Expected Outputs

#### 1. Training Log
**File**: `experiment_10k_output.log`
**Contents**:
- Timestamped progress updates
- Per-iteration metrics (CVFP loss, Diversity loss, Convergence %)
- Performance measurements (tokens/sec)
- Final analysis and recommendations

#### 2. Results File
**File**: `checkpoints/experiment_10k_results.pt`
**Contents**:
```python
{
    'config': { ... },           # Experiment configuration
    'performance': {
        'train_time_sec': ...,
        'val_time_sec': ...,
        'train_tokens_per_sec': ...,
        'iterations': ...,
        'train_convergence_rate': ...
    },
    'metrics': {
        'train': {
            'effective_rank': ...,
            'rank_ratio': ...,
            'mean_norm': ...,
            'std_norm': ...
        },
        'val': { ... }
    },
    'timestamp': '...'
}
```

#### 3. Model Checkpoint
**File**: `checkpoints/model_latest.pt`
**Contents**: Updated model weights after 10k token training

### Analysis Focus

#### 1. **Convergence Rate**
- Target: ≥95% convergence in 10 iterations
- If <50%: Consider increasing learning_rate or max_iterations
- If ≥95%: Current settings are effective

#### 2. **Effective Rank**
- Target: ≥80% of context_dim (≥614/768)
- If <50%: Increase dist_reg_weight
- If ≥80%: Diversity regularization working well

#### 3. **Train/Val Gap**
- Target: <10% difference
- Large gap may indicate overfitting/underfitting

#### 4. **Processing Speed**
- Expected: ~15-17 tokens/sec (based on prior measurements)
- Will provide accurate time projections for larger datasets

### Parameter Tuning Decisions

Based on results, we will adjust:

1. **phase1_learning_rate** (current: 0.002)
   - If convergence too slow → increase
   - If unstable → decrease

2. **phase1_max_iterations** (current: 10)
   - If not converging → increase
   - If converging early → can reduce

3. **dist_reg_weight** (current: 0.99)
   - If Effective Rank too low → increase
   - If CVFP loss not decreasing → decrease

4. **phase1_convergence_threshold** (current: 0.05)
   - Adjust based on actual convergence behavior

### Time Projections

Based on expected ~16 tok/s × 10 iterations:

| Dataset Size | Estimated Time |
|-------------|----------------|
| 10,000 | ~10-11 hours ✓ |
| 50,000 | ~52 hours (~2 days) |
| 100,000 | ~104 hours (~4 days) |
| 500,000 | ~521 hours (~22 days) |

**Note**: GPU would reduce these times by 10-20x

## Monitoring Progress

```bash
# Check current progress
tail -f experiment_10k_output.log

# Check if still running
ps aux | grep experiment_10k

# View last 50 lines
tail -50 experiment_10k_output.log
```

## Next Steps (After Completion)

1. **Load and analyze results**:
   ```python
   results = torch.load('checkpoints/experiment_10k_results.pt')
   print(results['performance'])
   print(results['metrics'])
   ```

2. **Review recommendations** in log file

3. **Update config.py** based on findings

4. **Run Phase 2 experiment** if Phase 1 results are good

---

**Note**: This experiment runs overnight. Check results in the morning.

**Last Updated**: 2025-11-24 00:56 JST
