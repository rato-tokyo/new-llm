# KA-Attention Experiment Results

Date: 2025-12-04

## Experiment Overview

KA-Attention (Key-Attention Output) は、過去トークンのValue vectors (V) を Attention Output (A) で置き換える新しいAttentionメカニズム。

### 用語

- **Attention Output (A)**: `attention_weights @ V` の結果。Context Vectorとも呼ばれる。
- **KAキャッシュ**: Key + Attention Output をキャッシュする方式（KVキャッシュの代替）

### KA-Attention Mechanism

**Standard Attention:**
```
A[i] = Σ_j (attention_weights[i,j] * V[j])  # Attention Output
```

**KA-Attention:**
```
For token n:
- Token 1: Q[1], K[1], V[1] → attention → A[1] (Attention Output)
- Token n: Q[n], K[1:n] → attention_weights → weighted sum of [A[1:n-1], V[n]] → A[n]

KAキャッシュ: K, A をキャッシュ（Vの代わりにAttention Outputを保存）
```

### Hypothesis

過去トークンのAttention Output (A) をValue (V) の代わりに再利用することで:
1. より文脈的な情報を活用できる可能性
2. KVキャッシュをKAキャッシュで置き換え可能か検証

## Configuration

| Parameter | Value |
|-----------|-------|
| Samples | 10,000 |
| Sequence length | 128 |
| Epochs | 10 (with early stopping) |
| Learning rate | 1e-4 |
| Batch size | 32 |
| Device | NVIDIA L4 (24GB) |

## Model Comparison

| Model | Parameters | Architecture |
|-------|------------|--------------|
| Pythia-70M | 70,426,624 | Standard attention |
| KA-Pythia | 70,426,624 | KA-Attention |

Both models have identical parameter counts - only the attention mechanism differs.

## Training Results

### Pythia-70M (Baseline)

| Epoch | Train PPL | Val PPL | Time | Note |
|-------|-----------|---------|------|------|
| 1 | 1255.6 | 875.6 | 50.4s | * |
| 2 | 275.4 | 597.4 | 51.8s | * |
| 3 | 150.1 | 499.0 | 53.4s | * |
| 4 | 97.3 | 452.4 | 52.7s | * |
| 5 | 68.2 | 442.2 | 53.0s | * best |
| 6 | 49.6 | 452.8 | 53.1s | |
| 7 | 36.9 | 472.6 | 52.9s | |
| 8 | 27.6 | 531.2 | 53.0s | Early stop |

**Best: Epoch 5, val_ppl = 442.2**

### KA-Pythia (KA-Attention)

| Epoch | Train PPL | Val PPL | Time | Note |
|-------|-----------|---------|------|------|
| 1 | 1380.0 | 949.3 | 247.9s | * |
| 2 | 312.5 | 642.9 | 246.8s | * |
| 3 | 170.2 | 533.1 | 247.5s | * |
| 4 | 111.7 | 475.7 | 248.3s | * |
| 5 | 80.0 | 453.6 | 247.5s | * |
| 6 | 59.5 | 449.3 | 246.7s | * best |
| 7 | 45.5 | 454.5 | 247.0s | |
| 8 | 35.4 | 481.4 | 246.6s | |
| 9 | 27.8 | 507.6 | 246.8s | Early stop |

**Best: Epoch 6, val_ppl = 449.3**

## Summary

| Model | Best PPL | Best Epoch | Time/Epoch |
|-------|----------|------------|------------|
| Pythia-70M | 442.2 | 5 | ~52s |
| KA-Pythia | 449.3 | 6 | ~247s |

**PPL Difference: +7.1 (+1.6%)**

## Analysis

### Performance

- KA-Attention achieves **comparable performance** to standard attention
- PPL difference of only +7.1 (1.6% worse) suggests the mechanism is viable
- KA-Pythia takes **~4.7x longer** per epoch due to sequential A computation

### Training Speed

The significant slowdown is due to the sequential nature of KA-Attention:
- Each token's A[i] depends on all previous A[1:i-1]
- Cannot be parallelized across sequence positions
- Standard attention can compute all positions in parallel

### Conclusions

1. **KA-Attention works**: The mechanism can learn effectively with only slight degradation
2. **Speed is a concern**: 4.7x slowdown makes training impractical for large scale
3. **KA cache potential**: During inference, storing A instead of V could have benefits
4. **Needs optimization**: Current implementation is not optimized for speed

## Future Directions

1. **Inference testing**: Compare KA cache vs KV cache during generation
2. **Approximation**: Test approximate KA-Attention that allows parallelization
3. **Hybrid approach**: Use standard attention during training, KA cache during inference
4. **Memory analysis**: Compare memory usage of KA cache vs KV cache

## Files

- Script: `scripts/experiment_ka_comparison.py`
- Model: `src/models/ka_attention.py`
