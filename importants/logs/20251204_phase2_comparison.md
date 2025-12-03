# Phase 2 比較実験結果 (2025-12-04)

## 実験概要

Context-Pythia（KVキャッシュ50%圧縮）とPythia-70M（ベースライン）の性能比較。

## 実験設定

| 項目 | 値 |
|------|-----|
| Device | NVIDIA L4, 22.2GB |
| Samples | 10,000 |
| Sequence length | 128 |
| Epochs | 最大10（Early stopping有効） |
| Batch size | 32 |
| Learning rate | 0.0001 |
| Train samples | 9,000 |
| Val samples | 1,000 |
| context_dim | 256 |

## 結果

| Model | Best PPL | Best Epoch | KV Cache | Reduction |
|-------|----------|------------|----------|-----------|
| Pythia-70M | 471.3 | 5 | 3072.0 KB | - |
| Context-Pythia | 554.2 | 4 | 1536.0 KB | 50% |

**PPL差: +83.0（Context-Pythiaが17.6%悪化）**

## 学習曲線

### Pythia-70M (Baseline)
```
Trainable: 70,426,624 / 70,426,624 parameters
Epoch  1: train_loss=7.2257 val_ppl=973.2 val_acc=14.25% [55.3s] *
Epoch  2: train_loss=5.6975 val_ppl=653.5 val_acc=16.01% [56.1s] *
Epoch  3: train_loss=5.0583 val_ppl=537.4 val_acc=17.11% [55.9s] *
Epoch  4: train_loss=4.5944 val_ppl=480.1 val_acc=17.95% [56.0s] *
Epoch  5: train_loss=4.2090 val_ppl=471.3 val_acc=18.28% [55.9s] *
Epoch  6: train_loss=3.8628 val_ppl=481.1 val_acc=18.06% [56.0s]
→ Early stop: val_ppl worsened (481.1 > 471.3)
```

### Context-Pythia (50% KV Compression)
```
Trainable: 70,559,232 / 70,756,608 parameters
Epoch  1: train_loss=7.3496 val_ppl=1062.7 val_acc=13.67% [61.7s] *
Epoch  2: train_loss=5.8054 val_ppl=691.4 val_acc=16.00% [61.9s] *
Epoch  3: train_loss=5.0934 val_ppl=581.1 val_acc=16.95% [62.0s] *
Epoch  4: train_loss=4.5864 val_ppl=554.2 val_acc=17.10% [62.0s] *
Epoch  5: train_loss=4.1675 val_ppl=572.3 val_acc=17.11% [62.0s]
→ Early stop: val_ppl worsened (572.3 > 554.2)
```

## 考察

### 性能差の原因（推測）

1. **情報ボトルネック**: 512-dim → 256-dim への圧縮で情報が失われている
2. **Phase 1学習の質**: OACDで91%収束達成したが、さらに高い収束率が必要かもしれない
3. **データ量不足**: 10,000サンプルでは圧縮モデルの学習には不十分な可能性

### ポジティブな点

- Context-Pythiaは**KVキャッシュを50%削減**しつつ、PPLは17.6%の悪化に留まっている
- 学習は安定して進行し、収束している
- 1エポックあたりの時間差は約10%（55秒 vs 62秒）

### 改善案

1. **Phase 1の改善**
   - 収束率95%以上を目指す
   - より多くのトークンで学習

2. **Phase 2の改善**
   - より多くのサンプルで学習（50,000+）
   - 学習率の調整
   - より長いエポック数

3. **アーキテクチャの改善**
   - context_dimを300や384に増やす（圧縮率と精度のトレードオフ）
   - ContextBlockの層数を増やす

## 次のステップ

- [ ] context_dim=300 で実験
- [ ] サンプル数50,000で実験
- [ ] Phase 1収束率95%以上での再実験
