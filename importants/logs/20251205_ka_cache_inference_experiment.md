# KA Cache Inference Experiment Results

Date: 2025-12-05

## Experiment Overview

案3「推論時のみKAキャッシュ」の検証実験。
標準Attentionで学習したモデルを使い、推論時にKAキャッシュを使用した場合の性能劣化を測定。

### 方式

```
学習: 標準Attention（並列計算、高速）
推論: KAキャッシュ（過去のA[1:i-1]をキャッシュ、V[i]は現在のみ計算）

A[i] = weights @ [A[1:i-1], V[i]]
     = weights_cached @ A_cached + weights_current @ V_current
```

### 目的

- 学習時の4.7x遅延を回避しつつ、推論時にKAキャッシュを使用可能か検証
- KAキャッシュによるPPL劣化を測定

## Configuration

| Parameter | Value |
|-----------|-------|
| Samples | 5,000 |
| Sequence length | 128 |
| Epochs | 10 (early stopping patience=1) |
| Learning rate | 1e-4 |
| Batch size | 8 |
| Device | NVIDIA L4 (24GB) |

## Training Results (Standard Attention)

| Epoch | Train PPL | Val PPL | Note |
|-------|-----------|---------|------|
| 1 | 903.0 | 377.8 | * |
| 2 | 210.3 | 214.6 | * |
| 3 | 106.6 | 156.3 | * |
| 4 | 63.3 | 128.6 | * |
| 5 | 40.4 | 114.7 | * |
| 6 | 26.6 | 108.5 | * best |
| 7 | 17.6 | 109.8 | |
| 8 | 11.8 | 115.8 | |
| 9 | 7.9 | 127.2 | Early stop |

**Best: Epoch 6, val_ppl = 108.5**

## Inference Evaluation

| Cache Type | PPL | Tokens/sec | Total Time |
|------------|-----|------------|------------|
| KA Cache | 300.1 | 1,264.9 | 50.60s |

**注**: KVキャッシュ（baseline）はスキップされた（`--skip-baseline`）

## Analysis

### PPL劣化

```
Training PPL (standard forward): 108.5
KA Cache Inference PPL: 300.1

劣化: +191.6 (+176.6%)
```

### 問題点

**重大なPPL劣化が発生している。**

原因の考察:
1. **AとVの意味の違い**: Attention Output (A) はVの重み付き和であり、Vとは異なる分布を持つ
2. **学習時との不整合**: 標準AttentionではVを直接使用して学習しているが、推論時にAを使用すると入力分布が異なる
3. **累積誤差**: 各レイヤーでの誤差が蓄積し、深いレイヤーほど劣化が大きくなる可能性

### 前回実験（KA-Attention学習）との比較

| 方式 | Training | Inference | Val PPL |
|------|----------|-----------|---------|
| 標準Attention | Standard | Standard | 108.5 |
| KA-Attention (2024-12-04) | KA | KA | 449.3 |
| 案3 (今回) | Standard | KA | 300.1 |

案3はKA-Attentionでの学習より良いが、標準Attentionからは大幅に劣化。

## Conclusions

### 案3（推論時のみKAキャッシュ）の評価

1. **実用性: 低い** - PPL +176%の劣化は許容できない
2. **原因**: 学習時と推論時のAttention出力分布の不整合
3. **代替案の必要性**: Adapter方式（案1）の検討が必要

### 次のステップ

1. **案1（Adapter方式）の実験**: A→V変換を学習するAdapterを追加
2. **KVキャッシュとの直接比較**: 同一モデルでKVキャッシュの結果を取得
3. **Reversal Curse評価**: 推論方式によるReversal Curseへの影響を確認

## Missing Evaluations

**注意**: 本実験ではReversal Curse評価が実施されていない。
今後の実験では必ず以下を含めること:
- Forward PPL（順方向文）
- Backward PPL（逆方向文）
- Reversal Ratio（Forward / Backward）

## Files

- Script: `scripts/experiment_ka_cache.py`
- Model: `src/models/ka_cache.py`
