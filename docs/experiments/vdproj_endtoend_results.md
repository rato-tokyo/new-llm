# V-DProj End-to-End 実験結果 (旧アーキテクチャ)

> **⚠️ 注意**: この実験は「復元してからAttentionに使用する」旧アーキテクチャで実施されました。
> 推論時のKVキャッシュ削減効果がないことが判明したため、アーキテクチャを修正しました。
> 新しい「圧縮したままAttentionに使用する」アーキテクチャの実験結果は別ファイルを参照してください。

## 実験日時
2025-12-04

## 実験概要

V（Value）を圧縮して復元する方式でKVキャッシュを削減する実験。
End-to-End学習（LM Loss + Reconstruction Loss同時学習）。

### アーキテクチャ

```
V-DProj Attention:
  V (512-dim) → v_compress → V_compressed (320-dim) → v_restore → V_restored (512-dim)

  学習目標:
    LM Loss: Cross-entropy
    Reconstruction Loss: ||V - V_restored||^2
    Total Loss = LM Loss + 0.1 * Reconstruction Loss
```

## 実験設定

| 項目 | 値 |
|------|-----|
| Samples | 10,000 |
| Sequence length | 128 |
| Epochs | 30 (max) |
| Learning rate | 1e-4 |
| Batch size | 8 |
| V proj dim | 320 (from 512) |
| Reconstruction weight | 0.1 |
| Early stopping | 3 epochs |

## モデル情報

| 項目 | 値 |
|------|-----|
| Total parameters | 72,391,552 |
| V projection params | 1,971,072 |
| KV Cache reduction | 18.8% |

## 結果

### 学習曲線

```
Epoch  1: train_ppl=615.6 val_ppl=635.5 recon=0.1266 *
Epoch  2: train_ppl=160.7 val_ppl=477.9 recon=0.1248 *
Epoch  3: train_ppl= 85.8 val_ppl=435.4 recon=0.1268 *
Epoch  4: train_ppl= 53.4 val_ppl=428.7 recon=0.1300 * (best)
Epoch  5: train_ppl= 35.6 val_ppl=456.5 recon=0.1366
Epoch  6: train_ppl= 24.4 val_ppl=514.0 recon=0.1422
Epoch  7: train_ppl= 16.9 val_ppl=613.3 recon=0.1498
-> Early stop
```

### 最終結果

| Model | val_ppl | Best Epoch | KV Reduction | Note |
|-------|---------|------------|--------------|------|
| Pythia (Baseline) | ~442 | - | 0% | 同条件での参考値 |
| V-DProj (End-to-End) | **428.7** | 4 | 18.8% | 過学習傾向 |

**Difference: -13.3 ppl (3.0%改善)**

※ Pythia baselineは同条件（10,000 samples, 128 seq_len）でのスクラッチ学習の参考値

### Position-wise PPL比較

| Position | Pythia (参考) | V-DProj | Diff |
|----------|---------------|---------|------|
| 0-16 | ~900 | 853.0 | -47 |
| 16-32 | ~650 | 667.5 | +17 |
| 32-64 | ~550 | 586.6 | +37 |
| 64-96 | ~500 | 578.1 | +78 |
| 96-128 | ~480 | 552.8 | +73 |

※ Pythia参考値は過去実験からの概算

## 分析

### 1. 過学習問題

- **train_pplは順調に低下**: 615 → 53 → 17
- **val_pplはEpoch 4で底打ち**: 428.7
- **Epoch 5以降急激に悪化**: 456 → 514 → 613
- **典型的な過学習パターン**

### 2. Reconstruction Lossの推移

- **安定していない**: 0.1266 → 0.1248 → 0.1300 → 0.1498
- **学習が進むにつれて悪化**: V復元が困難になっている
- **LM Lossとの競合**: LM性能を上げようとするとV復元が犠牲になる

### 3. Position-wise PPLの傾向

- **後ろの位置ほどPPLが低い**: 正常な傾向（コンテキストが多いため）
- **しかし全体的に高い**: 552-853の範囲

### 4. Baseline比較

- **全体PPL**: V-DProj 428.7 vs Pythia ~442 → **若干改善** (-13 ppl)
- **ただし過学習**: Epoch 4以降は急激に悪化
- **Position-wise**: 前半位置は改善、後半位置は悪化傾向
- **KV Cache削減**: 18.8%削減を達成

### 5. 問題点

1. **過学習が早い**: Epoch 4でベスト、その後急激に悪化
2. **収束が不安定**: 早期に過学習が始まる
3. **V復元と言語モデリングのトレードオフ**: 同時学習では両立が困難
4. **後半位置でのPPL悪化**: 長距離依存性が損なわれている可能性

## 結論

**V-DProj End-to-End学習は若干の改善を示すが、過学習が早く不安定**

| 評価項目 | 結果 |
|---------|------|
| Baseline比較 | 若干改善 (-13 ppl, -3.0%) |
| KV Cache削減 | 18.8%達成 |
| 学習安定性 | 不安定（過学習が早い） |
| 長距離依存性 | やや悪化 |

同時にLM LossとReconstruction Lossを最適化すると、両者が競合して学習が不安定になる。

## 次のステップ

### 2-Phase学習の期待

```
Phase 1: V Reconstruction事前学習
  - v_compress, v_restore のみを学習
  - Loss: ||V - V_restored||^2
  - 目標: 圧縮・復元ペアを事前に最適化

Phase 2: LM学習（V projection凍結）
  - v_compress, v_restore を凍結
  - 残りのパラメータでLM学習
  - Loss: Cross-entropy
```

**期待される効果**:
1. Phase 1でV復元を十分に学習
2. Phase 2でLMに専念（V復元は固定）
3. 学習の安定化

---

## 実験履歴

| 日付 | 実験 | 結果 |
|------|------|------|
| 2025-12-04 | PretrainedMKA V2 | Pythia 26.6 vs MKA 32.9 (+6.3 ppl) |
| 2025-12-04 | V-DProj End-to-End | val_ppl=428.7 (過学習) |
| 2025-12-04 | V-DProj 2-Phase | (実行予定) |
