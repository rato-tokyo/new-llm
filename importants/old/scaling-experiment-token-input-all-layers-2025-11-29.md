# スケーリング実験: token継ぎ足し方式 + Embedding凍結 (2025-11-29)

## 実験概要

**目的**: token継ぎ足し方式（token_input_all_layers=True）でのスケーリング則（α値）を測定

**設定**:
| 項目 | 値 |
|------|-----|
| アーキテクチャ | E案（token継ぎ足し方式） |
| モデル | 6層 / 768次元 |
| Embedding凍結 | **True** |
| GPU | NVIDIA L4 (22.2GB) |
| サンプル数 | 50, 100, 200, 500 |
| シード | 42 |

---

## 実験結果

### サマリー

| Samples | Tokens | Val PPL | Val Acc | Val ER | Phase 1 Iter | Time |
|---------|--------|---------|---------|--------|--------------|------|
| 50 | 56,602 | 787.3 | 17.8% | 71.7% | 7 | 0.5min |
| 100 | 110,516 | 648.8 | 19.2% | 73.3% | 7 | 1.1min |
| 200 | 216,119 | 482.4 | **21.1%** | 74.0% | 7 | 1.5min |
| 500 | 529,173 | **324.5** | 19.0% | 75.9% | 7 | 3.4min |

### スケーリング則

```
PPL = A × tokens^α

α = -0.403 (R² = 0.992)
```

**解釈**: 2倍のデータで約24% PPL削減

---

## 比較分析

### 等差減少設計との比較（同一500サンプル）

| 設計 | Val PPL | Val Acc | Val ER |
|------|---------|---------|--------|
| **token継ぎ足し** | **324.5** | **19.0%** | **75.9%** |
| 等差減少 | 536.0 | 15.4% | 8.6% |
| **改善率** | **-39%** | **+23%** | **+782%** |

### α値の比較

| 設計 | α値 | 備考 |
|------|-----|------|
| **token継ぎ足し** | **-0.403** | 今回実験 |
| 等差減少 | -0.459 | 11/29実験 |

**発見**: token継ぎ足しのα値は等差減少より小さい（-0.403 vs -0.459）

これは一見、スケーリング効率が悪いように見えるが、**絶対的なPPL値は常にtoken継ぎ足しが優れている**。

---

## 重要な発見

### 1. token継ぎ足しは絶対性能で圧倒的

同じ500サンプルで：
- PPL: 324.5 vs 536.0（**39%改善**）
- Acc: 19.0% vs 15.4%（**23%向上**）

### 2. Effective Rankが大幅向上

token継ぎ足し方式では：
- Train ER: 75-77%（等差減少は8-9%）
- Val ER: 72-76%（等差減少は7-9%）

### 3. Phase 1の収束が安定

全サンプル数で**7イテレーションで100%収束**。

### 4. 200→500でAccuracyが低下

| Samples | Val Acc | Best Epoch |
|---------|---------|------------|
| 200 | **21.1%** | 4 |
| 500 | 19.0% | 3 |

**原因推定**:
- 500サンプルでEpoch 3で早期停止（200はEpoch 4まで）
- 訓練データが多いとモデルが急速にフィットし、早期停止が早すぎる
- `phase2_patience`を増やすことで改善可能

---

## Phase 2詳細

### 50サンプル
```
Epoch 1: train_ppl=16643.4 val_ppl=2558.5 acc=8.7%
Epoch 2: train_ppl=768.9   val_ppl=1440.7 acc=9.5%
...
Epoch 8: train_ppl=42.3    val_ppl=787.3  acc=17.8% ★Best
Epoch 9: train_ppl=31.5    val_ppl=790.9  acc=18.2%
→ Early stop
```

### 100サンプル
```
Epoch 1: train_ppl=5746.4 val_ppl=1594.7 acc=8.8%
...
Epoch 8: train_ppl=32.3   val_ppl=648.8  acc=19.2% ★Best
Epoch 9: train_ppl=24.5   val_ppl=684.5  acc=19.3%
→ Early stop
```

### 200サンプル
```
Epoch 1: train_ppl=2567.7 val_ppl=779.1 acc=15.6%
...
Epoch 4: train_ppl=110.4  val_ppl=482.4 acc=21.1% ★Best
Epoch 5: train_ppl=72.7   val_ppl=517.1 acc=21.3%
→ Early stop
```

### 500サンプル
```
Epoch 1: train_ppl=1037.7 val_ppl=433.1 acc=15.8%
Epoch 2: train_ppl=219.9  val_ppl=331.7 acc=18.3%
Epoch 3: train_ppl=111.0  val_ppl=324.5 acc=19.0% ★Best
Epoch 4: train_ppl=69.8   val_ppl=351.9 acc=19.2%
→ Early stop
```

---

## PPL予測表

`PPL = A × tokens^α` (α = -0.403, A ≈ 195,000)

| Train Tokens | 予測 PPL | 備考 |
|--------------|----------|------|
| 56,602 | 787 | 実測値 |
| 110,516 | 649 | 実測値 |
| 216,119 | 482 | 実測値 |
| 529,173 | 325 | 実測値 |
| 1,000,000 | **245** | 予測 |
| 5,000,000 | **130** | 予測 |
| 10,000,000 | **98** | 予測 |

---

## 推奨事項

### 1. token継ぎ足し方式を標準採用

等差減少設計と比較して：
- 絶対性能が大幅に優れている（PPL 39%改善）
- Effective Rankも大幅向上（8% → 76%）
- 収束も安定（7イテレーションで100%）

### 2. 大規模データでは早期停止を調整

500サンプル以上では：
- `phase2_patience = 2` 以上に設定
- または固定エポック数で訓練

### 3. 1000サンプル以上での検証

予測PPL 245（1Mトークン）を実測で確認する価値あり。

---

## 実験環境

- **GPU**: NVIDIA L4 (22.2GB)
- **Framework**: PyTorch
- **Data**: UltraChat (HuggingFaceH4/ultrachat_200k)
- **Phase 1**: CVFP固定点学習（max 40 iterations, early stopping）
- **Phase 2**: Next-token prediction（10 epochs, early stopping patience=1）
- **Total Time**: 6.8 min

---

## 関連ファイル

- [key-findings-2025-11-29.md](key-findings-2025-11-29.md) - token継ぎ足しが本質的という発見
- [embedding-freeze-experiment-2025-11-27.md](embedding-freeze-experiment-2025-11-27.md) - Embedding凍結実験
- [scaling-experiment-analysis-2025-11-29.md](scaling-experiment-analysis-2025-11-29.md) - 等差減少設計のスケーリング分析

---

Last Updated: 2025-11-29
