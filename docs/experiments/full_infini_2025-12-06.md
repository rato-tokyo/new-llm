# Full Infini-Attention 実験結果

**実験日時**: 2025-12-06
**GPU**: NVIDIA L4 (23.8GB)

---

## 実験設定

| パラメータ | 値 |
|-----------|-----|
| サンプル数 | 10,000 |
| シーケンス長 | 512 |
| エポック数 | 50 (Early stopping: patience=1) |
| 学習率 | 1e-4 |
| バッチサイズ | 8 |
| Delta Rule | 有効 |

### データ

- **訓練データ**: 9,330サンプル (Pile: 9,000 + Reversal pairs: 330)
- **検証データ**: 1,000サンプル (Pileのみ、Reversal pairsなし)

---

## アーキテクチャ比較

| モデル | 構成 | パラメータ数 |
|--------|------|-------------|
| **Pythia** | 全6層 RoPE | 70,420,480 |
| **Full Infini** | 全6層 Infini-Attention (NoPE) | 70,408,240 |

### Full Infiniアーキテクチャ

```
Full Infini-Attention Model:
Token Embedding (512-dim)
       ↓
Layer 0-5: InfiniAttentionLayer (NoPE, Memory Only + Causal Linear Attention)
  ├─ Memory Attention: 過去セグメントから取得
  ├─ Causal Linear Attention: 現在セグメント内でO(n)アテンション
  └─ Learnable Gate: メモリとローカルを統合
       ↓
Output Head (512 → vocab)
```

**メモリサイズ**: 798,720 bytes (6層合計、固定)

---

## 結果サマリー

### Perplexity (PPL)

| モデル | Best PPL | Best Epoch | 訓練時間/epoch |
|--------|----------|------------|----------------|
| Pythia (RoPE) | **73.3** | 7 | ~270秒 |
| Full Infini | 107.6 | 9 | ~538秒 |

**差分: +34.3 PPL (46.8%劣化)**

### 訓練速度

| モデル | 時間/epoch | 相対速度 |
|--------|-----------|---------|
| Pythia | 270秒 | 1.0x (基準) |
| Full Infini | 538秒 | **0.50x (2倍遅い)** |

---

## 学習曲線

### Pythia (RoPE baseline)

| Epoch | Train PPL | Val PPL | 備考 |
|-------|-----------|---------|------|
| 1 | 503.6 | 271.4 | * |
| 2 | 154.9 | 156.3 | * |
| 3 | 92.0 | 115.0 | * |
| 4 | 63.2 | 93.5 | * |
| 5 | 46.3 | 82.9 | * |
| 6 | 35.0 | 75.8 | * |
| 7 | 26.9 | **73.3** | * Best |
| 8 | 20.8 | 73.4 | Early stop |

### Full Infini (全6層)

| Epoch | Train PPL | Val PPL | 備考 |
|-------|-----------|---------|------|
| 1 | 606.5 | 354.5 | * |
| 2 | 210.2 | 221.2 | * |
| 3 | 134.1 | 169.9 | * |
| 4 | 96.9 | 143.6 | * |
| 5 | 74.2 | 127.4 | * |
| 6 | 58.7 | 117.3 | * |
| 7 | 47.4 | 110.3 | * |
| 8 | 38.8 | 108.1 | * |
| 9 | 32.0 | **107.6** | * Best |
| 10 | 26.6 | 109.9 | Early stop |

**観察**:
- Full Infiniは2エポック多く学習が必要 (9 vs 7)
- 学習初期でPPL差が大きい (354.5 vs 271.4)
- 収束後も34.3 PPLの差が残る

---

## Position-wise PPL

| Position | Pythia | Full Infini | 差分 | 変化率 |
|----------|--------|-------------|------|--------|
| 0-16 | 154.3 | 201.4 | +47.2 | +30.6% |
| 16-32 | 85.9 | 124.0 | +38.2 | +44.4% |
| 32-64 | 76.4 | 110.2 | +33.8 | +44.3% |
| 64-96 | 69.6 | 98.3 | +28.7 | +41.2% |
| 96-512 | 70.9 | 107.8 | +36.9 | +52.0% |

**観察**:
- **全ポジションでFull Infiniが劣る**
- 特に後半 (96-512) で差が大きい (+52.0%)
- 序盤 (0-16) でも劣化 (+30.6%)

---

## Reversal Curse 評価

| モデル | Forward PPL | Backward PPL | Ratio | Gap |
|--------|-------------|--------------|-------|-----|
| Pythia | 1.7 | 461.8 | 0.004 | +460.1 |
| **Full Infini** | 1.8 | **408.1** | 0.004 | **+406.4** |

**改善率**: Backward PPL 11.6%改善 (461.8 → 408.1)

### 解釈

- **Forward PPL**: 両モデルとも1.7-1.8（訓練データの順方向文は完全に学習）
- **Backward PPL**: Full Infiniが53.7ポイント改善（逆方向推論が向上）
- **Reversal Gap**: Full Infiniは53.7ポイント改善 (460.1 → 406.4)

---

## 分析

### 性能劣化の要因

1. **位置エンコーディングの欠如 (NoPE)**
   - Full Infiniは全層で位置情報を持たない
   - Pythiaは全層でRoPEを使用し、トークン位置を正確に把握
   - 言語モデルでは語順が重要なため、位置情報の欠如は致命的

2. **Linear Attentionの表現力**
   - Softmax Attentionに比べて表現力が低い
   - φ(Q)・φ(K)^T ≠ softmax(Q・K^T/√d)
   - 特に鋭いアテンションパターンの再現が困難

3. **6層すべてでの制約**
   - 1層のみInfini（Infini-Pythia）では影響が限定的
   - 全層でLinear Attentionは累積的に性能低下

### 訓練速度の問題

| 操作 | 計算量 | 備考 |
|------|--------|------|
| Softmax Attention | O(n²) | 最適化されたFlash Attention |
| Causal Linear Attention | O(n) | einsum演算、累積和 |
| Delta Rule メモリ更新 | O(n×d²) | 追加コスト |

**速度低下の理由**:
- einsum操作はFlash Attentionほど最適化されていない
- 6層すべてでメモリ更新・取得を実行
- Delta Ruleによる追加計算

### Reversal Curse改善の要因

| 要因 | 説明 |
|------|------|
| 圧縮メモリ | 順序に依存しない情報蓄積 |
| 位置情報の欠如 | 逆方向でも同じメモリ構造 |
| 6層の圧縮効果 | 1層よりも強い双方向性 |

---

## 比較: Full Infini vs Infini-Pythia

| 項目 | Infini-Pythia (1層) | Full Infini (6層) |
|------|---------------------|-------------------|
| 構成 | Infini×1 + Pythia×5 | Infini×6 |
| PPL | ~105 | 107.6 |
| 速度 | Pythia同等 | 0.5x |
| Reversal Gap | ~400 | 406.4 |
| 位置情報 | RoPE (Layer 1-5) | なし |

**結論**: 1層のみInfiniの方がバランスが良い

---

## 結論

### 課題

1. **PPL劣化**: +34.3 PPL (46.8%悪化)
2. **訓練速度**: 2倍遅い
3. **全ポジションで劣化**: 位置情報の欠如が致命的

### 利点

1. **Reversal Curse改善**: Backward PPL 11.6%改善
2. **固定メモリ**: 798KB (シーケンス長に依存しない)
3. **理論上の長文対応**: O(n) 計算量

### 示唆

1. **全層Infiniは現時点で実用的でない**
   - 位置情報の欠如による性能劣化が大きすぎる
   - ALiBi等の位置エンコーディング導入が必要

2. **1層Infini + RoPE (Infini-Pythia) が推奨**
   - PPL改善 + Reversal Curse改善
   - 訓練速度もPythiaと同等

3. **今後の改善案**
   - 全層ALiBi付きFull Infiniの実験
   - より高速なLinear Attention実装（CUDA最適化）
   - Softmax Attentionとの混合アーキテクチャ

---

## 実行コマンド

```bash
# Full Infini実験
python3 scripts/experiment_infini.py \
    --all-infini \
    --samples 10000 \
    --seq-length 512 \
    --epochs 50

# ALiBi付きFull Infini（今後の実験）
python3 scripts/experiment_infini.py \
    --all-infini \
    --alibi \
    --samples 10000 \
    --seq-length 512 \
    --epochs 50
```

---

## 参考: 実験ログ

```
Device: cuda (NVIDIA L4, 23.8GB)
Samples: 10,000
Sequence length: 512
Epochs: 50
Learning rate: 0.0001
Delta rule: True
All Infini: enabled (6 layers)

1. PYTHIA (RoPE baseline)
   Total parameters: 70,420,480
   Best: epoch 7, ppl=73.3

2. FULL INFINI (6 Layers)
   Layer 0-5: Infini-Attention (Memory Only)
   Position encoding: None (NoPE)
   Total parameters: 70,408,240
   Total memory: 798,720 bytes (6 layers)
   Best: epoch 9, ppl=107.6

Difference: +34.3 ppl
```
