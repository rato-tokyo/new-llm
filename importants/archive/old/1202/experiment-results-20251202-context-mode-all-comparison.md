# Context Mode 全比較実験結果 (2025-12-02)

## 実験概要

**目的**: 4つのContext Mode（E案/A案/F案/G案）の性能比較

**環境**:
- GPU: NVIDIA L4 (22.2GB)
- Samples: 2000
- Context dim: 500
- Layers: 2
- 検証データ: 22,723 tokens
- 訓練データ: 2,403,563 tokens

**Context Modes**:
- **E案 (layerwise)**: TokenBlock Layer i は ContextBlock Layer i の出力を参照
- **A案 (final_only)**: 全TokenBlockレイヤーが ContextBlock の最終出力のみを参照
- **F案 (first_layer_only)**: 1層目のみに最終contextを注入、2層目はcontextなし
- **G案 (prev_and_current)**: 1層目に前トークンのcontext、最終層に現在のcontextを注入

---

## 結果サマリー

| Config | Mode | Context構造 | Params | Phase1 | Conv% | ER% | Val PPL | Val Acc | Total Time |
|--------|------|------------|--------|--------|-------|-----|---------|---------|------------|
| **L2_E** | E案 | Layer1→Layer1, Layer2→Layer2 | 41.8M | 14 iter | 93% | 79.7% | **128.1** | **24.9%** | 865s |
| L2_G | G案 | Layer1→prev, Layer2→current | 41.8M | 14 iter | 93% | 79.7% | **132.2** | 24.4% | 743s |
| L2_A | A案 | Layer1→final, Layer2→final | 41.8M | 14 iter | 93% | 79.7% | 136.9 | 24.6% | 982s |
| L2_F | F案 | Layer1→final, Layer2→none | 41.4M | 14 iter | 93% | 79.6% | 137.9 | 24.4% | 1068s |

**Winner**: E案 (layerwise) - PPL 128.1, Acc 24.9%

---

## ランキング

### Val PPL（低いほど良い）
1. **E案**: 128.1 (ベスト)
2. **G案**: 132.2 (+4.1, +3.2%)
3. **A案**: 136.9 (+8.8, +6.9%)
4. **F案**: 137.9 (+9.8, +7.7%)

### Val Acc（高いほど良い）
1. **E案**: 24.9% (ベスト)
2. **A案**: 24.6% (-0.3%)
3. **G案**: 24.4% (-0.5%)
4. **F案**: 24.4% (-0.5%)

---

## 各Modeの詳細

### E案 (layerwise) - ベスト
```
TokenBlock Layer 1 ← ContextBlock Layer 1 出力
TokenBlock Layer 2 ← ContextBlock Layer 2 出力

Phase 2結果:
  Best Epoch: 11
  Val PPL: 128.1
  Val Acc: 24.9%
  Train PPL: 81.7
```

**特徴**: 各TokenBlockレイヤーが対応する深さの文脈情報を受け取る

### G案 (prev_and_current) - 2位
```
TokenBlock Layer 1 ← 前トークン時点のcontext
TokenBlock Layer 2 ← 現在トークン時点のcontext

Phase 2結果:
  Best Epoch: 9
  Val PPL: 132.2
  Val Acc: 24.4%
  Train PPL: 79.7
```

**特徴**: 時間的に異なるcontextを使用。収束が速い（9エポック）

### A案 (final_only) - 3位
```
TokenBlock Layer 1 ← ContextBlock 最終出力
TokenBlock Layer 2 ← ContextBlock 最終出力（同じ）

Phase 2結果:
  Best Epoch: 13
  Val PPL: 136.9
  Val Acc: 24.6%
  Train PPL: 87.7
```

**特徴**: シンプルだが中間レイヤー情報を失う

### F案 (first_layer_only) - 4位
```
TokenBlock Layer 1 ← ContextBlock 最終出力
TokenBlock Layer 2 ← context入力なし

Phase 2結果:
  Best Epoch: 15
  Val PPL: 137.9
  Val Acc: 24.4%
  Train PPL: 86.9
```

**特徴**: 2層目にcontextなし。パラメータ数が約40万少ない（41.4M vs 41.8M）

---

## 分析

### 1. E案が最も優れている理由

**階層的な文脈表現の活用**:
- 浅いレイヤー: 局所的・具体的な文脈
- 深いレイヤー: 大域的・抽象的な文脈
- E案はこの階層構造を保持

```
E案: TokenBlock Layer i ← ContextBlock Layer i
     → 各深度で適切な抽象度の文脈情報
```

### 2. G案が2位になった理由

**時間的context差分の活用**:
- 1層目: 「直前の状態」を参照（過去の文脈）
- 2層目: 「現在の状態」を参照（更新後の文脈）
- 変化情報を暗黙的に捉えられる

**収束が最速（9エポック）**:
- prev/current contextの差が学習の手がかりに

### 3. A案とF案の差が小さい理由

**F案のcontext削減の影響は限定的**:
- F案は2層目のcontext入力をなくしても、PPLは+1程度
- → 2層目ではcontext依存度が低い可能性

**A案（全レイヤーで同じcontext）の冗長性**:
- 同じ情報を繰り返し入力しても効果が薄い

---

## Context入力パターンの比較

| Layer | E案 | A案 | F案 | G案 |
|-------|-----|-----|-----|-----|
| Layer 1 | ctx_layer1 | ctx_final | ctx_final | ctx_prev |
| Layer 2 | ctx_layer2 | ctx_final | none | ctx_current |
| **情報量** | 高 | 中 | 低 | 中 |

---

## 結論と推奨

### 推奨順位

1. **E案 (layerwise)**: 最高性能。デフォルトとして推奨
2. **G案 (prev_and_current)**: 時間的文脈が重要な場合に有効
3. **A案/F案**: 特に優位性なし

### 使用場面

| Mode | 推奨場面 |
|------|---------|
| E案 | 一般用途、最高精度が必要な場合 |
| G案 | シーケンス予測、時系列データ |
| A案 | シンプルさ重視（ただしE案が全指標で優位） |
| F案 | パラメータ削減が重要な場合（約40万減） |

### 設定方法

```python
# E案（推奨・デフォルト）
model = LLM(..., use_final_context_only=False)

# G案
model = LLM(..., use_prev_and_current_context=True)

# A案
model = LLM(..., use_final_context_only=True)

# F案
model = LLM(..., use_first_layer_context_only=True)
```

---

## 生データ

### L2_E (E案)
```
PPL=128.1, Acc=24.9%, ER=79.7%
Phase 1: 14 iter, 172.9s
Phase 2: 11 epochs, 692.5s
Total: 865.4s
```

### L2_G (G案)
```
PPL=132.2, Acc=24.4%, ER=79.7%
Phase 1: 14 iter, 178.3s
Phase 2: 9 epochs, 565.0s
Total: 743.4s (最速)

Epoch-by-epoch:
  Epoch 1: train_ppl=424.0 val_ppl=210.2 acc=20.9%
  Epoch 5: train_ppl=106.9 val_ppl=137.0 acc=24.0%
  Epoch 9: train_ppl=79.7  val_ppl=132.2 acc=24.4% ★ Best
  Epoch 10: → Early stop
```

### L2_A (A案)
```
PPL=136.9, Acc=24.6%, ER=79.7%
Phase 1: 14 iter, 176.8s
Phase 2: 13 epochs, 805.3s
Total: 982.1s

Epoch-by-epoch:
  Epoch 1:  train_ppl=438.9 val_ppl=220.2 acc=20.3%
  Epoch 5:  train_ppl=132.4 val_ppl=147.8 acc=23.6%
  Epoch 10: train_ppl=97.8  val_ppl=137.8 acc=24.3%
  Epoch 13: train_ppl=87.7  val_ppl=136.9 acc=24.6% ★ Best
  Epoch 14: → Early stop
```

### L2_F (F案)
```
PPL=137.9, Acc=24.4%, ER=79.6%
Phase 1: 14 iter, 179.8s
Phase 2: 15 epochs, 888.5s
Total: 1068.3s

Epoch-by-epoch:
  Epoch 1:  train_ppl=449.9 val_ppl=223.6 acc=20.2%
  Epoch 5:  train_ppl=135.8 val_ppl=150.9 acc=23.4%
  Epoch 10: train_ppl=101.5 val_ppl=140.0 acc=24.1%
  Epoch 15: train_ppl=86.9  val_ppl=137.9 acc=24.4% ★ Best
  Epoch 16: → Early stop
```

---

## 1層モデルとの比較（参考）

| Config | Layers | Mode | Val PPL | Val Acc |
|--------|--------|------|---------|---------|
| L1_F1 | 1 | E案 | 127.2 | 24.7% |
| L2_E | 2 | E案 | 128.1 | 24.9% |
| L2_G | 2 | G案 | 132.2 | 24.4% |
| L2_A | 2 | A案 | 136.9 | 24.6% |
| L2_F | 2 | F案 | 137.9 | 24.4% |

**注目点**: 1層モデル(L1_F1)がPPLではわずかに良い(127.2 vs 128.1)が、Accは2層E案が上(24.9% vs 24.7%)

---

## 採用決定 (2025-12-02)

### G案に一本化

**決定**: G案（prev_and_current_context）を採用し、E案/A案/F案は削除予定。

**理由**:
1. **メモリ効率**: E案はレイヤー数倍のキャッシュが必要
2. **拡張性**: G案は3層以上に自然に拡張可能
3. **メンテナンス性**: 複数Mode維持のコストが高い
4. **精度差は許容範囲**: PPL +3.2%、Acc -0.5%

**E案が理論上最良であることは認識**:
- 将来的に必要になった場合は「オンデマンド計算」で対応可能
- この決定はCLAUDE.mdにも記載

---

*Last Updated: 2025-12-02*
*Experiments: context_mode_f_experiment.py, context_mode_g_experiment.py*
*Decision: G案採用、E案/A案/F案削除予定*
