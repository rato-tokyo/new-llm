# Cascade Context Architecture

**最終更新**: 2025-12-03

---

## 概要

Cascade Context は、2つの ContextBlock (A, B) をカスケード連結し、高次元コンテキストを効率的に生成するアーキテクチャです。

### 構造図

```
Token Embedding (GPT-2, 768-dim, frozen)
    │
    ▼
┌─────────────────────────────────────────┐
│  Phase 1: Context Learning (OACD)       │
├─────────────────────────────────────────┤
│  ContextBlock A (cd=500)                │
│    ├─ 入力: token_embeds (前半データ)   │
│    └─ 出力: context_a (500-dim)         │
│                  │                      │
│                  ▼                      │
│  ContextBlock B (cd=500)                │
│    ├─ 入力: context_a (後半データ)      │
│    └─ 出力: context_b (500-dim)         │
└─────────────────────────────────────────┘
    │
    ▼
concat(context_a, context_b) = 1000-dim
    │
    ▼
┌─────────────────────────────────────────┐
│  Phase 2: Token Prediction              │
├─────────────────────────────────────────┤
│  TokenBlock (cd=1000, 1層)              │
│    ├─ 入力: combined_context            │
│    └─ 出力: hidden_state                │
└─────────────────────────────────────────┘
    │
    ▼
Output Head (Weight Tying with Embedding)
    │
    ▼
Next Token Prediction
```

---

## 性能

| 指標 | 値 | 比較 (C1T1-500) |
|------|-----|-----------------|
| **Val PPL** | **111.9** | -12% |
| **Val Acc** | **25.6%** | +0.9% |
| 実効次元 | 736/1000 (73.6%) | - |
| パラメータ | 41.2M | +1M |
| 処理時間 | 1745s | +45% |

---

## 設計原則

### 1. 1層固定

各ブロックは1層のみ。multi-layer は廃止（複雑化しても改善なし）。

| Config | Val PPL | Val Acc |
|--------|---------|---------|
| **C1T1** | **127.2** | **24.7%** |
| C2T2 | 132.2 | 24.4% |
| C1T2 | 300.8 | 17.4% |

**C1T2（Token層だけ深い）は絶対に避ける**

### 2. context_dim = 500

単一で1000にするより、500×2のカスケードが効率的。

| context_dim | Val PPL | ER% |
|-------------|---------|-----|
| **500** (単体) | **127.2** | **79.7%** |
| 1000 (単体) | 134.0 | 69.3% |
| **500×2** (Cascade) | **111.9** | 73.6% |

### 3. Early Stopping 90%

Phase 1 の収束率は90%で停止。99%は過収束で悪化。

| 閾値 | Val PPL | 備考 |
|------|---------|------|
| **90%** | **127.2** | 最適 |
| 99% | 235.1 | 過収束 |

### 4. Embedding凍結 + Weight Tying

| 設定 | Val PPL | パラメータ |
|------|---------|-----------|
| Embedding学習 | 1189.15 | 49.2M |
| **Embedding凍結** | **334.31** | **7.09M** |

---

## 2段階学習

### Phase 1: Context Learning

- **学習対象**: ContextBlock A, B（順次学習）
- **損失関数**: OACD（多様性損失のみ）
- **停止条件**: 収束率90%

```python
# ContextBlock A: 前半データで学習
context_a = context_block_a(token_embeds[:split])

# ContextBlock B: 後半データで学習、入力は context_a
context_b = context_block_b(context_a[split:])
```

### Phase 2: Token Prediction

- **ContextBlock**: frozen（重み固定）
- **学習対象**: TokenBlock のみ
- **損失関数**: CrossEntropy

```python
# コンテキスト連結
combined_context = torch.cat([context_a, context_b], dim=-1)  # 1000-dim

# TokenBlock で処理
output = token_block(combined_context, token_embeds)
```

---

## OACDアルゴリズム

Phase 1 で使用する唯一の多様性損失アルゴリズム。

```python
def oacd_loss(contexts, centroid_weight=0.1):
    """Origin-Anchored Centroid Dispersion"""
    context_mean = contexts.mean(dim=0)
    deviation = contexts - context_mean

    # 分散最大化（重心からの偏差を最大化）
    dispersion_loss = -torch.norm(deviation, p=2) / len(contexts)

    # 重心固定（重心を原点に引き寄せ）
    centroid_loss = torch.norm(context_mean, p=2) ** 2

    return dispersion_loss + centroid_weight * centroid_loss
```

### 特徴

- **分散最大化**: 重心からの偏差を最大化 → 多様な表現
- **重心固定**: 重心を原点に引き寄せ → 安定した平衡点
- **計算コスト**: O(n×d) で高速

---

## スケーリング則

### ベストモデル: 飽和モデル（Saturation Model）

```
PPL = PPL_min + A × n^(-a)
```

AIC比較の結果、飽和モデルが最も適合する（AIC=7.12）。

### PPL_min（理論的下限）

| 構成 | PPL_min | 1600 samples PPL | R² |
|------|---------|------------------|-----|
| 2-block (p=0) | **95.4** | 127.4 | 0.9995 |
| 2-block (p=2) | **87.3** | 114.7 | 0.9985 |

**重要な発見**:
- **飽和モデルが最適**（AIC最小、外挿予測も高精度）
- **理論限界値 PPL_min が存在**（データ増加では突破不可）
- **prev_context_steps で限界値自体が改善**（p=0の95.4 → p=2の87.3）

詳細は [SCALING_LAW_ANALYSIS.md](SCALING_LAW_ANALYSIS.md) を参照。

---

## 推奨設定

```python
# config.py
context_dim = 500  # per block
num_input_tokens = 1
early_stopping_threshold = 0.90
phase2_freeze_embedding = True
use_weight_tying = True

# Phase 2
phase2_epochs = 20
phase2_patience = 3
phase2_batch_size = 512
```

---

## 実験スクリプト

```bash
# Cascade Context 実験
python3 scripts/experiment_cascade_context.py -s 2000
```

---

## 避けるべき設定

- **C1T2**: Token層だけ深い → 性能崩壊
- **context_dim = 1000 (単体)**: 500×2より効率悪い
- **early_stopping = 0.99**: 過収束で悪化

---

*Last Updated: 2025-12-03*
