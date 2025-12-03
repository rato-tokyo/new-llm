# Current Architecture

**最終更新**: 2025-12-03

---

## 概要

2-Block Cascade Context アーキテクチャ。各ブロックが異なるデータで学習することで、異なる表現を獲得する。

```
Token Embedding (GPT-2, 768-dim, frozen)
    │
    ▼
ContextBlock A (cd=256) ← 前半データで学習
    │
    ▼
ContextBlock B (cd=256) ← 後半データで学習
    │
    ▼
concat(context_a, context_b) = 512-dim
    │
    ▼
TokenBlock (1層)
    │
    ▼
Output Head (Weight Tying)
```

---

## スケーリング則

### 飽和モデル（Saturation Model）

```
PPL = PPL_min + A × n^(-a)
```

| 構成 | PPL_min | 1600 samples PPL | R² |
|------|---------|------------------|-----|
| 2-block (p=0) | **95.4** | 127.4 | 0.9995 |
| 2-block (p=2) | **87.3** | 114.7 | 0.9985 |

**重要**: 理論限界値 PPL_min が存在し、データ増加だけでは突破不可。

### 実験データ（p=0）

| Samples | Val PPL | Val Acc | ER% |
|---------|---------|---------|-----|
| 200 | 258.3 | 20.5% | 78% |
| 400 | 187.6 | 22.0% | 78% |
| 800 | 150.5 | 23.5% | 77% |
| 1600 | 127.4 | 24.5% | 78% |
| 3200 | 114.8 | 25.0% | 78% |

---

## prev_context_steps

連続履歴（interval=1）が有効。大きなintervalは悪化。

| 構成 | combined_dim | 1600 PPL | PPL_min |
|------|--------------|----------|---------|
| p=0 | 512 | 127.4 | 95.4 |
| p=1 | 1024 | 118.2 | ~85-90 |
| p=2 | 1536 | 114.7 | 87.3 |

**注意**: interval=8は効果なし（悪化）。interval=1のみ推奨。

---

## 設計原則

### 1. 1層固定

| Config | Val PPL | Val Acc |
|--------|---------|---------|
| **C1T1** | **127.2** | **24.7%** |
| C2T2 | 132.2 | 24.4% |
| C1T2 | 300.8 | 17.4% |

**C1T2（Token層だけ深い）は絶対に避ける**

### 2. Embedding凍結 + Weight Tying

| 設定 | Val PPL | パラメータ |
|------|---------|-----------|
| Embedding学習 | 1189 | 49.2M |
| **Embedding凍結** | **334** | **7.09M** |

### 3. Early Stopping 90%

Phase 1の収束率は90%で停止。99%は過収束で悪化。

### 4. OACD損失

```python
def oacd_loss(contexts, centroid_weight=0.1):
    context_mean = contexts.mean(dim=0)
    deviation = contexts - context_mean
    dispersion_loss = -torch.norm(deviation, p=2) / len(contexts)
    centroid_loss = torch.norm(context_mean, p=2) ** 2
    return dispersion_loss + centroid_weight * centroid_loss
```

---

## 推奨設定

```python
context_dim = 256  # per block (2-block: 512 total)
early_stopping_threshold = 0.90
phase2_freeze_embedding = True
use_weight_tying = True
phase2_epochs = 20
phase2_patience = 3
```

---

## 実験コマンド

```bash
# 基本実験
python3 scripts/experiment_cascade_context.py -s 2000

# prev_context付き
python3 scripts/experiment_cascade_context.py -s 2000 --prev-interval 1 --prev-count 2
```

---

*Last Updated: 2025-12-03*
