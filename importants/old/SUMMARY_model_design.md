# モデル設計まとめ (Model Design Summary)

**最終更新**: 2025-12-02

---

## 1. 採用アーキテクチャ: Cascade Context

### 構造

```
Token Embedding (GPT-2, 768-dim, frozen)
    ↓
ContextBlock A (cd=500, 1層) ← 前半データで学習
    ↓
ContextBlock B (cd=500, 1層) ← 後半データで学習、入力=A出力
    ↓
concat(context_a, context_b) = 1000-dim
    ↓
TokenBlock (cd=1000, 1層)
    ↓
Output Head (Weight Tying with Embedding)
```

### 特徴

- **1層固定**: multi-layer廃止（複雑化しても改善なし）
- **cd=500×2連結**: 単一cd=1000より高効率
- **Embedding凍結**: GPT-2事前学習を活用

---

## 2. Embedding凍結 - 必須

| 指標 | Embedding学習 | Embedding凍結 | 改善率 |
|------|--------------|--------------|--------|
| Val PPL | 1189.15 | **334.31** | **-71.9%** |
| Val Acc | 11.58% | **18.88%** | **+63.0%** |
| 学習パラメータ | 49.2M | **7.09M** | **-85.6%** |

**理由**:
- 過学習抑制
- GPT-2の意味表現を保持
- Weight Tyingとの相乗効果

---

## 3. Weight Tying - 必須

Token EmbeddingとOutput Headで重みを共有。

| 項目 | Without | With |
|------|---------|------|
| パラメータ | 91.43M | **52.78M** (-42%) |
| Output Head | 38.65M | **0** (共有) |

---

## 4. レイヤー数

### 結論: 1層が最適

| Config | Val PPL | Val Acc |
|--------|---------|---------|
| **C1T1** | **127.2** | **24.7%** |
| C2T2 | 132.2 | 24.4% |
| C1T2 | 300.8 | 17.4% |

**C1T2は絶対に避ける**（Token層だけ深いと性能崩壊）

---

## 5. 2段階学習

### Phase 1: Context学習

- ContextBlock A/B を順次学習
- 多様性損失（OACD）のみ
- 収束率90%で停止

### Phase 2: Token学習

- ContextBlock固定（frozen）
- TokenBlockのみ学習
- CrossEntropy損失

---

## 6. 推奨設定

```python
# Embedding
phase2_freeze_embedding = True
use_weight_tying = True

# アーキテクチャ
context_dim = 500  # per block
num_input_tokens = 1

# Phase 1
early_stopping_threshold = 0.90

# Phase 2
phase2_epochs = 20
phase2_patience = 3
phase2_batch_size = 512
```

---

*Last Updated: 2025-12-02*
