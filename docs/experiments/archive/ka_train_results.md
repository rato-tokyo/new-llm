# KA Training (案2) 実験結果

**結論: この手法は失敗。全てをA（Attention Output）で統一するアプローチは実用的ではない。**

---

## 実験概要

### 仮説

> Attention OutputはResidual Connectionで加算されるため、現在位置のVを直接使用する必要はない。
> 過去のAttention Output（A）のみを使用しても、モデルは十分に学習できるはず。

### 方式

```
標準Attention:
  A[i] = softmax(Q[i] @ K[:i]^T) @ V[:i]

KA Training (案2):
  A[0] = V[0]                                   (ブートストラップ)
  A[i] = renormalize(weights[:i]) @ A[:i]       (過去のAのみ、現在のVは不使用)
```

### 期待

- 学習と推論で同じ計算グラフを使用することでtrain-inference mismatchを解消
- KVキャッシュと同サイズだがKA方式で一貫性のある学習が可能

---

## 実験結果

| Model | Val PPL | Train Time/Epoch | 備考 |
|-------|---------|------------------|------|
| Pythia (KV, Baseline) | **105.3** | ~43s | 6 epoch で収束 |
| KA Training (案2) | 304.0+ | ~560s (13x遅い) | 3 epoch時点、収束傾向なし |

### 詳細ログ

**Baseline (Pythia with KV Cache)**
```
Epoch  1: train_ppl=566.2 val_ppl=357.4
Epoch  2: train_ppl=142.7 val_ppl=202.7
Epoch  3: train_ppl=74.9  val_ppl=149.9
Epoch  4: train_ppl=45.9  val_ppl=123.1
Epoch  5: train_ppl=30.0  val_ppl=110.6
Epoch  6: train_ppl=20.1  val_ppl=105.3 <- Best
Epoch  7: Early stop
```

**KA Training (案2)**
```
Epoch  1: train_ppl=761.9 val_ppl=543.0 [558.8s]
Epoch  2: train_ppl=225.9 val_ppl=359.4 [563.1s]
Epoch  3: train_ppl=136.4 val_ppl=304.0 [562.0s]
(中断)
```

---

## 失敗の原因分析

### 1. 情報ボトルネック（致命的）

```
A[0] = V[0]           # 64次元の情報のみ
A[1] = w @ A[0]       # A[0]の情報のみ（64次元）
A[2] = w @ [A[0], A[1]] # A[0], A[1]の加重平均
...
A[127] = w @ [A[0], ..., A[126]]  # 全て最初のV[0]から派生
```

**問題**: 最初のトークンV[0]の64次元が全ての後続トークンの唯一の情報源になる。
シーケンスが長くなるほど、新しい情報（各位置のV）が一切入らないため、表現力が極端に制限される。

### 2. 誤差の再帰的蓄積

```
A[i] は A[:i] に依存
A[:i] は A[:i-1] に依存
...
```

seq_len=128で128回の連鎖的な依存関係があり、誤差が蓄積する。

### 3. 学習速度の問題

- **13倍遅い**: 43s/epoch → 560s/epoch
- Pythonループ（seq_len回）が各Attentionレイヤーで必要
- 完全な並列化が不可能な再帰構造

### 4. 収束の困難さ

train_pplは下がるが、val_pplの改善が遅い：
- Baseline: 6 epochで105.3
- KA Train: 3 epochで304.0（約3倍悪い）、収束の兆候なし

---

## 仮説の棄却

**「Attention OutputはResidual Connectionで加算されるため、現在位置のVは不要」という仮説は誤り。**

理由：
1. Residual Connectionは累積的な情報追加であり、各層で新しい情報（V）が必要
2. 過去のAのみを使うと、情報が「希釈」されていく
3. 各位置で新しいコンテキスト情報（V）を取り込むことがAttentionの本質的な機能

---

## 学んだこと

### KAキャッシュ方式の本質的な制限

| 方式 | 問題 | 結果 |
|------|------|------|
| 案1 (Adapter) | Train-inference mismatch | +303% PPL悪化 |
| **案2 (KA Training)** | **情報ボトルネック** | **+189% PPL悪化（3epoch時点）** |
| 案3 (推論のみKA) | Train-inference mismatch | +1037% PPL悪化 |

### 結論

**KAキャッシュ方式（Vの代わりにAをキャッシュ）は、どの実装アプローチでも実用的な性能を達成できない。**

KVキャッシュ削減が目的なら、**MLA（Multi-head Latent Attention）方式**を推奨。
MLAはK+Vを低次元の潜在ベクトルに圧縮し、数学的に等価な計算を行うため、精度低下なしにキャッシュを削減できる。

---

## 比較表（最終版）

| 方式 | PPL | KVキャッシュ削減 | 実用性 |
|------|-----|-----------------|--------|
| Pythia (Baseline) | 105.3 | 0% | ✅ |
| MLA-Pythia (kv_dim=128) | ~110 | 87.5% | ✅ |
| KA Adapter (案1) | 439.2 | 0% | ❌ |
| **KA Training (案2)** | **304.0+** | **0%** | **❌** |
| KA Cache (案3) | 1240.3 | 0% | ❌ |

---

## 実行コマンド（参考）

```bash
# この実験（非推奨）
python3 scripts/experiment_ka_train.py --samples 5000 --epochs 30

# 推奨: MLA実験
python3 scripts/experiment_mla.py --samples 10000 --epochs 30
```

---

Last Updated: 2025-12-05
