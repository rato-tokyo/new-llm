# KAキャッシュ方式の比較

KVキャッシュの代替としてKA（Key + Attention Output）キャッシュを使用する3つの方式の比較。

---

## 用語定義

### Attention Output (A)

```python
A = softmax(Q @ K^T) @ V  # shape: [batch, heads, seq, head_dim]
```

Attention weightsとValue vectorsの重み付き和。

### KAキャッシュ

```
標準KVキャッシュ: K, V をキャッシュ
KAキャッシュ:     K, A をキャッシュ（Vの代わりにAttention Outputを保存）
```

---

## 3つの方式

### 案1: KA Cache + Adapter（学習時KV、推論時KA）

**ファイル**: `src/models/ka_adapter.py`, `scripts/experiment_ka_adapter.py`

```
学習時:
  A[i] = softmax(Q[i] @ K[:i]^T) @ V[:i]  ← 標準KV方式

推論時:
  A[i] = softmax(Q[i] @ K[:i]^T) @ [Adapter(A[1:i-1]), V[i]]
                                    ↑Adapterで変換
```

**特徴**:
- 学習と推論で異なる計算グラフ（train-inference mismatch）
- Adapterで補正を試みる
- 基本モデルは凍結、Adapterのみ学習

**問題点**:
- 学習時にVを使い、推論時にAdapterを使うため不整合が大きい
- AdapterがA→Vの完全な変換を学習できない

---

### 案2: KA Training（学習時からKA、全てA統一）

**ファイル**: `src/models/ka_train.py`, `scripts/experiment_ka_train.py`

```
学習時・推論時ともに:
  A[0] = V[0]                                   (ブートストラップ)
  A[i] = renormalize(weights[:i]) @ A[:i]       (過去のAのみ、現在のVは不使用)
```

**仮説**: Attention OutputはResidual Connectionで加算されるため、現在位置のVを直接使用する必要はない。

**特徴**:
- 学習と推論で同じ計算グラフ（一貫性あり）
- モデルがKA方式に最適化される
- 追加パラメータなし
- **現在位置のVを使わない**（全てAで統一）

**問題点**:
- 誤差の再帰的蓄積（後述）
- 学習時に autoregressive ループが必要（遅い）

---

### 案3: KA Cache（推論のみKA、学習不要）

**ファイル**: `src/models/ka_cache.py`, `scripts/experiment_ka_cache.py`

```
学習時:
  A[i] = softmax(Q[i] @ K[:i]^T) @ V[:i]  ← 標準KV方式

推論時:
  A[i] = softmax(Q[i] @ K[:i]^T) @ [A[1:i-1], V[i]]
                                    ↑過去はA  ↑現在はV
```

**特徴**:
- 追加学習なしで既存モデルをそのまま使用
- 推論時のみKAキャッシュに切り替え

**問題点**:
- 学習時にVを使い、推論時にAを使うため不整合（案1と同様）
- 誤差が蓄積

---

## 誤差蓄積の問題

KAキャッシュの根本的な問題は**誤差の再帰的蓄積**です。

```
位置 1: A[1] = weights @ V[1]                    ← V使用（正確）
位置 2: A[2] = weights @ [A[1], V[2]]            ← A[1]はV由来なのでOK
位置 3: A[3] = weights @ [A[1], A[2], V[3]]      ← A[2]がすでにAを含む
位置 4: A[4] = weights @ [A[1], A[2], A[3], V[4]] ← 誤差蓄積が加速
...
位置 i: A[i] = weights @ [A[1], ..., A[i-1], V[i]] ← 大きな誤差
```

各 A[j] は過去の A[1:j-1] に依存し、その過去も更に過去に依存する。
この再帰構造により、位置が進むほど誤差が蓄積する。

---

## 実験結果比較

### 案1: KA Cache + Adapter

| Method | PPL | KV比 |
|--------|-----|------|
| KV Cache (baseline) | 109.0 | - |
| KA Cache (no adapter) | 1240.3 | +1037% |
| KA Cache (with adapter) | 439.2 | +303% |

**結論**: Adapterで64.6%改善するも、依然+303%悪い。実用困難。

---

### 案2: KA Training

（実験実行待ち）

**期待される結果**:
- 以前の類似実験（KA-Attention V→A置換）では +7 ppl 程度
- 学習と推論の一貫性により、案1/案3より大幅に良い結果を期待

---

### 案3: KA Cache（学習不要）

| Method | PPL | KV比 |
|--------|-----|------|
| KV Cache (baseline) | 109.0 | - |
| KA Cache | 1240.3 | +1037% |

**結論**: 推論のみの変更では大幅な精度低下。使用不可。

---

## 比較表

| 項目 | 案1 (Adapter) | 案2 (KA Training) | 案3 (推論のみ) |
|------|--------------|-------------------|----------------|
| 学習方式 | KV | KA | KV |
| 推論方式 | KA + Adapter | KA | KA |
| 追加パラメータ | あり（Adapter） | なし | なし |
| Train-Inference一貫性 | ❌ 不一致 | ✅ 一致 | ❌ 不一致 |
| 追加学習 | Adapterのみ | End-to-End | なし |
| 期待PPL悪化 | +303% | +数% | +1037% |
| 実用性 | ❌ | △ 検証中 | ❌ |

---

## 理論的考察

### なぜ案2が最も有望か

1. **一貫性**: 学習と推論で同じ計算グラフを使用
2. **最適化**: モデルがKA方式の誤差を吸収するよう学習
3. **シンプル**: 追加パラメータなし

### 案2の課題

1. **学習速度**: autoregressive ループが必要で遅い
2. **誤差蓄積**: 長いシーケンスで精度低下の可能性
3. **KVキャッシュ削減効果**: K+AとK+Vでサイズは同じ

---

## KVキャッシュ削減の観点

| キャッシュ方式 | 保存内容 | サイズ | 削減率 |
|---------------|----------|--------|--------|
| KV Cache | K, V | 2 × hidden_size | 0% |
| KA Cache | K, A | 2 × hidden_size | 0% |
| MLA (kv_dim=128) | c_kv | kv_dim | 87.5% |

**重要**: KAキャッシュはKVキャッシュと**同じサイズ**。
KVキャッシュ削減が目的なら、**MLA方式**の方が適切。

KAキャッシュの潜在的メリット:
- Aは過去の集約情報を含むため、**再計算不要**の場面がある可能性
- しかし、現在の実験では精度低下が大きく、メリットを確認できていない

---

## 実行コマンド

```bash
# 案1: KA Cache + Adapter
python3 scripts/experiment_ka_adapter.py --samples 5000 --epochs 10

# 案2: KA Training
python3 scripts/experiment_ka_train.py --samples 5000 --epochs 30

# 案3: KA Cache (推論のみ)
python3 scripts/experiment_ka_cache.py --samples 5000 --epochs 30
```

---

## 結論

| 方式 | 推奨度 | 理由 |
|------|--------|------|
| **案1 (Adapter)** | ❌ | Train-inference mismatchで+303%悪化 |
| **案2 (KA Training)** | △ | 検証中。一貫性により改善を期待 |
| **案3 (推論のみ)** | ❌ | +1037%悪化。使用不可 |

KVキャッシュ削減が目的なら、KA方式ではなく**MLA方式**を推奨。

---

Last Updated: 2025-12-05
