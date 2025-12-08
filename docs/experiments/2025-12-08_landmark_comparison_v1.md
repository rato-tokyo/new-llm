# Landmark方式比較実験 v1

**日付**: 2025-12-08
**結論**: 差がほぼ出ず（+0.1 PPL）。実験設計に問題あり。

---

## 実験概要

MultiMemoryLayerのLandmark計算方式を比較：

| 方式 | 定義 | 備考 |
|------|------|------|
| `memory_norm` | Σ σ(k) | 書き込み操作の副産物 |
| `learned` | Linear(hidden) | 学習可能な射影 |

**仮説**: HSA方式（mean(K)）に近い`learned`が優れるはず

---

## 実験設定

```
Device: NVIDIA L4 (23.8GB)
Samples: 5,000
Seq length: 256
Epochs: 30
Memories: 4
Early stopping patience: 5
```

---

## 結果

| Landmark Type | Best PPL | Epoch | Params |
|---------------|----------|-------|--------|
| memory_norm   | **105.4** | 7 | 70,418,440 |
| learned       | 105.6 | 7 | 70,422,536 |

**PPL差**: +0.1（ほぼ同等）
**追加パラメータ**: 4,096

---

## 詳細ログ

### memory_norm

```
Epoch  1: train= 1025.7, val=  435.5 *
Epoch  2: train=  253.0, val=  242.1 *
Epoch  3: train=  133.9, val=  173.5 *
Epoch  4: train=   83.6, val=  137.9 *
Epoch  5: train=   56.1, val=  119.3 *
Epoch  6: train=   39.0, val=  108.8 *
Epoch  7: train=   27.6, val=  105.4 *  ← Best
Epoch  8: train=   19.7, val=  106.3
...
Epoch 12: Early stopping
```

### learned

```
Epoch  1: train= 1018.3, val=  430.3 *
Epoch  2: train=  251.0, val=  237.8 *
Epoch  3: train=  132.9, val=  170.0 *
Epoch  4: train=   82.4, val=  138.7 *
Epoch  5: train=   55.1, val=  119.4 *
Epoch  6: train=   38.3, val=  109.1 *
Epoch  7: train=   27.0, val=  105.6 *  ← Best
Epoch  8: train=   19.1, val=  106.9
...
Epoch 12: Early stopping
```

---

## 分析：なぜ差が出なかったか

### 1. メモリ数が少ない（4個）

選択肢が4つしかないため、どのLandmark方式でも同じメモリが選ばれやすい。

```
理想: 16+ memories → 選択精度が性能に直結
現実: 4 memories → 選択ミスの影響が小さい
```

### 2. メモリ内容の均一性

ランダムなPileテキストでは、各メモリに格納される内容が似通う：

```
Memory 0: [一般的なテキスト]
Memory 1: [一般的なテキスト]
Memory 2: [一般的なテキスト]
Memory 3: [一般的なテキスト]
→ Landmarkが似てしまい、選択の意味がない
```

### 3. タスクの単純さ

一般的な言語モデリングでは、「どのメモリから取り出すか」より「メモリを使うか使わないか」の方が重要。

---

## 今後の改善案

### A. メモリ数を増やす

```python
# 4 → 16 memories
model = create_model("multi_memory", num_memories=16)
```

### B. 意味的に異なるメモリを事前構築

```python
# 異なるドメインのテキストを各メモリに格納
memory_sources = [
    "scientific_papers.txt",    # Memory 0: 科学
    "news_articles.txt",        # Memory 1: ニュース
    "fiction_books.txt",        # Memory 2: フィクション
    "code_documentation.txt",   # Memory 3: コード
]
```

### C. メモリ選択が重要なタスク設計

```
クエリ: "量子力学の基礎方程式は？"
→ 正解メモリ: 科学メモリ
→ 誤選択: ニュースメモリ → PPL大幅悪化

Landmark精度がPPLに直結するタスク設計が必要
```

---

## 結論

この実験では差が出なかったが、**実験設計の問題**であり、Landmark方式自体の優劣は判断できない。

**次のステップ**:
1. 異なるドメインのテキストで各メモリを事前構築
2. メモリ選択の精度が性能に直結するタスクで再実験

---

## 採用方針

HSA方式（mean(K)）に一本化する方針は維持：

- **理論的根拠**: Landmarkが「メモリの内容」を直接表現
- **メンテナンス性**: 単一方式でコードがシンプルに
- **追加パラメータ不要**: learned方式より効率的

差が出なかったのは実験設計の問題であり、HSA方式の採用判断には影響しない。

---

## 追加実験: Memory Selection Accuracy

### 実験設計

ランダム初期化モデルで、異なるドメインのテキストを各メモリに格納し、
クエリに対して正しいメモリを選択できるかを評価。

```
ドメイン:
- science: 物理、量子力学、相対性理論など
- history: 世界大戦、ルネサンス、産業革命など
- technology: 機械学習、Python、クラウドなど
- geography: エベレスト、アマゾン川、サハラ砂漠など

各ドメイン5クエリ × 4ドメイン = 20クエリ
```

### 結果

| ドメイン | 正解率 |
|----------|--------|
| science | 0% (0/5) |
| history | 0% (0/5) |
| technology | 40% (2/5) |
| geography | 40% (2/5) |
| **全体** | **20% (4/20)** |

### 分析

- **ランダム（25%）とほぼ同等**: ランダム初期化モデルでは意味的な選択ができない
- **Landmarkは異なる値を持つ**（5.19〜6.22）が、意味的な区別には不十分
- **事前学習が必要**: 意味的なメモリ選択には、事前学習済みモデルの使用または fine-tuning が必要

### 実装

```bash
# 実験スクリプト
python3 scripts/experiment_memory_selection.py

# MemoryBuilderユーティリティ
src/utils/memory_builder.py
```
