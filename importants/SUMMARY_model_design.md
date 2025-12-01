# モデル設計・訓練手法まとめ (Model Design Summary)

**作成日**: 2025-12-01
**統合元ファイル**: embedding-freeze-experiment-2025-11-27.md, layer-and-noise-experiment-2025-11-26.md, phase2-context-fixed-design.md

---

## 1. Embedding凍結 (Embedding Freeze) - 推奨

### 結論: **小〜中規模データでは凍結を強く推奨**

| 指標 | Embedding学習 | Embedding凍結 | 改善率 |
|------|--------------|--------------|--------|
| Val PPL (500s) | 1189.15 | **334.31** | **-71.9%** |
| Val Acc (500s) | 11.58% | **18.88%** | **+63.0%** |
| Val PPL (1000s) | 840.46 | **280.27** | **-66.7%** |
| Val Acc (1000s) | 13.03% | **19.91%** | **+52.8%** |

### 学習パラメータ削減効果

| 設定 | 学習パラメータ | 削減率 |
|------|--------------|--------|
| Embedding学習 | ~49.2M | - |
| **Embedding凍結** | **7.09M** | **-85.6%** |

### 効果の理由

1. **過学習抑制**: パラメータ削減により汎化性能向上
2. **GPT-2事前学習の活用**: 大規模コーパスで学習済みの意味表現を保持
3. **Weight Tyingの相乗効果**: 入出力の一貫性を維持

### 推奨設定

```python
phase2_freeze_embedding = True  # 推奨
use_weight_tying = True          # 推奨
```

---

## 2. Weight Tying (重み共有) - 推奨

### 概要

Token EmbeddingとOutput Headで重みを共有（GPT-2と同様）

| 項目 | Without Weight Tying | With Weight Tying |
|------|---------------------|-------------------|
| 全体パラメータ | 91.43M | **52.78M** (-42%) |
| Output Head | 38.65M | **0** (共有) |

### 採用理由

1. **パラメータ効率**: 小〜中規模モデル（100M以下）で特に効果的
2. **業界標準**: GPT-2, GPT-3, BERT, LLaMA, Mistralなどで採用

### 注意点

- Embedding凍結時、Output Headも自動的に凍結される

---

## 3. レイヤー数の影響

### 結論: **現状では1層が最適**

num_layers=1を固定として運用中。

### 過去の比較実験（3層 vs 6層）

| 指標 | 3層 | 6層 | 差分 |
|------|-----|-----|------|
| Val ER | 69.5% | 61.4% | **-8.1%** |
| Best Val PPL | 774.75 | 995.97 | **+28.5%悪化** |
| Phase 1時間 | 46.5s | 79.4s | +70.8% |

**6層が悪化した理由**:
- Effective Rank低下
- データ量不足（64Kトークンでは6層には少なすぎる）
- 過学習加速

---

## 4. コンテキストノイズ（参考）

### 実験結果

| 指標 | ノイズなし | ノイズあり(0.1) | 効果 |
|------|-----------|----------------|------|
| 継続改善エポック数 | 1 | **6** | **+5エポック** |
| Train/Val PPL乖離 | 101x | **8.3x** | **12x改善** |
| Final Val Acc | 17.36% | **20.80%** | **+3.44%** |

### ノイズの効果

- **正則化効果**: 過学習を抑制
- **汎化性能向上**: より汎用的な固定点を学習
- **学習曲線の安定化**: 継続的な改善が可能

*注: 現在の実装では使用していない*

---

## 5. 2段階処理設計（Phase 2）

### Stage 1: C*の生成（パラメータ更新なし）

Phase 2開始時に1回だけ実行。固定文脈ベクトルC*を生成。

```python
with torch.no_grad():
    context = torch.zeros(...)
    C_star = []
    for token_id in token_ids:
        context = context_block(context, token_embed)
        C_star.append(context)
```

### Stage 2: 学習（パラメータ更新あり）

C*を使用してTokenBlockを学習。

```python
for i, token_id in enumerate(input_ids):
    input_context = C_star[i-1] if i > 0 else zero_vector
    token_out = token_block(input_context, token_embed)
    logits = token_output(token_out)
    loss = CrossEntropy(logits, target[i+1])
```

---

## 6. token継ぎ足し方式（token_input_all_layers）

### 結論: **True（全レイヤーでtoken入力）が本質的**

| 設定 | ER | Val PPL | Val Acc |
|------|-----|---------|---------|
| **token継ぎ足しあり** | 76.3% | **334** | **18.9%** |
| 等差減少（なし） | 8.6% | 536 | 15.4% |

- PPL **38%改善**、Acc **23%向上**
- ERはtoken継ぎ足しの副産物

---

## 7. 現在の推奨設定まとめ

```python
# 基本設定
num_layers = 1                    # 固定
context_dim = 500-768             # 最適範囲

# Embedding設定
phase2_freeze_embedding = True    # 推奨
use_weight_tying = True           # 推奨

# Phase 2設定
phase2_epochs = 10-20
phase2_patience = 2-3             # 早期停止
phase2_batch_size = 512

# token継ぎ足し
token_input_all_layers = True     # 推奨
```

---

*Last Updated: 2025-12-01*
