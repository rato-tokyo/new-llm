# HierarchicalLayer Expansion Gate - アーカイブ

**日付**: 2025-12-08
**状態**: 削除済み（アーカイブ）
**削除理由**: MultiMemoryLayerと実質同等の機能のため

---

## 概要

HierarchicalLayerは「情報不足かどうかを判定するゲート」を持っていた。
このゲート（expansion_gate）は、粗いメモリ検索で十分か、細かいメモリ検索が必要かを判断する。

---

## アーキテクチャ

```
Query
  ↓
Coarse Memory (全メモリの合計) で検索
  ↓
expansion_gate: 「この結果で十分か？」を判定
  ↓
  ├─ 十分 (prob低い) → Coarse結果を使用
  └─ 不十分 (prob高い) → Fine Memory (個別メモリ) から検索
  ↓
最終出力 = prob * Fine + (1-prob) * Coarse
```

---

## 実装詳細

### expansion_gate の構造

```python
self.expansion_gate = nn.Sequential(
    nn.Linear(hidden_size, hidden_size // 4),  # 512 → 128
    nn.ReLU(),
    nn.Linear(hidden_size // 4, 1),             # 128 → 1
)
```

**パラメータ数**: 65,793 (hidden_size=512の場合)
- 全体パラメータの約0.09%

### 使用方法

```python
def forward(self, hidden_states, ...):
    # 1. Coarse検索
    M_coarse, z_coarse = self._get_coarse_memory()  # 全メモリの合計
    output_coarse = self._retrieve_from_memory(sigma_q, M_coarse, z_coarse)

    # 2. expansion_gateで判定
    coarse_repr = output_coarse.view(batch_size, seq_len, hidden_size)
    expansion_prob = torch.sigmoid(self.expansion_gate(coarse_repr))
    # expansion_prob: (batch, seq, 1) - 各位置で0〜1の値

    # 3. Fine検索（必要に応じて）
    output_fine = self._retrieve_fine_grained(sigma_q)

    # 4. 混合
    memory_output = expansion_prob * output_fine + (1 - expansion_prob) * output_coarse
```

### ゲートの学習

- **入力**: Coarse検索結果（粗いメモリからの出力）
- **出力**: 0〜1のスカラー（各位置ごと）
- **学習**: 言語モデリングのlossを通じてend-to-endで学習
- **意図**: Coarse結果が不十分な場合に高い値を出力し、Fine検索を重視

---

## 実験結果（2025-12-06）

| モデル | Best PPL | Reversal Backward PPL | 改善率 |
|--------|----------|----------------------|--------|
| Single Memory | 105.9 | 604.5 | - |
| Multi Memory (4) | 105.8 | 508.6 | 16% |
| **Hierarchical (4)** | **105.6** | **437.8** | **27.6%** |

Hierarchicalが最も良い結果を示したが、PPL差は0.3とほぼ同等。

---

## 削除理由の詳細

1. **Fine検索がMultiMemoryと同じ**:
   - HierarchicalのFine検索 = 全メモリを個別に検索 → 加重平均
   - MultiMemoryの検索 = 全メモリを個別に検索 → 加重平均
   - **完全に同じ方式**

2. **Coarse検索の付加価値が不明確**:
   - Coarse = 全メモリの合計で検索
   - Fine = 全メモリを個別に検索
   - expansion_gateで混合するが、実質的にはFine結果が支配的になりやすい

3. **Top-K選択がない**:
   - HSA論文では「関連するチャンクをTop-Kで選択」が核心
   - HierarchicalLayerは全メモリを参照（スパースではない）
   - 計算効率の改善がない

---

## 今後の活用可能性

expansion_gateのコンセプトは以下の場面で有用かもしれない：

### 1. メモリ情報の十分性判定

```
「圧縮メモリの情報で十分か？」を判定
  ├─ 十分 → メモリからの出力を使用
  └─ 不十分 → 外部知識検索/より詳細な検索
```

### 2. 知識と思考の分離における使用

```
Query: "フランスの首都は？"
  ↓
Compressed Memory から検索
  ↓
gate: 「この情報で回答可能か？」
  ├─ 可能 (prob高い) → メモリ情報を使用
  └─ 不可能 (prob低い) → "I don't know" or 追加検索
```

### 3. 実装時のポイント

- ゲートの入力は「検索結果の表現」
- sigmoid出力で連続的な混合（離散的な判定はNG）
- パラメータ数は少なくてOK（0.1%程度）

---

## 参考コード（削除済み）

完全な実装は git history から復元可能：

```bash
git show HEAD~1:src/models/layers/hierarchical.py
```

または以下のコミットを参照：
- コミット: `29b3940` 以前

---

Last Updated: 2025-12-08
