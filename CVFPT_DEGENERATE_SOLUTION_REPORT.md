# CVFPT実験における退化解の診断レポート

## 🚨 重大な問題の発見

CVFPT（Context Vector Fixed Point Training）比較実験において、**グローバルアトラクター問題**による退化解を確認しました。

## 問題の詳細

### 発見された異常

実験結果は一見「非常に良好」に見えました：

| 指標 | 値 | 評価 |
|------|-----|------|
| L2 Distance | 0.588 | 低い（良好に見える） |
| Cosine Similarity | 0.999 | 非常に高い（良好に見える） |
| Convergence Steps | 1.0/10 | 超高速（良好に見える） |

しかし、**これは誤った実装による見かけ上の高精度でした**。

### 真の問題：グローバルアトラクター

詳細な診断の結果、以下が判明：

```
🚨 GLOBAL ATTRACTOR DETECTED!

Pairwise L2 Distance between different tokens' fixed points:
  Mean: 0.000002  ← ほぼゼロ（すべて同じ点）
  Std:  0.000001
  Min:  0.000000
  Max:  0.000006

Per-Dimension Variance:
  Mean variance: 0.000000
  Dimensions with near-zero variance: 256 / 256 (100%)
```

**すべてのトークンが同一の固有点に収束している。**

### 具体例

異なるトークンの収束後の文脈ベクトル（最初の10次元）：

```python
Token  1513 ('ted'):    [-1.0432844,  0.9477694, -1.0424911, ...]
Token  2237 (' six'):   [-1.0432844,  0.9477693, -1.0424911, ...]
Token  2612 (' heart'): [-1.0432844,  0.9477693, -1.0424910, ...]
Token  3623 (' gas'):   [-1.0432844,  0.9477693, -1.0424910, ...]
```

**完全に同一です。**

## 実験結果の再解釈

### 元の解釈（誤り）

- ✅ CVFPTが効果的
- ✅ モデルは固有点を学習している
- ✅ 単一パスでも固有点に近い

### 正しい解釈

- ❌ **モデルはトークン情報を無視している**
- ❌ すべてのトークンが同じ「グローバルアトラクター」に収束
- ❌ 文脈ベクトルがトークン固有の情報を保持していない

### なぜ「高精度」に見えたのか？

1. **Single-pass context**: トークン固有の初期文脈（Step 0）
2. **Fixed-point context**: グローバルアトラクター（Step 1以降、すべて同じ）

両者の距離（L2 = 0.588）は：
- トークンごとの初期状態 → グローバルアトラクターへの移動距離
- **すべてのトークンで同じ距離になる**（だから収束ステップが全部1.0）

## 根本原因の分析

### 使用したモデル

```
Checkpoint: checkpoints/new_llm_repetition_final.pt
Context updater: Simple Overwrite Updater
Training: Repetition training (繰り返し訓練)
```

### Simple Overwrite Updaterの問題

```python
class SimpleOverwriteUpdater(BaseContextUpdater):
    def forward(self, hidden, context):
        # Generate completely new context (overwrite old one)
        new_context = torch.tanh(self.context_update(hidden))
        return new_context  # ← 前の文脈を完全に無視
```

**前の文脈を完全に無視する更新戦略**

繰り返し訓練において：
- 同じトークンを繰り返す
- 毎回同じ `hidden` が生成される
- 同じ `new_context` が生成される
- 前の文脈を参照しないので、固有点は **入力トークンのみ** に依存

しかし、なぜグローバルアトラクターになったのか？

### 仮説：訓練中の文脈正規化の影響

```python
# new_llm.py の forward() 内
context = self.context_norm(context)  # LayerNorm
context = torch.clamp(context, min=-10.0, max=10.0)  # Clipping
```

繰り返し訓練で：
1. すべてのトークンが同じように正規化される
2. L2ノルムが 15.887 付近に固定される
3. トークン固有性が失われる

## 実験の欠陥

### 1. 収束判定の問題

```python
def analyze_convergence(trajectory):
    final_context = trajectory[-1, :]

    distances = []
    for t in range(len(trajectory)):
        dist = torch.norm(trajectory[t, :] - final_context).item()
        distances.append(dist)
```

**Step 1以降がすべて同じ**なので、当然すぐに収束する。

### 2. 比較方法の問題

- Fixed-point: すべて同じグローバルアトラクター
- Single-pass: トークン固有の初期文脈

両者を比較しても、**トークン固有性の有無を検証できていない**。

## 正しい検証方法

### 必須チェック1: トークンごとの固有点の多様性

```python
# 異なるトークンの固有点が異なるか確認
for token_id in random_tokens:
    fixed_point = get_fixed_point_context(model, token_id)
    # 固有点間の距離を計算
```

**今回の実験で欠けていた視点**

### 必須チェック2: 文脈ベクトルの次元ごとの分散

```python
# 256次元すべてで分散がゼロ → グローバルアトラクター
variance = np.var(all_fixed_points, axis=0)
```

## 教訓

### 1. 「高精度」の数値に疑問を持つ

- コサイン類似度 0.999 は異常に高い
- すべてのトークンで収束ステップが1.0 も異常
- **数値が良すぎる場合は実装バグを疑う**

### 2. 多様性のチェックを怠らない

CVFPT実験では：
- ✅ Single vs Repeated の比較
- ❌ **異なるトークン間の固有点の多様性チェック** ← 欠落していた

### 3. Simple Overwrite Updaterの限界

前の文脈を無視する戦略では：
- 固有点が入力トークンのみに依存
- 正規化の影響でグローバルアトラクターに収束しやすい

**Gated Updaterなど、文脈を保持する戦略が必要**

## 次のステップ

### 推奨される対応

1. **Gated Context Updaterで再訓練**
   ```python
   config.context_update_strategy = 'gated'
   ```

2. **訓練データの多様化**
   - 単一トークンの繰り返しだけでなく
   - 実際のテキストシーケンスで訓練

3. **診断メトリクスの追加**
   - トークン間の固有点の分散
   - 次元ごとの活性化パターン分析

4. **実験の再実行**
   - 新しいモデルでCVFPT実験を再実行
   - 多様性チェックを含める

## まとめ

CVFPT実験は以下の理由で**退化解**を示していました：

1. ✅ **実装自体はバグなし**（ロジックは正しい）
2. ❌ **モデルがグローバルアトラクターに収束**（訓練の問題）
3. ❌ **多様性チェックの欠如**（実験設計の問題）
4. ❌ **Simple Updaterの限界**（アーキテクチャの問題）

**結論**: CVFPTの概念自体は有効だが、現在のモデルは退化解に陥っており、再訓練が必要。

---

**作成日**: 2025-11-20
**診断実行スクリプト**:
- `scripts/debug_cvfpt.py`
- `scripts/check_global_attractor.py`
