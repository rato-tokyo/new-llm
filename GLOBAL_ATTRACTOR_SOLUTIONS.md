# グローバルアトラクター問題の解決策

## 問題の根本原因の分析

### 原因1: Simple Overwrite Updater（最も重要）

```python
class SimpleOverwriteUpdater:
    def forward(self, hidden, context):
        new_context = torch.tanh(self.context_update(hidden))
        return new_context  # ← 前の文脈を完全に無視
```

**問題点**:
- `hidden`のみから文脈を生成（前の`context`を参照しない）
- 同じトークンを繰り返すと、同じ`hidden`が生成される
- 結果として同じ`new_context`が生成される
- トークン固有性が保持されない

### 原因2: 文脈の正規化による均質化

```python
context = self.context_norm(context)  # LayerNorm
context = torch.clamp(context, min=-10.0, max=10.0)
```

**問題点**:
- LayerNormがすべての文脈ベクトルを同じ分布に正規化
- L2ノルムが15.887付近に固定される
- トークン間の差異が縮小される

### 原因3: 訓練データの単純性

**現在の訓練**:
- 単一トークンの繰り返し（`[token] * N`）
- 実際のテキストシーケンスではない
- 文脈の多様性が学習されない

---

## 解決策の候補

### 📋 候補1: Gated Context Updaterへの切り替え（推奨度: ★★★★★）

#### 内容

```python
class GatedAdditiveUpdater:
    def forward(self, hidden, context):
        context_delta = torch.tanh(self.context_update(hidden))
        forget_gate = torch.sigmoid(self.forget_gate(hidden))
        input_gate = torch.sigmoid(self.input_gate(hidden))

        # 前の文脈を保持しながら更新
        new_context = forget_gate * context + input_gate * context_delta
        return new_context
```

#### メリット

✅ **前の文脈を保持**: `forget_gate * context`で過去情報を維持
✅ **LSTM実績**: LSTMで実証済みのメカニズム
✅ **実装済み**: `src/models/components/context_updaters.py`に既存
✅ **最小限の変更**: `config.context_update_strategy = 'gated'`だけ

#### デメリット

⚠️ パラメータ数が増加（3倍）
⚠️ 訓練時間が若干増加

#### 実装手順

```bash
# train.pyで訓練
python3 train.py \
    --context-update-strategy gated \
    --max-samples 10000 \
    --epochs 10 \
    --batch-size 32 \
    --lr 5e-4 \
    --device cpu
```

#### 期待される効果

- トークンごとに異なる固有点を学習
- 文脈の多様性が保持される
- CVFPTの本来の効果を検証可能

---

### 📋 候補2: 文脈への直接的なトークン埋め込み注入（推奨度: ★★★★☆）

#### 内容

```python
class TokenAwareContextUpdater:
    def __init__(self, hidden_dim, context_dim, embed_dim):
        self.context_update = nn.Linear(hidden_dim, context_dim)
        self.token_projection = nn.Linear(embed_dim, context_dim)

    def forward(self, hidden, context, token_embed):
        # hiddenからの更新
        delta_from_hidden = torch.tanh(self.context_update(hidden))

        # トークン埋め込みを直接注入
        token_influence = self.token_projection(token_embed)

        # 両方を組み合わせ
        new_context = delta_from_hidden + 0.3 * token_influence
        return new_context
```

#### メリット

✅ トークン固有情報が確実に保持される
✅ Simple Updaterより軽量（Gatedより少ないパラメータ）
✅ 解釈性が高い（トークンの影響が明示的）

#### デメリット

⚠️ `forward()`のシグネチャ変更が必要
⚠️ 新しいアプローチ（実績が少ない）
⚠️ トークン埋め込みの依存度が高い

#### 実装の複雑度

**中程度**（`new_llm.py`と`context_updaters.py`の両方を修正）

---

### 📋 候補3: 残差接続の追加（推奨度: ★★★☆☆）

#### 内容

```python
class ResidualContextUpdater:
    def forward(self, hidden, context):
        delta = torch.tanh(self.context_update(hidden))

        # 残差接続で前の文脈を保持
        new_context = context + 0.1 * delta

        # 正規化
        new_context = torch.tanh(new_context)  # [-1, 1]に収める
        return new_context
```

#### メリット

✅ Simple Updaterへの最小限の変更
✅ パラメータ数は同じ
✅ 実装が簡単

#### デメリット

⚠️ 勾配消失の可能性（長いシーケンスで）
⚠️ 正規化との相性が悪い（残差が消える可能性）
⚠️ Gatedほど柔軟でない

#### 効果

**不確実**（残差が正規化で消える可能性あり）

---

### 📋 候補4: LayerNormの除去または緩和（推奨度: ★★☆☆☆）

#### 内容

```python
# 現在
context = self.context_norm(context)  # LayerNorm
context = torch.clamp(context, min=-10.0, max=10.0)

# 修正案A: LayerNormを除去
context = torch.clamp(context, min=-10.0, max=10.0)

# 修正案B: BatchNormに変更（バッチ全体で正規化）
context = self.context_norm(context)  # BatchNorm
```

#### メリット

✅ トークン間の差異が保持される
✅ 実装が簡単

#### デメリット

❌ 訓練の不安定化（勾配爆発のリスク）
❌ 文脈ベクトルのスケールが制御不能
❌ 根本解決にならない（Simple Updaterの問題は残る）

#### 推奨度が低い理由

**正規化は訓練安定性に重要** - 除去はリスクが高い

---

### 📋 候補5: 訓練データの改善（推奨度: ★★★★☆）

#### 内容

**現在の訓練**:
```python
# 単一トークンの繰り返し
input_ids = [token] * 100
```

**改善案**:
```python
# 実際のテキストシーケンス
input_ids = [token1, token2, token3, ..., tokenN]

# CVFPT用の混合訓練
# 1. 通常のテキストシーケンス（80%）
# 2. トークン繰り返し（20%） - CVFPT用
```

#### メリット

✅ 文脈の多様性が学習される
✅ 実際の使用シナリオに近い
✅ アーキテクチャ変更不要

#### デメリット

⚠️ 訓練データ準備が必要
⚠️ CVFPTの効果が薄れる可能性

#### 実装

既存の`train.py`は実際のWikiTextで訓練している → **すでに実装済み**

---

### 📋 候補6: 多様性損失の追加（推奨度: ★★★☆☆）

#### 内容

```python
def diversity_loss(context_vectors):
    """
    異なるサンプル間の文脈ベクトルの多様性を促進
    """
    batch_size = context_vectors.size(0)

    # バッチ内の平均文脈ベクトル
    mean_context = context_vectors.mean(dim=0)

    # 各文脈ベクトルと平均の距離
    distances = torch.norm(context_vectors - mean_context, dim=1)

    # 多様性損失（距離が大きいほど良い）
    diversity_loss = -distances.mean()

    return diversity_loss

# 訓練時
total_loss = token_loss + recon_loss + 0.01 * diversity_loss
```

#### メリット

✅ グローバルアトラクターを直接的に防ぐ
✅ 既存アーキテクチャと併用可能

#### デメリット

⚠️ ハイパーパラメータ調整が必要（重み）
⚠️ 過度な多様性が逆効果の可能性
⚠️ 計算コストが増加

---

## 推奨される解決策

### 🏆 第1推奨: Gated Context Updater（★★★★★）

**理由**:
1. ✅ **実証済み**: LSTMで長年使われている
2. ✅ **実装済み**: すぐに使える
3. ✅ **最小限の変更**: configを変えるだけ
4. ✅ **根本解決**: 前の文脈を保持する

**実装手順**:

```bash
# ステップ1: Gated Updaterで再訓練
python3 train.py \
    --context-update-strategy gated \
    --max-samples 10000 \
    --epochs 10 \
    --batch-size 32 \
    --lr 5e-4 \
    --device cpu

# ステップ2: チェックポイント保存
# → checkpoints/gated_model.pt

# ステップ3: グローバルアトラクターチェック
python3 scripts/check_global_attractor.py \
    --checkpoint checkpoints/gated_model.pt

# ステップ4: CVFPT実験再実行
python3 scripts/cvfpt_context_comparison.py \
    --checkpoint checkpoints/gated_model.pt \
    --num-tokens 100
```

**期待される結果**:
```
Pairwise L2 Distance: > 0.1（多様性あり）
Per-Dimension Variance: > 0.001（トークン固有性あり）
```

---

### 🥈 第2推奨: Gated + 訓練データ改善（★★★★★）

**組み合わせアプローチ**:

1. **Gated Context Updater**で文脈保持
2. **実際のテキストシーケンス**で訓練（既存のtrain.pyを使用）
3. **多様性チェック**を定期的に実施

**メリット**:
- 最も堅牢な解決策
- 実際の使用シナリオに対応

**実装**:

```bash
# WikiTextで通常訓練（Gated Updater使用）
python3 train.py \
    --context-update-strategy gated \
    --max-samples 50000 \
    --epochs 20 \
    --batch-size 32 \
    --lr 5e-4 \
    --device cpu \
    --output-dir checkpoints/gated_wikitext
```

---

### 🥉 第3推奨: 簡易版 - Residual + LayerNorm緩和（★★★☆☆）

**Gatedが重すぎる場合の代替案**:

```python
class LightweightResidualUpdater:
    def forward(self, hidden, context):
        delta = torch.tanh(self.context_update(hidden))

        # 残差接続（強め）
        new_context = 0.7 * context + 0.3 * delta

        return new_context  # LayerNormなし
```

**メリット**:
- Simple Updaterと同じパラメータ数
- 訓練が速い

**デメリット**:
- 効果が不確実
- 訓練の安定性が低い

---

## 実装の優先順位

### 即座に実行可能

1. ✅ **Gated Updaterで再訓練**（1時間以内）
2. ✅ **グローバルアトラクターチェック**（5分）

### 中期的に検討

3. ⏳ **訓練データの改善**（WikiText訓練は既存）
4. ⏳ **多様性損失の追加**（実験的）

### 長期的に検討

5. 🔮 **新しいUpdater設計**（トークン埋め込み注入など）

---

## まとめ

| 候補 | 推奨度 | 実装難度 | 期待効果 | 備考 |
|------|--------|---------|---------|------|
| **1. Gated Updater** | ★★★★★ | 低 | 高 | **最優先推奨** |
| 2. トークン埋め込み注入 | ★★★★☆ | 中 | 高 | 実験的 |
| 3. 残差接続 | ★★★☆☆ | 低 | 中 | 効果不確実 |
| 4. LayerNorm除去 | ★★☆☆☆ | 低 | 低 | リスク高 |
| **5. 訓練データ改善** | ★★★★☆ | 低 | 中 | Gatedと併用 |
| 6. 多様性損失 | ★★★☆☆ | 中 | 中 | 補助的 |

---

## 次のアクションプラン

### フェーズ1: 検証（今すぐ）

```bash
# Gated Updaterで小規模訓練（10分）
python3 train.py \
    --context-update-strategy gated \
    --max-samples 1000 \
    --epochs 5 \
    --batch-size 16 \
    --device cpu \
    --output-dir experiments/gated_test

# グローバルアトラクターチェック
python3 scripts/check_global_attractor.py \
    --checkpoint experiments/gated_test/final_model.pt
```

### フェーズ2: 本格訓練（問題が解決していたら）

```bash
# WikiTextで本格訓練
python3 train.py \
    --context-update-strategy gated \
    --max-samples 50000 \
    --epochs 20 \
    --batch-size 32 \
    --device cpu
```

### フェーズ3: CVFPT実験再実行

```bash
python3 scripts/cvfpt_context_comparison.py \
    --checkpoint checkpoints/gated_model.pt \
    --num-tokens 100
```

---

**結論**: まず**Gated Context Updater**で再訓練し、グローバルアトラクター問題が解決するか確認することを強く推奨します。
