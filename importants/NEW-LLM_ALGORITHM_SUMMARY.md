# New-LLM Algorithm Summary

**外部Claude相談用サマリー - 前提知識不要版**

---

## 🎯 プロジェクト概要

New-LLMは、**CVFP (Context Vector Fixed-Point)** という新しい手法を用いた言語モデルアーキテクチャです。従来のTransformerとは異なり、固定点学習による文脈ベクトルの収束を利用してテキスト理解と生成を行います。

---

## 🧠 核心アルゴリズム: CVFP (Context Vector Fixed-Point)

### 基本原理

**固定点 (Fixed-Point) とは？**
- 関数 f において、f(x) = x となる点を「固定点」と呼ぶ
- New-LLMでは、文脈ベクトル c が繰り返し処理で収束する点を学習

**数式表現**:
```
c_{t+1} = c_t + FNN(concat(c_t, token_embed))
```
- c_t: イテレーション t での文脈ベクトル
- token_embed: トークンの埋め込みベクトル（GPT-2事前学習済み、768次元）
- FNN: Feed-Forward Network（残差接続付き）

---

## 📐 モデルアーキテクチャ

### 1. CVFPLayer（基本計算ユニット）

```
Input: context [768-dim] + token_embed [768-dim]
       ↓
FNN: [1536-dim hidden layer with ReLU]
       ↓
Split: → delta_context [768-dim] + delta_token [768-dim]
       ↓
Residual: new_context = context + delta_context
          new_token = token_embed + delta_token
       ↓
Optional: LayerNorm (layernorm_mix=1.0 で全適用)
```

**重要な制約**:
- `hidden_dim = context_dim + embed_dim` (1536 = 768 + 768) **必須**
- この制約により、FNN出力を正確に分割可能

### 2. CVFPBlock（複数層のグルーピング）

- 6個のCVFPLayerを順次実行
- 各層で文脈とトークン埋め込みを更新

### 3. LLMモデル全体構造

```
Token IDs
    ↓
Token Embedding (GPT-2 pretrained, frozen)
    ↓ [embedding normalized]
Context = zero-vector (初期化)
    ↓
┌─────────────────────────┐
│  CVFPBlock × 6 layers   │
│  (Context propagation)  │
└─────────────────────────┘
    ↓
Output Head: Linear(context_dim + embed_dim → vocab_size)
    ↓
Next Token Prediction
```

---

## 🔄 2段階学習プロセス

### Phase 1: CVFP固定点学習（並列処理版）

**目的**: 文脈ベクトルが収束する「固定点」を学習

**プロセス**:

1. **Iteration 0（シーケンシャル処理）**:
   - 目的: 固定点目標を確立
   - コンテキストをゼロから開始し、全トークンを順次処理
   - 出力を `target_contexts` として保存（**この値は以降不変**）

2. **Iteration 1～10（並列処理）**:
   - 目的: 固定点への収束（23倍高速化）
   - Token i には `previous_contexts[i-1]` を使用（1トークン分のずれ）
   - 全トークンをバッチ処理で並列計算

**損失関数**:

```python
# CVFP損失: 固定点目標への収束
cvfp_loss = MSE(contexts, target_contexts)

# 多様性損失: 全トークンの平均からの偏差を最大化
diversity_loss = -‖contexts - mean(contexts)‖₂ / num_tokens

# 総合損失（並列版最適設定: dist_reg_weight=0.9）
total_loss = 0.1 * cvfp_loss + 0.9 * diversity_loss
```

**重要ポイント**:
- ❌ **絶対禁止**: `target_contexts` を更新してはいけない
  - これは固定点学習の定義を破壊する致命的バグになる
- ✅ **必須**: イテレーション間でコンテキストを引き継ぐ
  - `context = previous_contexts[-1]` （ゼロリセット禁止）

**性能（並列版）**:
- Effective Rank: 55.9% (429/768次元) - 検証データ
- 処理時間: 約11秒（シーケンシャル版265秒の23倍高速）
- 収束率: 27.2%（多様性優先のため低めだが正常）

### Phase 2: Next-Token Prediction（トークン予測）- Context-Fixed Learning

**目的**: Phase 1で学習した文脈表現を使用して次トークン予測

**2段階処理**:

#### Stage 1: 初期化（パラメータ更新なし）
- Phase 2開始時に1回だけ実行
- 訓練データの全トークンを処理し、固定文脈ベクトルC*を生成
- **C*は以降絶対に変更しない**

```python
# 固定文脈C*の生成
C*[0] = CVFPブロック(token_embed[0], zero_vector).context_out
C*[1] = CVFPブロック(token_embed[1], C*[0]).context_out
...
```

#### Stage 2: 学習（パラメータ更新あり）
- 入力: `[C*[i-1], token_embed[i]]` - 固定文脈を使用
- 出力: `[context_out, token_out]` - CVFPブロックの出力
- **context_outはC*[i]で完全に置換**（MSE制約ではなく値そのもの）
- 予測: `logits = Linear(concat(C*[i], token_out))`

```python
for i in range(num_tokens):
    # 入力: 固定文脈C*[i-1]
    input_context = C_star[i-1] if i > 0 else zero_vector

    # CVFPブロック処理
    context_out, token_out = cvfp_block(input_context, token_embed[i])

    # context_outは使わず、C*[i]で完全置換
    combined = concat(C_star[i], token_out)
    logits = Linear(combined)

    # 損失は予測損失のみ
    loss = CrossEntropy(logits, target[i+1])
```

**勾配フロー**:
- ✅ token_out経由でCVFPブロックが更新される
- ❌ context_out経由の勾配は流れない（完全固定のため）
- ✅ token_outputは新規学習

**重要な設計変更（2025-11-26）**:
- ❌ **旧設計（v1.0）**: MSE制約による「緩い」固定
- ✅ **新設計（v2.0）**: context_outをC*[i]で完全置換（完全固定）
- **理由**: Phase 1で学習した文脈表現を確実に保護するため

---

## 📊 評価指標

### 1. Effective Rank（多様性指標）

**定義**: 特異値分布のエントロピーから計算される「実効的な次元数」

```python
# 特異値から計算
singular_values = svd(context_matrix)
probabilities = (singular_values / sum(singular_values))
entropy = -sum(p * log(p) for p in probabilities)
effective_rank = exp(entropy) / total_dimensions
```

**解釈**:
- 100%: 全次元が均等に使用されている（理想的）
- 55.9%: 768次元中、約429次元が実効的に使用（並列版）
- <30%: 次元が偏っている（多様性不足）

### 2. 収束率（Convergence Rate）

**定義**: MSE < threshold となったトークンの割合

```python
token_losses = MSE(context_t, context_{t-1}) per token
converged_tokens = count(token_losses < 0.1)
convergence_rate = converged_tokens / total_tokens
```

**解釈**:
- 並列版: 27.2% - 多様性優先のため低めだが正常
- 100%: 全トークンが収束（多様性とのトレードオフ）

### 3. CVFP損失

**定義**: 固定点目標との平均二乗誤差

```python
cvfp_loss = MSE(contexts, target_contexts)
```

---

## ⚡ 並列処理最適化

### 性能比較

| 指標 | シーケンシャル版 | 並列版 |
|------|----------------|--------|
| Effective Rank | 66.6% | 55.9% |
| 処理時間 | 265秒 | 11秒 |
| 高速化率 | 1x | **23x** |
| 収束率 | 30.0% | 27.2% |

### トレードオフ

**並列版の特徴**:
- ✅ 23倍高速化により実用性が大幅向上
- ✅ 55.9% ERは実用的な多様性を維持
- ⚠️ 1トークン分の情報遅延（Token i は previous_contexts[i-1] を使用）
- ✅ dist_reg_weight=0.9 により多様性強化で補償

---

## 🚨 過去の致命的バグと教訓

### Bug 1: 固定点目標の上書き（2025-11-25修正）

**問題**:
```python
# ❌ 間違い
Iteration 1: CVFP損失 = MSE(contexts_1, contexts_0)
             target = contexts_1  # 目標を上書き！
Iteration 2: CVFP損失 = MSE(contexts_2, contexts_1)  # 目標が変わっている
```

**正しい実装**:
```python
# ✅ 正解
Iteration 0: target = contexts_0  # 固定保存
Iteration 1: CVFP損失 = MSE(contexts_1, target)  # 固定目標
Iteration 2: CVFP損失 = MSE(contexts_2, target)  # 同じ目標
```

### Bug 2: コンテキストのゼロリセット（2025-11-24修正）

**問題**:
```python
# ❌ 間違い: 毎イテレーションでゼロリセット
for iteration in range(10):
    context = torch.zeros(...)  # 固定点学習が機能しない！
```

**正しい実装**:
```python
# ✅ 正解: イテレーション間で引き継ぎ
context = torch.zeros(...)  # 初回のみ
for iteration in range(10):
    if iteration > 0:
        context = previous_contexts[-1]  # 前回の最終値を引き継ぐ
```

### Bug 3: Phase 2での独立処理（2025-11-24修正）

**問題**:
```python
# ❌ 間違い: 各トークンが独立
for token in tokens:
    context = torch.zeros(...)  # 文脈情報が伝わらない
```

**正しい実装**:
```python
# ✅ 正解: 文脈伝播
context = torch.zeros(...)  # 最初のみ
for token in tokens:
    context = cvfp_forward(token, context)  # 文脈を引き継ぐ
```

---

## 🔧 データ仕様

**訓練データ**:
- ソース: UltraChat (HuggingFaceH4/ultrachat_200k)
- サンプル数: 50
- トークン数: 6400
- キャッシュ: `./cache/ultrachat_50samples_128len.pt`

**検証データ**:
- ソース: 訓練データの最後20%から生成
- トークン数: 1280
- ファイル: `./data/ultrachat_50samples_val.txt`
- **必須条件**: 全トークンが訓練データに存在すること

**重要**: `auto_split` は厳禁（エラー発生）

---

## 💡 核心的な設計原則

1. **固定点学習の本質**:
   - 目標（target_contexts）は絶対に不変
   - イテレーション間でコンテキスト引き継ぎ必須

2. **Phase 1とPhase 2の一貫性**:
   - 両方で文脈伝播が必須
   - Phase 2でゼロリセットは禁止

3. **多様性と収束のバランス**:
   - 並列版: dist_reg_weight=0.9（多様性90%, CVFP 10%）
   - Effective Rank維持が最優先

4. **勾配管理**:
   - トークン間: `context.detach()`（勾配遮断）
   - Phase 2: Context安定性損失で固定

---

## 📈 現在の性能（並列版ベースライン）

**Phase 1（検証データ）**:
- Effective Rank: **55.9%** (429/768次元)
- 処理時間: **約11秒** (23倍高速化)
- 収束率: 27.2%

**Phase 2（実装完了）**:
- 文脈伝播による一貫性確保
- Context + Token Embed予測
- 文脈安定性損失による固定

---

## 🎓 このアルゴリズムの独自性

1. **固定点学習**: 従来のAttentionではなく、反復収束による文脈理解
2. **残差接続**: ResNetスタイルの安定した学習
3. **多様性正則化**: 次元使用の偏りを防ぐ新しい手法
4. **2段階学習**: 文脈学習と予測を分離した効率的な訓練
5. **並列処理最適化**: 23倍高速化により実用性を実現

---

## 📝 相談時の重要情報

- **アーキテクチャ**: 6層CVFPブロック、768次元、GPT-2事前学習埋め込み
- **並列版性能**: 55.9% ER、11秒処理時間、23倍高速化
- **最重要設定**: dist_reg_weight=0.9（多様性優先）
- **データ規模**: 訓練6400トークン、検証1280トークン
- **再現性**: 乱数シード固定により完全な再現性保証

---

**作成日**: 2025-11-26
**並列処理版**: 完全採用（2025-11-25）
**主要ファイル**: src/models/llm.py, src/trainers/phase1.py, src/trainers/phase2.py
