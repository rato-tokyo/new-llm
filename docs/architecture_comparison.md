# New-LLM Architecture Comparison

**Date**: 2025-11-21

このドキュメントでは、New-LLMの3つの主要アーキテクチャの違いを説明します。

---

## 概要

New-LLMには3つの異なるアーキテクチャファミリーがあります：

1. **Residual Architecture（残差結合型）** - ResNetスタイルの残差結合を使用
2. **Gated Architecture（ゲート機構型）** - LSTMスタイルのゲート更新を使用
3. **Context-Only Variants（コンテキスト専用型）** - 各アーキテクチャのContext-Only変種

---

## 1. Residual Architecture（残差結合型）

### 1.1 NewLLMResidual（Standard版）

**ファイル**: `src/models/new_llm_residual.py`

**特徴**:
- ResNetスタイルの残差結合
- `context`と`token`の両方を更新
- FNN出力を2つに分割: `_context` + `_token`
- 残差結合: `context += _context`, `token += _token`
- `token`ベクトルから最終予測

**アーキテクチャ（Layer-wise [1,1,1,1]の場合）**:

```
Layer 1-4（各層で以下を実行）:

Input: [context, token]  # 結合 [256 + 256 = 512]
  ↓
FNN(512 → 256)  # hidden_dim = 256 に出力
  ↓ ReLU
  ↓ Dropout
  ↓
y (512次元)  # FNN出力
  ↓ 分割
_context (256), _token (256)
  ↓ 残差結合
context += _context  # Addition
token += _token      # Addition
  ↓ LayerNorm
context (256), token (256)

Final Output:
token → Linear(256 → vocab_size) → next_token_logits
```

**数式**:

```
各層での更新:
y = FNN([context, token])
_context, _token = split(y)
context = context + _context
token = token + _token
context = LayerNorm(context)
token = LayerNorm(token)

最終出力:
logits = Linear(token)
```

**パラメータ数** (vocab=50257, embed=256, context=256, hidden=256, [1,1,1,1]):
- Token Embedding: 50,257 × 256 = 12,865,792
- FNN Blocks (4層): 4 × (512×256 + 256) = 525,312
- LayerNorm (context + token, 4層): 4 × (256×2 + 256×2) = 4,096
- Output: 256 × 50,257 = 12,865,792
- **Total**: ~26,261,000 parameters

---

### 1.2 NewLLMResidualContextOnly（Context-Only版）

**ファイル**: `src/models/new_llm_residual_context_only.py`

**特徴**:
- ResNetスタイルの残差結合
- `context`**のみ**を更新（`token`は更新されない）
- FNN出力から`_context`のみ抽出
- 残差結合: `context += _context`のみ
- `context`ベクトルから最終予測（追加のNN層を経由）

**アーキテクチャ（Layer-wise [1,1,1,1]の場合）**:

```
Layer 1-4（各層で以下を実行）:

Input: [context, token]  # 結合 [256 + 256 = 512]
  ↓
FNN(512 → 256)  # hidden_dim = 256 に出力
  ↓ ReLU
  ↓ Dropout
  ↓
hidden (256次元)
  ↓ Projection
_context = Linear(hidden) → 256次元
  ↓ 残差結合
context += _context  # Addition（tokenは更新されない）
  ↓ LayerNorm
context (256)

Final Output:
context → NN(2-layer) → next_token_logits
  NN: context → Linear(256→256) → ReLU → Dropout → Linear(256→vocab_size)
```

**数式**:

```
各層での更新:
hidden = FNN([context, token])
_context = Linear(hidden)
context = context + _context
context = LayerNorm(context)

最終出力（2-layer NN）:
h = ReLU(Linear1(context))
logits = Linear2(h)
```

**パラメータ数** (vocab=50257, embed=256, context=256, hidden=256, [1,1,1,1], output_layers=2):
- Token Embedding: 12,865,792
- FNN Blocks (4層): 525,312
- Context Projection (4層): 4 × (256×256 + 256) = 263,168
- LayerNorm (context, 4層): 4 × (256 + 256) = 2,048
- Output NN (2層): (256×256 + 256) + (256×50257 + 50257) = 13,129,281
- **Total**: ~26,785,600 parameters

---

## 2. Gated Architecture（ゲート機構型）

### 2.1 NewLLMGated（旧 NewLLMFlexible）

**ファイル**: `src/models/new_llm_flexible.py` ← リネーム予定

**特徴**:
- LSTMスタイルのゲート更新機構
- `context`のみを更新（`token`は更新されない）
- Forget Gate、Input Gate を使用
- `hidden`ベクトルから最終予測

**アーキテクチャ（Layer-wise [1,1,1,1]の場合）**:

```
Layer 1-4（各層で以下を実行）:

Input: [token_embed, context]  # 結合 [256 + 256 = 512]
  ↓
FNN(512 → 256)  # hidden_dim = 256 に出力
  ↓ ReLU
  ↓ Dropout
  ↓
hidden (256次元)
  ↓
context_delta = tanh(W_delta @ hidden)
forget_gate = sigmoid(W_forget @ hidden)
input_gate = sigmoid(W_input @ hidden)
  ↓ ゲート更新（LSTM型）
context = forget * context + input * context_delta
  ↓ LayerNorm
context (256)

Final Output:
hidden → Linear(256 → vocab_size) → next_token_logits
```

**数式**:

```
各層での更新:
hidden = FNN([token_embed, context])
context_delta = tanh(W_delta @ hidden)
forget = sigmoid(W_forget @ hidden)
input = sigmoid(W_input @ hidden)
context = forget ⊙ context + input ⊙ context_delta
context = LayerNorm(context)

最終出力:
logits = Linear(hidden)
```

**パラメータ数** (vocab=50257, embed=256, context=256, hidden=256, [1,1,1,1]):
- Token Embedding: 12,865,792
- FNN Blocks (4層): 525,312
- Context Updaters (4層): 4 × (3 × (256×256 + 256)) = 787,968
- LayerNorm (4層): 2,048
- Output: 12,865,792
- **Total**: ~27,046,912 parameters

---

### 2.2 NewLLMContextOnly（旧実装、Gated + Context-Only）

**ファイル**: `src/models/new_llm_context_only.py`

**特徴**:
- ゲート機構を使用
- `context`のみを更新
- `context`ベクトルから最終予測（追加のNN層を経由）

**アーキテクチャ**: NewLLMGatedとほぼ同じだが、最終出力が`context`から予測

---

## 3. アーキテクチャ比較表

| 特徴 | Residual (Standard) | Residual (Context-Only) | Gated (Standard) | Gated (Context-Only) |
|------|---------------------|-------------------------|------------------|----------------------|
| **ファイル** | `new_llm_residual.py` | `new_llm_residual_context_only.py` | `new_llm_flexible.py` | `new_llm_context_only.py` |
| **更新メカニズム** | 残差結合（Addition） | 残差結合（Addition） | ゲート機構（LSTM型） | ゲート機構（LSTM型） |
| **更新対象** | context + token | context のみ | context のみ | context のみ |
| **FNN出力の処理** | 分割 (_context, _token) | 抽出 (_context) | ゲート計算 (delta, forget, input) | ゲート計算 |
| **最終予測元** | token | context | hidden | context |
| **出力層** | 1-layer Linear | 1 or 2-layer NN | 1-layer Linear | 1 or 2-layer NN |
| **パラメータ数** | ~26.3M | ~26.8M | ~27.0M | ~27.6M |

---

## 4. 主要な違いの詳細

### 4.1 更新メカニズム

**Residual（残差結合）**:
```python
# FNN出力を分割
_context, _token = split(FNN_output)

# 残差結合（Addition）
context = context + _context
token = token + _token
```

**利点**:
- ✅ ResNetスタイルの勾配伝播が容易
- ✅ 深い層でも学習が安定
- ✅ 情報の損失が少ない（Addition）

---

**Gated（ゲート機構）**:
```python
# ゲート計算
context_delta = tanh(W_delta @ hidden)
forget = sigmoid(W_forget @ hidden)
input = sigmoid(W_input @ hidden)

# ゲート更新（LSTM型）
context = forget * context + input * context_delta
```

**利点**:
- ✅ LSTMスタイルの選択的な情報保持
- ✅ 長期依存関係の学習が可能
- ✅ 動的なゲート制御

**欠点**:
- ❌ Forget Gateによる情報損失の可能性
- ❌ ゲートが閉じると勾配消失

---

### 4.2 Token Vector の更新

**Residual (Standard)**:
- ✅ `token`ベクトルを各層で更新
- ✅ `token`は現在のトークンの表現を洗練
- ✅ 最終的な`token`から予測

**Others (Context-Only / Gated)**:
- ❌ `token`ベクトルは更新されない
- ❌ `token_embed`は固定（入力のまま）
- ❌ `context`のみが情報を蓄積

---

### 4.3 最終予測

**Standard variants (Residual, Gated)**:
```python
# token または hidden から直接予測
logits = Linear(token)  # Residual
logits = Linear(hidden)  # Gated
```

**Context-Only variants**:
```python
# context から追加のNN層を経由して予測
h = ReLU(Linear1(context))
logits = Linear2(h)
```

---

## 5. どのアーキテクチャを使うべきか？

### 5.1 実験推奨順序

1. **NewLLMResidual (Standard)** ← まずこれをテスト
   - ResNetスタイルの残差結合
   - 理論的に最も安定
   - `token`と`context`の両方を更新

2. **NewLLMGated (Standard)** ← 次にこれと比較
   - LSTMスタイルのゲート機構
   - 既存の実験データあり
   - 長期依存関係に強い可能性

3. **Context-Only variants** ← Context単独で十分か検証
   - 必要パラメータが少し多い
   - 理論的に情報損失の可能性

---

### 5.2 予想される性能順位

**仮説**:
```
NewLLMResidual (Standard) ≥ NewLLMGated (Standard) > Context-Only variants
```

**理由**:
1. Residualは情報損失が少ない（Addition）
2. Gatedは動的制御が可能だが、ゲートが閉じると情報損失
3. Context-Onlyは`token`更新がないため情報が制限される

**実験で検証すべき**:
- 各アーキテクチャの収束速度（Phase 1）
- 最終的なPPL（Phase 2）
- パラメータ効率（PPL per parameter）

---

## 6. 実装状況

| モデル | ファイル | 実装状況 | テスト状況 |
|--------|---------|---------|-----------|
| **NewLLMResidual** | `new_llm_residual.py` | ✅ 完了 | ⏳ 未テスト |
| **NewLLMResidualContextOnly** | `new_llm_residual_context_only.py` | ✅ 完了 | ⏳ 未テスト |
| **NewLLMGated** | `new_llm_flexible.py` | ✅ 完了（リネーム予定） | ✅ テスト済み |
| **NewLLMContextOnly** | `new_llm_context_only.py` | ✅ 完了 | ✅ テスト済み |

---

## 7. Phase 2の学習モード

### 7.1 通常モード（デフォルト）

Phase 2で**全層を学習**：
- 文脈ベクトル生成層も学習
- トークン出力層も学習

```bash
python3 tests/phase2_experiments/test_residual.py --num-samples 10
```

### 7.2 Freeze Context モード

Phase 2で**トークン出力層のみ学習**：
- 文脈ベクトル生成層は固定（勾配を止める）
- トークン出力層のみ学習

```bash
python3 tests/phase2_experiments/test_residual.py --num-samples 10 --freeze-context
```

**実装の仕組み**:
```python
# Phase 2で文脈ベクトルを固定
context_new, token_updated = model._update_context_one_step(...)

if freeze_context:
    token_updated = token_updated.detach()  # 勾配を止める

logits = model.token_output(token_updated)  # トークン出力層のみ学習
```

**目的**:
- Phase 1: 文脈ベクトルの出力方法を学習
- Phase 2: トークン出力方法のみを学習（文脈は固定）

**どちらが良いか**:
- 実験で検証が必要
- Freeze Context モードは過学習を防ぐ可能性あり

---

## 8. Phase 1とPhase 2の責任分離 - CRITICAL

**⚠️ Phase 1とPhase 2で学習対象を完全に分離**

### 責任分離の原則

**Phase 1: 文脈ベクトル生成層のみを学習**
- token_embedding, fnn_blocks, context_norms など
- token_output層は**学習しない**（optimizerから除外）

**Phase 2: token_output層のみを学習**
- token_output層のみ
- 文脈ベクトル生成層は**固定**（optimizerから除外）

### なぜ分離が必要か

1. **Phase 1で学習した固定点を保護**: Phase 2で文脈ベクトル生成層を再学習すると、Phase 1で確立した情報構造が破壊される
2. **責任の明確化**: Phase 1は「良い表現」、Phase 2は「良い出力」を学習
3. **安定した訓練**: Phase 1の成果を保ったままPhase 2を学習

### 実装詳細

```python
# Phase 1: 文脈ベクトル生成層のみ
context_params = [p for name, p in model.named_parameters() if 'token_output' not in name]
optimizer = torch.optim.Adam(context_params, lr=0.0001)

# Phase 2: token_output層のみ
output_params = [p for name, p in model.named_parameters() if 'token_output' in name]
optimizer = torch.optim.Adam(output_params, lr=0.0001)
```

---

## 9. Phase 1 Early Stopping - CRITICAL

**⚠️ Phase 1では99.5%以上の収束が必須条件**

### なぜ99.5%が必要か

Phase 1の収束率が低いと、Phase 2の訓練が不安定になります：

1. **不完全な固定点**: 95%収束では5%のトークンが未収束
2. **Phase 2での不安定性**: 未収束トークンがノイズとなり学習を妨害
3. **性能劣化**: 最終的なPPLが悪化

### 実装詳細

```python
# Phase 1 Early Stopping設定（推奨）
early_stopping = Phase1EarlyStopping(
    convergence_threshold=0.995,  # 99.5% convergence required
    patience=5,                    # Wait 5 iterations after best
    min_delta=0.001                # 0.1% improvement threshold
)
```

**停止条件**:
1. 収束率 >= 99.5% → 即座に停止
2. 収束率が5イテレーション改善しない → 停止（patience）

**間違った設定（95%では不十分）**:
```python
# ❌ 95%では不十分 - Phase 2が不安定になる
early_stopping = Phase1EarlyStopping(convergence_threshold=0.95)
```

### 既存スクリプトの修正状況

✅ `tests/phase2_experiments/test_residual.py` - 99.5%に修正済み
✅ `tests/phase2_experiments/test_multi_sample.py` - 99.5%に修正済み

---

## 10. 次のステップ

1. ✅ NewLLMResidualとNewLLMResidualContextOnlyを実装（完了）
2. ✅ テストスクリプト作成（`tests/phase2_experiments/test_residual.py`）
3. ✅ `--freeze-context` フラグ実装（完了）
4. ✅ Phase 1/2の責任分離実装（完了）
5. ✅ Phase 1 Early Stopping修正（99.5%収束）
6. ⏳ 10サンプルで各アーキテクチャを比較
7. ⏳ Freeze Context モードの効果を検証
8. ⏳ 結果を `docs/experiments/residual_vs_gated_comparison.md` にまとめる
9. ⏳ 最良のアーキテクチャで大規模訓練

---

**作成日**: 2025-11-21
**更新日**: 2025-11-21
