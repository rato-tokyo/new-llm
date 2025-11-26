# New-LLM Project Guidelines

## ⚡ PARALLEL PROCESSING ADOPTED - 並列処理版採用 (2025-11-25)

**並列処理版を標準実装として完全採用しました。**

### 性能指標

**並列版実装** ([src/trainers/phase1.py](src/trainers/phase1.py)):
- **Effective Rank**: 55.9% (429/768) - 検証データ
- **処理時間**: ~11秒（シーケンシャル版265秒の**23x高速化**）
- **dist_reg_weight**: 0.9（多様性90%, CVFP 10%）
- **max_iterations**: 10
- **収束率**: 27.2%（多様性優先のためCVFP収束率は低め、これは正常）

### 設計詳細

**Iteration 0**: シーケンシャル処理（固定点目標確立）
**Iteration 1+**: 並列処理（前回contextを使用）

**並列化の特徴**:
- Token i には previous_contexts[i-1] を使用（1トークン分のずれ）
- 情報遅延があるが、dist_reg_weight=0.9により多様性を補償
- バッチ処理による高速化

**旧シーケンシャル版との比較**:
- シーケンシャル版: 66.6% ER, 265秒
- 並列版: 55.9% ER, 11秒
- トレードオフ: -10.7% ER vs 23x高速化

---

## 📊 MANDATORY: 数値報告ルール - 具体的な数値での報告義務

### 絶対遵守: すべての実験結果は具体的な数値で報告する

**禁止事項**:
- ❌ "GOOD", "EXCELLENT", "MODERATE" などの抽象的表現での報告
- ❌ "改善した", "良好", "適切" などの定性的評価のみの報告
- ❌ 数値を伴わない判定結果の報告

**必須報告項目**:
- ✅ **収束率（訓練・検証両方）**: **具体的なパーセンテージと収束トークン数** (例: 訓練 0.0% (0/6400), 検証 0.0% (0/1280))
- ✅ Effective Rank: **実数値/総次元数とパーセンテージ** (例: 627.29/768 = 81.7%)
- ✅ CVFPロス: **実数値** (例: 0.001873)
- ✅ 収束差分: **実数値** (例: final_diff = 0.000745)
- ✅ イテレーション数: **実数** (例: 10/10イテレーション完了)

**⚠️ 収束率の報告は絶対に省略してはいけない**:
- 訓練データと検証データの両方の収束率を必ず報告
- 収束率 = 収束したトークン数 / 総トークン数
- 0.0%は「収束失敗」ではなく「全イテレーション完走」を意味する

**報告フォーマット例**:
```
訓練結果:
- 収束率: 0.0% (0/6400トークン) - 10イテレーション完走
- Effective Rank: 689.26/768 (89.7%)
- CVFPロス: 0.001732

検証結果:
- 収束率: 0.0% (0/1280トークン) - 10イテレーション完走
- Effective Rank: 627.29/768 (81.7%)

CVFP収束チェック:
- final_diff = 0.000745 (閾値 < 0.001クリア)
```

---

## 🚨🚨🚨 CRITICAL DESIGN - CVFP FIXED-POINT LEARNING (2025-11-26 修正) 🚨🚨🚨

### CVFP理論: 固定点学習の正しい実装

**固定点学習の定義**: `f(x) = x` となる点に収束させる

これは「同じ入力を繰り返し処理したとき、出力が変化しなくなる」ことを意味する。

**正しい実装（CVFP理論に基づく）**:
```
Iteration 0: contexts_0 を出力 → previous_contexts = contexts_0（学習なし）
Iteration 1: contexts_1 を出力 → CVFP損失 = MSE(contexts_1, previous_contexts)
             previous_contexts = contexts_1 に更新
Iteration 2: contexts_2 を出力 → CVFP損失 = MSE(contexts_2, previous_contexts)
             previous_contexts = contexts_2 に更新
...
```

**重要ポイント**:
- ✅ CVFP損失は**前回のコンテキスト（previous_contexts）**と比較
- ✅ previous_contextsは**毎イテレーション更新**してよい
- ✅ 収束判定も前回との差で行う
- ❌ ~~Iteration 0を固定目標として保存~~ ← これは間違い

**なぜ前回との比較が正しいか**:
1. 固定点 = 変化がなくなる点
2. `MSE(current, previous) → 0` は「出力が安定した」ことを意味する
3. これが固定点 `f(x) = x` の定義に合致する

**正しいコード（phase1.py）**:
```python
# CVFP損失: 前回のコンテキストと比較（固定点への収束）
cvfp_loss = compute_cvfp_loss(contexts, previous_contexts)

# 更新
previous_contexts = contexts.detach()
```

---

## 🚨🚨🚨 CRITICAL BUG FIX - CONTEXT CARRYOVER (2025-11-24) 🚨🚨🚨

### 致命的バグ修正: イテレーション間のコンテキスト引き継ぎ（絶対に忘れてはいけない）

**致命的な問題**:
- 訓練・検証の両方で各イテレーションごとにコンテキストがゼロリセットされていた
- **これはCVFP学習の根本を破壊する致命的バグ**
- 固定点学習が全く機能していなかった

**修正内容**:
```python
# ❌❌❌ 絶対にやってはいけない間違った実装（削除済み）
# 毎イテレーションでコンテキストをリセット = CVFP学習の破壊
context = torch.zeros(1, self.model.context_dim, device=device)  # 致命的バグ

# ✅✅✅ 必須の正しい実装（修正済み）
# イテレーション間でコンテキストを必ず引き継ぐ
if self.previous_contexts is None:
    # 初回のみゼロ初期化
    context = torch.zeros(1, self.model.context_dim, device=device)
else:
    # 前イテレーションの最終コンテキストを必ず引き継ぐ（CVFP学習の核心）
    context = self.previous_contexts[-1].unsqueeze(0).detach()
```

**なぜこれが致命的か**:
1. **CVFP = Context Vector Fixed-Point**: 固定点への収束が目的
2. **固定点学習**: イテレーションを重ねて同じ点に収束することが目標
3. **引き継がない = 学習していない**: 毎回リセットでは固定点に到達不可能
4. **検証ロスゼロの謎**: バグのせいで見かけ上良い結果に見えていた

**二度と同じ間違いをしないために**:
- ⚠️ イテレーション間のコンテキスト引き継ぎは**CVFP学習の生命線**
- ⚠️ `previous_contexts`の最終値を次の初期値にすることは**絶対必須**
- ⚠️ この修正なしでは、すべての実験結果が無意味になる

---

## 🚨🚨🚨 CRITICAL DESIGN - PHASE 2 CONTEXT-FIXED LEARNING (2025-11-26) 🚨🚨🚨

### Phase 2: Context-Fixed Token Prediction（完全固定方式）

**Phase 2は2段階処理で実行される**:

#### Stage 1: 初期化（パラメータ更新なし）
- Phase 2開始時に1回だけ実行
- 訓練データの全トークンを処理し、固定文脈ベクトルC*を生成
- **C*は以降絶対に変更しない**

```python
# Stage 1: 固定文脈C*の生成
with torch.no_grad():
    context = torch.zeros(...)
    C_star = []  # 固定文脈ベクトル
    for token_id in token_ids:
        token_embed = get_embedding(token_id)
        context, token_out = cvfp_block(context, token_embed)
        C_star.append(context)  # C*[i]として保存
```

#### Stage 2: 学習（パラメータ更新あり）
- 入力: `[C*[i-1], token_embed[i]]` - 固定文脈を使用
- 出力: `[context_out, token_out]` - CVFPブロックの出力
- **context_outはC*[i]で完全に置換**（MSE制約ではなく値そのもの）
- 予測: `logits = Linear(concat(C*[i], token_out))`

```python
# Stage 2: Context-Fixed Learning
for i, token_id in enumerate(input_ids):
    # 入力: 固定文脈C*[i-1]（i=0の場合はゼロベクトル）
    input_context = C_star[i-1] if i > 0 else torch.zeros(...)

    # CVFPブロック処理
    context_out, token_out = cvfp_block(input_context, token_embed)

    # CRITICAL: context_outは使わず、C*[i]で完全置換
    fixed_context = C_star[i].detach()

    # 予測: 固定文脈 + token_out
    combined = torch.cat([fixed_context, token_out], dim=-1)
    logits = token_output(combined)

    # 損失は予測損失のみ（context_stability_lossは不要）
    loss = CrossEntropy(logits, target)
```

### 記号定義

| 記号 | 意味 |
|------|------|
| `C*[i]` | **固定目標文脈** - Stage 1で計算した、トークンiを処理した後の文脈ベクトル（不変） |
| `context_out` | **学習時の出力文脈** - Stage 2でCVFPブロックが出力する文脈（**使用しない**） |
| `token_out` | **学習時の出力トークン** - Stage 2でCVFPブロックが出力するトークン表現（予測に使用） |

### 勾配フロー

```
入力: [C*[i-1], token_embed[i]]
         ↓
    CVFPブロック
         ↓
出力: [context_out, token_out]
         ↓
    context_outは破棄、C*[i]を使用
         ↓
    combined = [C*[i], token_out]
         ↓
    logits = token_output(combined)
         ↓
    loss = CrossEntropy(logits, target)
```

**勾配の流れ**:
- ✅ `token_out` → CVFPブロック（更新される）
- ✅ `token_output`層（更新される）
- ❌ `context_out` → CVFPブロック（勾配なし - 未使用のため）
- ❌ `C*[i]` → CVFPブロック（勾配なし - detachのため）

### 重要な設計変更（2025-11-26）

- ❌ **旧設計（v1.0）**: MSE制約による「緩い」固定
  ```python
  context_stability_loss = MSE(context_out, C_star[i])  # 緩い制約
  ```
- ✅ **新設計（v2.0）**: context_outをC*[i]で完全置換（完全固定）
  ```python
  fixed_context = C_star[i].detach()  # 完全固定
  combined = torch.cat([fixed_context, token_out], dim=-1)
  ```

### なぜ完全固定が必要か

1. **Phase 1の保護**: Phase 1で学習した文脈表現を確実に保護
2. **明確な役割分離**: context部分の学習を制限し、token_out部分に集中
3. **安定した学習**: 文脈が固定されているため、予測タスクに集中できる

---

## ⚡ 55.9% Effective Rank - Parallel Version Baseline (2025-11-25)

### 並列処理版の性能ベースライン

**並列処理版は検証データで55.9% Effective Rankを達成（23x高速化）**

**実測値**:
- **検証データ**: 55.9% Effective Rank (429/768)
- **訓練データ**: ~60% Effective Rank
- **処理時間**: ~11秒（シーケンシャル版265秒の23x高速化）
- **収束率**: 27.2%（多様性優先のためCVFP収束率は低め、これは正常）

**旧シーケンシャル版（参考）**:
- 検証データ: 66.6% Effective Rank
- 処理時間: 265秒
- 収束率: 30.0%

**並列版採用の理由**:
- ✅ 23x高速化により実用性が大幅向上
- ✅ 55.9% ERは実用的な多様性を維持
- ✅ トレードオフ: -10.7% ER vs 圧倒的高速化

---

## Core Implementation - Parallel Processing

### 1. Diversity Loss: Global Mean-Based Tracking

**✅ 並列版実装 ([src/trainers/phase1.py](src/trainers/phase1.py))**:

```python
def compute_diversity_loss(contexts):
    """
    多様性損失: 全トークンの平均からの偏差（負の損失で最大化）

    Args:
        contexts: 現在のコンテキスト [num_tokens, context_dim]

    Returns:
        diversity_loss: 多様性損失（スカラー）
    """
    context_mean = contexts.mean(dim=0)  # [context_dim]
    deviation = contexts - context_mean  # [num_tokens, context_dim]
    diversity_loss = -torch.norm(deviation, p=2) / len(contexts)
    return diversity_loss
```

**重要ポイント**:
- 全トークンのコンテキスト平均からの偏差を計算
- 負の損失により、平均からの偏差を最大化（多様性促進）
- バッチ処理に最適化された実装

### 2. データ仕様 - 絶対固定

**訓練データ**:
- ソース: UltraChat (HuggingFaceH4/ultrachat_200k)
- サンプル数: 50
- トークン数: 6400
- キャッシュ: `./cache/ultrachat_50samples_128len.pt`

**検証データ** (絶対仕様):
- ソース: 訓練データの最後20%から生成
- トークン数: 1280
- ファイル: `./data/example_val.txt`
- **必須条件**: 全トークンが訓練データに存在すること
- 生成スクリプト: `scripts/create_val_from_train.py`

### 3. 検証データ生成ルール - 厳格

**禁止事項**:
- ❌ `val_data_source = "auto_split"` は厳禁（エラー発生）
- ❌ 訓練データにないトークンを含む検証データ
- ❌ 手動で作成したランダムな検証テキスト

**必須手順**:
```bash
# 訓練データから検証データを生成
python3 scripts/create_val_from_train.py

# config.pyの設定（絶対固定）
val_data_source = "text_file"
val_text_file = "./data/example_val.txt"
```

### 4. 達成結果 - 並列版ベースライン (2025-11-25)

**実測値 (並列版, dist_reg_weight=0.9)**:
- **訓練データ**: ~60% Effective Rank - 6400トークン
- **検証データ**: **55.9% Effective Rank (429/768)** - 1280トークン
- **処理時間**: ~11秒（シーケンシャル版265秒の23x高速化）
- **収束率**: 訓練 27.2%（多様性優先のため低め、これは正常）

**並列版性能**:
- `dist_reg_weight = 0.9` により、並列版の情報遅延を多様性強化で補償
- 23x高速化により実用性が大幅向上
- この数値を並列版ベースラインとする

### 5. 検証データ収束性チェック - Validation Convergence Check

**検証データの収束率は訓練時に計算されない**（1回の順伝播のみ）。
代わりに、学習済みモデルでの収束性を以下のスクリプトで確認：

```bash
python3 check_val_convergence.py --num_trials 10
```

**動作**:
1. 学習済みモデルをロード
2. 検証データを複数回順伝播（デフォルト: 10回）
3. 各試行でCVFP損失（前回との差分MSE）を計算
4. 損失の推移を表示し、減少傾向を自動判定

**出力例**:
```
Trial  1/10: CVFP Loss = N/A (baseline, no previous context)
Trial  2/10: CVFP Loss = 0.245123
Trial  3/10: CVFP Loss = 0.183456
Trial  4/10: CVFP Loss = 0.142789
...
Trial 10/10: CVFP Loss = 0.098234

Statistics:
  - Initial Loss (Trial 2): 0.245123
  - Final Loss (Trial 10): 0.098234
  - Reduction: -59.93%
  - Slope (linear fit): -0.018234

Verdict:
  ✅ CONVERGING: Loss is decreasing - model is converging on validation data
```

**判定基準**:
- ✅ CONVERGING: 損失が明確に減少（slope < -0.001）
- ✅ CONVERGED: 損失が安定（|slope| < 0.001 かつ std < 0.01）
- ❌ DIVERGING: 損失が増加（slope > 0.001）
- ⚠️ UNSTABLE: 損失が不安定（上記以外）

**使用場面**:
- 訓練完了後に検証データでの収束性を確認
- モデルの固定点学習が汎化しているかを検証
- 異なるチェックポイントの比較

---

## Architecture Configuration - Parallel Version

```python
# Model Architecture
num_layers = 6                  # 6-layer CVFP blocks
context_dim = 768               # GPT-2 aligned
embed_dim = 768                 # GPT-2 pretrained
hidden_dim = 1536               # 2 × embed_dim
layernorm_mix = 1.0             # Full LayerNorm (CRITICAL)

# Diversity Regularization (並列版最適化)
dist_reg_weight = 0.9           # 90% diversity, 10% CVFP (parallel optimized)
                                # 並列版の情報遅延を多様性強化で補償

# Training
phase1_learning_rate = 0.002    # Fast convergence
phase1_max_iterations = 10      # 並列処理による高速化
```

---

## Training Pipeline - Standard Workflow

### Phase 1: CVFP Learning

```bash
# Standard test (uses fixed train/val data)
python3 test.py
```

**実行内容**:
1. 訓練データロード (6400トークン from cache)
2. 検証データロード (1280トークン from text file)
3. モデル訓練 (Phase1Trainer)
4. 検証データで評価
5. **3つの必須チェック実行** (詳細は下記)

---

## 3 Critical Checks - ABSOLUTELY REQUIRED (絶対必要な3つのチェック)

**これらのチェックを省くと問題が多発します。test.pyで必ず実行してください。**

### Check 1: Effective Rank (多様性確認)

**目的**: コンテキストベクトルが多様な次元を使用しているか確認

**実装**: `analyze_fixed_points(contexts)` in `src/evaluation/metrics.py`

**合格基準**:
- 訓練データ: 88-89% Effective Rank
- 検証データ: 81-82% Effective Rank

**失敗例**:
- ❌ Effective Rank < 30%: 次元が偏っている（多様性なし）
- ❌ Global attractor: 全トークンが同じコンテキストに収束

### Check 2: Identity Mapping Check (恒等写像チェック)

**目的**: モデルが学習できているか、単なる恒等写像でないか確認

**実装**: `check_identity_mapping(model, token_embeds, contexts, device)` in `src/evaluation/metrics.py`

**合格基準**:
- ✅ Zero context との差分 > 0.1
- ✅ Token embedding との類似度 < 0.95

**失敗例**:
- ❌ 学習後のコンテキストがゼロベクトルと同じ → 学習なし
- ❌ コンテキストがトークン埋め込みと同一 → 恒等写像

### Check 3: CVFP Convergence Check (固定点収束チェック)

**目的**: 固定点学習ができているか、反復実行で安定した結果になるか確認

**実装**: `check_cvfp_convergence(trainer, token_ids, device)` in `src/evaluation/metrics.py`

**合格基準**:
- ✅ Final diff < 1e-3 (GOOD以上)
- ✅ イテレーション間の変化が減少傾向

**失敗例**:
- ❌ Final diff > 1e-2: 固定点に収束していない
- ❌ イテレーション間で変化が増加 → 発散している

---

## Reproducibility - 完全な再現性保証

**乱数シード固定 (必須)**:

```python
def set_seed(seed=42):
    """全ての乱数生成器のシードを固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

**なぜ必要か**:
- 同じコード、同じデータで**完全に同じ結果**を保証
- 実装が維持されているかの確認に不可欠
- デバッグとトラブルシューティングを容易に

**期待される結果**:
- 訓練データ Effective Rank: **完全に同じ値** (小数点以下まで一致)
- 検証データ Effective Rank: **完全に同じ値** (小数点以下まで一致)
- 3つのチェック結果: 毎回同じ

---

## File Structure - Final Organization

**Main Scripts**:
- `test.py` - 標準テストスクリプト（6400訓練 + 1280検証）
- `train.py` - フル訓練スクリプト
- `config.py` - 設定ファイル

**Data Generation**:
- `scripts/create_val_from_train.py` - 検証データ生成（訓練データから）

**Core Implementation**:
- `src/training/phase1_trainer.py` - Phase 1訓練ロジック（Dimension Usage Statistics）
- `src/models/new_llm_residual.py` - モデルアーキテクチャ
- `src/data/loader.py` - データローダー（auto_split禁止ロジック）

---

## Validation Data Policy - CRITICAL

### 必須仕様

**検証データは訓練データの部分集合でなければならない**:
- 全ての検証データトークンが訓練データに存在
- ランダムな分割は禁止（`auto_split` 使用でエラー）
- 訓練データから直接生成（`create_val_from_train.py`）

### エラー発生ロジック

`loader.py` で実装済み:
```python
if config.val_data_source == "auto_split":
    raise ValueError(
        "❌ CRITICAL ERROR: auto_split is STRICTLY FORBIDDEN!"
        "Use val_data_source='text_file' with data/example_val.txt"
    )
```

---

## Code Quality Standards

### Principles

1. **No Hardcoding**: All hyperparameters in config.py
2. **Single Responsibility**: Each module has one clear purpose
3. **Immutable Data**: Training/validation data are fixed
4. **Error Prevention**: Auto-split is forbidden with error

### Anti-Patterns to Avoid

- ❌ Changing train/val data without regeneration
- ❌ Using auto_split for validation
- ❌ Modifying diversity loss implementation
- ❌ Changing architecture without full retraining

---

## Performance Benchmarks

**CPU Performance (Apple Silicon/Intel)**:
- Training speed: 250-330 tok/s
- 6400 tokens: ~25 seconds per iteration
- Validation: ~4 seconds (1280 tokens)

**Expected Results (commit 9ee3281 baseline)**:
- Training Effective Rank: 74.0% (568.31/768)
- Validation Effective Rank: 66.6% (511.56/768)
- Training Convergence: 30.0% (1923/6400 tokens)
- Validation Convergence: 100.0% (1280/1280 tokens)
- Full 10 iterations complete

---

## No Hardcoding Policy - Reinforced

**全てのパラメータはconfig.pyで定義**:
```python
# ✅ Good
learning_rate = config.phase1_learning_rate
num_samples = config.num_samples

# ❌ Bad
learning_rate = 0.002  # Hardcoded!
num_samples = 50       # Hardcoded!
```

---

## 🐛 CRITICAL BUG FIX HISTORY - November 24, 2025

### Bug #1: F.normalize() in CVFP Loss Calculation (src/training/phase1_trainer.py)

**Problem**:
- Location: [phase1_trainer.py:265-267](src/training/phase1_trainer.py#L265-L267)
- CVFP loss used `F.normalize()` on both `new_context` and `previous_context`
- This only enforces **cosine similarity** (direction), not **value equality**
- Fixed points require `f(x) = x` (exact values), not just same direction

**Symptoms**:
- 0% convergence rate despite 10 iterations
- MSE ~32-33 (vs threshold 0.1 = 300x larger)
- CVFP loss increasing instead of decreasing

**Root Cause**:
```python
# ❌ WRONG: Normalization prevents value convergence
cvfp_loss = F.mse_loss(
    F.normalize(new_context, p=2, dim=1),      # Only matches direction
    F.normalize(previous_context, p=2, dim=1)  # Norms can still diverge
)
```

**Fix**:
```python
# ✅ CORRECT: Raw MSE for exact value matching
cvfp_loss = F.mse_loss(new_context, previous_token_context)
```

**Affected File**: [src/training/phase1_trainer.py:267](src/training/phase1_trainer.py#L267)

---

### Bug #2: Missing context.detach() Between Tokens (src/training/phase1_trainer.py)

**Problem**:
- Location: [phase1_trainer.py:226-240](src/training/phase1_trainer.py#L226-L240)
- Context passed between tokens without `detach()`
- Gradient graph reused across token sequence
- RuntimeError: "Trying to backward through the graph a second time"

**Root Cause**:
```python
# ❌ WRONG: Gradient graph carries over
context = self._train_one_token(
    token_embed.unsqueeze(0),
    context,  # No detach - gradient accumulates across tokens
    token_idx=t
)
current_contexts[t] = context.squeeze(0)  # No detach for convergence check
```

**Fix**:
```python
# ✅ CORRECT: Detach between tokens
context = self._train_one_token(
    token_embed.unsqueeze(0),
    context.detach(),  # Break gradient flow between tokens
    token_idx=t
)
current_contexts[t] = context.squeeze(0).detach()  # Detach for convergence tracking
```

**Affected Lines**:
- [phase1_trainer.py:228](src/training/phase1_trainer.py#L228) - Training token processing
- [phase1_trainer.py:240](src/training/phase1_trainer.py#L240) - Convergence tracking

---

### Verification Results (After Fixes)

**With dist_reg_weight=0.01** (99% CVFP, 1% Diversity):
- ✅ Convergence mechanism works: 96.0% training, 100.0% validation
- ✅ CVFP loss decreases: 1.02 → 0.021 → 0.025
- ❌ Effective Rank collapsed: 6.9% training, 1.1% validation (vs 66.6% baseline)
- **Conclusion**: Bug fixed, but diversity weight too low

**With dist_reg_weight=0.5** (50% CVFP, 50% Diversity) - Baseline (commit 9ee3281):
- ✅ Training Effective Rank: 74.0% (568.31/768)
- ✅ Validation Effective Rank: 66.6% (511.56/768)
- ✅ Training Convergence: 30.0% (1923/6400)
- ✅ Validation Convergence: 100.0% (1280/1280)

---

## 📐 NEW-LLM Detailed Architecture Specification

### Core Components

**1. CVFPLayer (Context Vector Fixed-Point Layer)**
- Location: [src/models/new_llm_residual.py:15-102](src/models/new_llm_residual.py#L15-L102)
- Input: `context [batch, context_dim]`, `token_embed [batch, embed_dim]`
- Output: `new_context [batch, context_dim]`, `new_token [batch, embed_dim]`
- Architecture:
  - FNN: `[context + token] → [hidden_dim]` with ReLU
  - Split: `hidden_dim → delta_context + delta_token`
  - Residual: `new_context = context + delta_context`
  - LayerNorm: Optional mixing with `layernorm_mix` parameter

**2. CVFPBlock (Multiple Layers)**
- Location: [src/models/new_llm_residual.py:105-150](src/models/new_llm_residual.py#L105-L150)
- Sequential execution of `num_layers` CVFPLayer instances
- Passes context and token through all layers

**3. NewLLMResidual (Main Model)**
- Location: [src/models/new_llm_residual.py:153-314](src/models/new_llm_residual.py#L153-L314)
- Token Embedding: GPT-2 pretrained (768-dim, frozen)
- CVFP Blocks: 6 blocks (configurable via `layer_structure`)
- Output Head: Linear layer `context_dim → vocab_size`

**4. Phase1Trainer (CVFP Fixed-Point Learning)**
- Location: [src/training/phase1_trainer.py](src/training/phase1_trainer.py)
- Training loop: Iterative refinement until convergence
- Loss function:
  - CVFP Loss: `MSE(context_t, context_{t-1})` - **NO normalization**
  - Diversity Loss: EMA-based per-dimension variance tracking
  - Total: `(1-w) * cvfp_loss + w * diversity_loss`
- Convergence: MSE < threshold (0.1) for 95% of tokens
- Early stopping: When 95% converged (training only)

### Key Design Decisions

**Dimension Constraints**:
- `hidden_dim = context_dim + embed_dim` (MANDATORY)
- Default: `context_dim=768, embed_dim=768, hidden_dim=1536`
- Reason: FNN output must split into delta_context + delta_token

**Context Carryover** (CRITICAL):
- Between iterations: `context = previous_contexts[-1]` (NOT zero reset)
- Between tokens: `context = context.detach()` (gradient isolation)
- Reason: Fixed-point learning requires continuity

**Gradient Management**:
- Token embeddings: Frozen (GPT-2 pretrained)
- Context params: Trained (all CVFP layers)
- Between tokens: Detached (prevent cross-token gradients)
- Reason: Stable training with efficient gradient flow

**Diversity Regularization**:
- Method: Per-dimension variance tracking with EMA
- Implementation: Negative L2 norm of deviation from mean
- Memory: O(context_dim) - 6KB for 768-dim
- Reason: Encourage usage of all dimensions

---

## Context Size Monitoring Policy

**Claude Codeコンテキスト管理**:
- 100,000トークン超過時: 初回報告
- 以降10,000トークン刻みで継続報告
- 190,000トークン以上: 新セッション開始を強く推奨

---

Last Updated: 2025-11-24 (Bug Fixes + Architecture Documentation)
