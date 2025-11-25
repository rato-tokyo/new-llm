# New-LLM Project Guidelines

## 🚫 IMMUTABLE CODE RULE - DELETE NOT ALLOWED

### `verbose=(iteration == 0)` in phase1.py:172 - ABSOLUTE IMMUTABLE

**削除不能ルール**:
- ✅ [src/trainers/phase1.py:172](src/trainers/phase1.py#L172)の`verbose=(iteration == 0)`は**絶対に変更してはいけない**
- ✅ この設定により、Iteration 0（順伝播のみ）でのみ進捗表示が行われる
- ✅ Iteration 1以降（最適化実行中）は進捗非表示

**理由**:
- 各イテレーションの最後に収束率・損失のログが表示される
- 途中の進捗表示は不要であり、ログの可読性を保つための設計
- この設定は意図的なものであり、削除や変更は厳禁

**この設定を変更してはいけない理由**:
- ❌ 「進捗が見えないから変更」は誤った判断
- ❌ 「ユーザビリティ向上のため」は不要
- ✅ 設計意図を理解し、現状維持すること

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

## 🚨🚨🚨 CRITICAL DESIGN FIX - FIXED-POINT TARGET PRESERVATION (2025-11-25) 🚨🚨🚨

### 致命的設計ミス: 固定点目標の上書き（絶対に忘れてはいけない）

**致命的な問題**:
- `Network.update_convergence()`が毎イテレーション`previous_contexts`を上書き
- CVFP損失が「固定点への収束」ではなく「前回との差分」を学習
- **これは固定点学習の定義に完全に反する**

**間違った動作フロー**:
```
Iteration 0: contexts_0 を出力 → previous_contexts = contexts_0
Iteration 1: contexts_1 を出力 → CVFP損失 = MSE(contexts_1, contexts_0)
             previous_contexts = contexts_1 に上書き ← ⚠️ 致命的バグ
Iteration 2: contexts_2 を出力 → CVFP損失 = MSE(contexts_2, contexts_1) ← 目標が変わっている！
             previous_contexts = contexts_2 に上書き
```

**正しい修正**:
```python
# Phase1Trainer (src/trainers/phase1.py)

# ❌❌❌ 絶対にやってはいけない間違った実装（削除済み）
# Networkのprevious_contextsを直接Optimizerに渡す = 毎回更新される
self.cvfp_optimizer.start_new_iteration(
    iteration,
    self.network.previous_contexts  # これは毎回更新されてしまう！
)

# ✅✅✅ 必須の正しい実装（修正済み）
# Iteration 0の出力を固定保存
target_contexts = None

for iteration in range(self.max_iterations):
    if is_training and iteration > 0:
        self.cvfp_optimizer.start_new_iteration(
            iteration,
            target_contexts  # 固定された目標を渡す
        )

    contexts = self.network.forward_all(...)

    # Iteration 0の出力を保存（以降は変更しない）
    if iteration == 0:
        target_contexts = contexts.detach().clone()

    # 収束状態を更新（収束判定専用 - これは毎回更新してよい）
    self.network.update_convergence(contexts)
```

**正しい動作フロー**:
```
Iteration 0: contexts_0 を出力 → target_contexts = contexts_0（固定保存）
Iteration 1: contexts_1 を出力 → CVFP損失 = MSE(contexts_1, target_contexts) ← 固定点と比較
Iteration 2: contexts_2 を出力 → CVFP損失 = MSE(contexts_2, target_contexts) ← 同じ目標！
Iteration 3: contexts_3 を出力 → CVFP損失 = MSE(contexts_3, target_contexts) ← 同じ目標！
```

**なぜこれが致命的か**:
1. **Fixed-Point = 固定点**: f(x) = x となる点への収束が目標
2. **目標が動く = 固定点ではない**: 毎回目標が変わると収束判定が無意味
3. **Moving Target問題**: 常に1ステップ前との差分最小化になり、固定点学習ではない
4. **CVFP損失の本質的破壊**: 固定点への距離ではなく、差分最小化になる

**二度と同じ間違いをしないために**:
- ⚠️ **Iteration 0の出力を固定保存**し、以降は変更しない
- ⚠️ `Network.previous_contexts`は収束判定専用（毎回更新してよい）
- ⚠️ CVFP損失計算には**固定されたtarget_contexts**を使用
- ⚠️ 「前回との差分」≠「固定点への収束」を理解する

**責任分離**:
- `Phase1Trainer`: 固定点目標（`target_contexts`）の保存と管理
- `Network`: 収束判定用の前回値（`previous_contexts`）の管理
- `Optimizer`: 固定された目標との損失計算

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

## 🚨🚨🚨 CRITICAL DESIGN FIX - PHASE 2 CONTEXT PROPAGATION (2025-11-24) 🚨🚨🚨

### 致命的設計ミス: Phase 2での各トークン独立処理（絶対に忘れてはいけない）

**致命的な問題（修正前の実装）**:
- Phase 2で各トークンが完全に独立して処理されていた（毎回0ベクトルから開始）
- **これはPhase 1と矛盾し、言語モデルとして致命的な欠陥**
- 文脈情報が全く伝わらず、Phase 1の学習が無駄になっていた

**修正内容**:
```python
# ❌❌❌ 絶対にやってはいけない間違った実装（削除済み）
# 各トークンが独立 = 文脈伝達なし = Phase 1との不整合
for token_id in input_ids:
    context = torch.zeros(...)  # 毎回リセット！致命的バグ
    context = CVFP(token_embed, context)
    logits = predict(context)

# ✅✅✅ 必須の正しい実装（修正済み）
# Context伝播 + Token Embed予測
context = torch.zeros(...)  # 最初のトークンのみ0から開始

for token_id in input_ids:
    context = context.detach()  # 勾配遮断（重要）
    token_embed = embedding(token_id)

    # CVFP処理（文脈とトークン埋め込み両方を伝播）
    for block in blocks:
        context, token_embed = block(token_embed, context)

    # Token embedから予測（contextからではない）
    logits = predict(token_embed)

    # Contextは次のトークンに引き継がれる
```

### なぜこれが致命的か

1. **Phase 1との不整合**: Phase 1ではcontext伝播が必須、Phase 2でも同様であるべき
2. **文脈情報の欠如**: 各トークンが独立では系列全体の理解が不可能
3. **Phase 1学習の無駄**: 文脈伝達メカニズムがPhase 2で活用されない
4. **言語モデルとして不完全**: 前のトークン情報が全く使えない

### 正しい設計の重要ポイント

**1. Context伝播（Phase 1と同じ）**:
- 最初のトークンのみ0ベクトルから開始
- 以降は前のcontextを引き継ぐ
- Phase 1で学習した文脈伝達メカニズムを活用

**2. Context勾配遮断**:
- `context = context.detach()` で勾配を遮断
- 理由: 系列全体への勾配伝播を防ぎ、学習を安定化

**3. Token Embed予測**:
- 予測は`token_embed`から（`context`からではない）
- Contextは文脈記憶、Token Embedは出力表現として分離

**4. 全トークン一括処理**:
- バッチ分割なし（context伝播があるため）
- Phase 1と同じ処理フロー

---

## ⚠️ 89.4% Effective Rank Implementation - IMMUTABLE SPECIFICATION

### 絶対仕様: この実装は変更禁止 (ABSOLUTE: This implementation is IMMUTABLE)

**この仕様は検証データで89.4% Effective Rankを達成した最終実装です（dist_reg_weight=0.5）。**
**以下の実装と矛盾する内容は全て無効です。**

---

## Core Implementation - Dimension Usage Statistics

### 1. Diversity Loss: Per-Dimension Usage Tracking

**✅ 正しい実装 (現在のphase1_trainer.py)**:

```python
# 各イテレーション開始時にリセット
dim_stats = torch.zeros(context_dim, device=device)

# 各トークン処理時
dim_weights = 1.0 / (dim_stats + 1.0)  # 使用頻度の逆数（detached）
diversity_loss = -(dim_weights * context.abs().squeeze(0)).mean()  # 負の損失で活性化最大化

# 統計更新（勾配なし） - 次のトークン用
with torch.no_grad():
    dim_stats += context.abs().squeeze(0)
```

**重要ポイント**:
- `dim_weights` は detached（勾配なし）
- `context` には勾配が流れる
- 負の損失により、使用頻度が低い次元を優先的に活性化

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

### 4. 達成結果 - 最終仕様

**実測値 (2025-11-24, dist_reg_weight=0.5)**:
- **訓練データ**: 89.7% Effective Rank (689.26/768) - 6400トークン
- **検証データ**: **89.4% Effective Rank (686.90/768)** - 1280トークン ✅
- **CVFP収束チェック**: final_diff = 0.000745 < 0.001 ✅

**なぜ89.4%か**:
- `dist_reg_weight = 0.5` により、CVFP学習と多様性の両立を実現
- 訓練データと検証データで同等の高い多様性（89.7% vs 89.4%）
- 85%目標を大幅に超える成果

---

## Architecture Configuration - Fixed

```python
# Model Architecture
num_layers = 6                  # 6-layer CVFP blocks
context_dim = 768               # GPT-2 aligned
embed_dim = 768                 # GPT-2 pretrained
hidden_dim = 1536               # 2 × embed_dim
layernorm_mix = 1.0             # Full LayerNorm (CRITICAL)

# Diversity Regularization
dist_reg_weight = 0.5           # 50% diversity, 50% CVFP (balanced)

# Training
phase1_learning_rate = 0.002    # Fast convergence
phase1_max_iterations = 10      # Usually converges in 2 iterations
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

**Expected Results**:
- Training Effective Rank: 89.7%
- Validation Effective Rank: 89.4%
- CVFP Convergence Check: final_diff < 0.001
- Convergence: Usually 10 iterations (all iterations complete)

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
- ❌ Effective Rank collapsed: 6.9% training, 1.1% validation (vs 89.4% target)
- **Conclusion**: Bug fixed, but diversity weight too low

**With dist_reg_weight=0.5** (50% CVFP, 50% Diversity) - Expected:
- ✅ Convergence mechanism: Should work (proven above)
- ✅ Effective Rank: ~89.4% (balanced training)
- ✅ All 3 critical checks should pass

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
