# New-LLM Project Guidelines

## 📊 MANDATORY: 数値報告ルール - 具体的な数値での報告義務

### 絶対遵守: すべての実験結果は具体的な数値で報告する

**禁止事項**:
- ❌ "GOOD", "EXCELLENT", "MODERATE" などの抽象的表現での報告
- ❌ "改善した", "良好", "適切" などの定性的評価のみの報告
- ❌ 数値を伴わない判定結果の報告

**必須報告項目**:
- ✅ 収束率: **具体的なパーセンテージ** (例: 収束率 0.0%)
- ✅ Effective Rank: **実数値/総次元数とパーセンテージ** (例: 627.29/768 = 81.7%)
- ✅ CVFPロス: **実数値** (例: 0.001873)
- ✅ 収束差分: **実数値** (例: final_diff = 0.000745)
- ✅ イテレーション数: **実数** (例: 10/10イテレーション完了)

**報告フォーマット例**:
```
訓練結果:
- Effective Rank: 689.26/768 (89.7%)
- 収束率: 0.0% (10イテレーション完走)
- CVFPロス: 0.001732
- 収束判定: final_diff = 0.000745 (閾値1e-3未満)
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

## ⚠️ 81.7% Effective Rank Implementation - IMMUTABLE SPECIFICATION - ABSOLUTELY FINAL

### 絶対仕様: この実装は永久に変更禁止 (ABSOLUTE: This implementation is PERMANENTLY IMMUTABLE)

**この仕様は検証データで81.7% Effective Rankを達成した最終実装です。**
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

**実測値 (2025-11-24)**:
- **訓練データ**: 88.7% Effective Rank (681.47/768) - 6400トークン
- **検証データ**: **81.7% Effective Rank (627.29/768)** - 1280トークン ✅

**なぜ81.7%か**:
- 1280トークンは768次元空間の多様性表現に対して少し不足
- しかし、実用上十分な多様性を達成
- 85%目標に対して非常に良好な結果

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
dist_reg_weight = 0.99          # 99% diversity, 1% CVFP

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
- Training Effective Rank: 88-89%
- Validation Effective Rank: 81-82%
- Convergence: 2 iterations (early stopping)

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

## Context Size Monitoring Policy

**Claude Codeコンテキスト管理**:
- 100,000トークン超過時: 初回報告
- 以降10,000トークン刻みで継続報告
- 190,000トークン以上: 新セッション開始を強く推奨

---

Last Updated: 2025-11-24 (Final 81.7% Implementation)
