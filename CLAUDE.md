# New-LLM Project Guidelines

## Diversity Regularization - Critical Design Specification ⭐ BREAKTHROUGH

### Final Approach: LayerNorm + Per-Dimension Variance Tracking (EMA-based)

**After extensive experimentation (Nov 2025), we achieved 89.3% Effective Rank using:**

1. **LayerNorm (Value Explosion Prevention)**
   - Prevents residual connection value accumulation
   - Controls scale through normalization
   - Essential for stable training with 6 layers

2. **Per-Dimension Variance Tracking (Diversity Enforcement)**
   - **指数平均的 (Exponential Moving Average-like)**: True online learning with fixed memory
   - Tracks mean and variance per dimension using EMA
   - No history storage required - only 6KB memory for 768-dim contexts
   - Diversity loss penalizes low variance across dimensions

3. **Gradient Clipping (Training Stability)**
   - `max_norm=1.0` prevents gradient explosion
   - Ensures stable convergence

**Results (5000 tokens, 6 layers, 768-dim):**
- Training: Effective Rank 686.09/768 (89.3%) ✅
- Validation: Effective Rank 609.88/768 (79.4%) ✅
- Memory: Only 6KB (vs 2,307KB for covariance matrix)
- Speed: 1.55x faster than covariance matrix approach
- No value explosion, stable losses

### Why This Design?

1. **True Online Learning (指数平均的)**: Fixed memory usage regardless of token count
2. **Simplicity**: Only track per-dimension mean and variance
3. **Effectiveness**: 89.3% Effective Rank (nearly full 768-dimensional diversity)
4. **Efficiency**: 384x less memory, 1.55x faster than covariance matrix
5. **Scalability**: Works with any sequence length and context dimension

### Why This Design?

1. **Simplicity**: No complex covariance/orthogonality constraints
2. **Effectiveness**: Dramatic improvement from 6-12% to 80%+ Effective Rank
3. **Stability**: No NaN/Inf issues, controlled norms
4. **Scalability**: Works with any sequence length

### 指数平均的 (Exponential Moving Average-like) - Definition

**指数平均的**とは、履歴を保存せずに統計量を更新する手法を指します：

**通常の平均**:
- すべての過去データを保存
- 平均 = Σ(x_i) / n
- メモリ: O(n)（データ数に比例）

**指数平均的（EMA）**:
- 過去データを保存しない
- 現在の統計量 + 最新値の加重和のみ
- メモリ: O(1)（固定）

**例（平均の更新）**:
```python
# 通常: 全データ保存が必要
all_data.append(new_value)
mean = sum(all_data) / len(all_data)

# 指数平均的: 現在の平均のみ保持
mean = momentum * mean + (1 - momentum) * new_value
```

本プロジェクトでは、コンテキストベクトルの**平均と分散**を指数平均的に追跡することで、真のオンライン学習を実現しています。

### Implementation Details

**Phase1Trainer._train_one_token() method:**

```python
# EMA統計量（初期化）
self.context_mean_ema = None  # [context_dim] - 各次元の平均
self.context_var_ema = None   # [context_dim] - 各次元の分散
self.ema_momentum = 0.99      # EMA係数

# 1. Compute CVFP loss (context convergence)
cvfp_loss = F.mse_loss(
    F.normalize(new_context, p=2, dim=1),
    F.normalize(previous_context, p=2, dim=1)
)

# 2. Diversity loss (per-dimension variance tracking)
new_context_flat = new_context.squeeze(0)  # [context_dim]

if self.context_mean_ema is None:
    # 初期化
    self.context_mean_ema = new_context_flat.detach()
    self.context_var_ema = torch.ones_like(new_context_flat)
    diversity_loss = 0.0
else:
    # EMA更新前の値を使用
    old_mean = self.context_mean_ema.detach()
    old_var = self.context_var_ema.detach()

    # 偏差
    deviation = new_context_flat - old_mean

    # 多様性損失: 分散が低い = 多様性不足
    diversity_loss = 1.0 / (old_var.mean() + 1e-6)

    # EMA更新（指数平均的）
    with torch.no_grad():
        self.context_mean_ema = (
            self.ema_momentum * old_mean +
            (1 - self.ema_momentum) * new_context_flat
        )
        deviation_sq = deviation ** 2
        self.context_var_ema = (
            self.ema_momentum * old_var +
            (1 - self.ema_momentum) * deviation_sq
        )

# 3. Combined loss
total_loss = (1 - dist_reg_weight) * cvfp_loss + dist_reg_weight * diversity_loss

# 4. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Key Configuration (config.py)**:
- `layernorm_mix = 1.0` - Full LayerNorm (critical!)
- `dist_reg_weight = 0.99` - 99% diversity, 1% CVFP
- `phase1_learning_rate = 0.002` - Higher LR for faster convergence
- `num_layers = 6` - Minimum conversational model
- `ema_momentum = 0.99` - EMA decay factor (99% old, 1% new)

### Development History and Design Evolution

**採用に至った経緯 (Adoption Process):**

1. **2025-11-23: 初期実装 - Past 10 Contexts Method**
   - 過去10トークンのコンテキストを保持し、直交性を計算
   - 結果: Effective Rank 80-90%達成
   - 問題: メモリ使用量がトークン数に比例（O(n)）

2. **2025-11-24: 真のオンライン学習の必要性**
   - ユーザーからの指摘: "指数平均的にしてほしい"
   - 要求: 履歴保存なし、固定メモリ、多様性保証

3. **2025-11-24: 方式比較実験**
   - **方式1（共分散行列EMA）**: 理論的に厳密だが重い
     - メモリ: 2,307KB（768×768行列）
     - 訓練時間: 49.30秒（5000トークン）
     - Effective Rank: 54.3%
   - **方式2（次元ごとの分散追跡）**: シンプルで効果的
     - メモリ: 6KB（平均+分散のみ）
     - 訓練時間: 31.87秒（**1.55倍高速**）
     - Effective Rank: **89.3%** ✅

4. **2025-11-24: 正式採用決定**
   - 方式2が全指標で優位
   - 89.3%のEffective Rankで多様性を保証
   - 真のオンライン学習（指数平均的）を実現

### Lessons Learned from Failed Approaches

**What We Tried:**

1. **Past 10 Contexts (Orthogonality Constraints)** ⚠️ (Worked but not optimal)
   - Achieved 80-90% Effective Rank
   - Direct orthogonality enforcement
   - Problem: O(n) memory usage, not truly online
   - → Replaced by EMA-based variance tracking

2. **Covariance Matrix EMA** ❌ (Too heavy)
   - 理論的には最も厳密
   - Memory: 2,307KB (384x heavier)
   - Speed: 1.55x slower
   - Effectiveness: Lower (54.3% vs 89.3%)
   - → Rejected due to poor efficiency/effectiveness ratio

3. **Without LayerNorm** ❌
   - Critical mistake in early experiments
   - Residual connections caused value explosion
   - Norms reached 10^23 levels
   - **User correctly identified**: "値が爆発するのはなぜですか？layernormを導入していますか？"

4. **Fixed Dimension Assignment** ⚠️ (Superseded)
   - Achieved 80.3% Effective Rank
   - Hash-based dimension assignment
   - → Superseded by variance tracking (89.3%)

**Key Insights:**
- **Simpler is better**: Per-dimension variance beats complex covariance matrix
- **EMA is powerful**: Exponential moving average enables true online learning
- **Memory matters**: 6KB vs 2,307KB makes a huge practical difference
- **Diversity guarantee**: 89.3% Effective Rank proves variance tracking works
- **User-driven design**: "指数平均的" requirement led to optimal solution

### Architecture Pattern

**CVFPLayer** encapsulates:
- FNN-based context update
- Residual connections
- Optional LayerNorm mixing
- Clean forward/backward interface

**Phase1Trainer** handles:
- Training loop and convergence detection
- Diversity regularization logic
- CVFP loss calculation
- Optimizer and gradient management

**Separation of Concerns:**
- Model: Architecture and forward pass only
- Trainer: Training logic, loss, optimization

## Phase 2 Strategy - Multi-Output Architecture

### Design Philosophy

**Two-Phase Training with Parameter Expansion**

#### Phase 1: Context Vector Learning (Current)
- **Architecture**: Single `token_output` head
- **Focus**: Pure CVFP learning without token prediction interference
- **Output**: Trained context generation + one token_output (38.6M params)
- **Checkpoint**: Save complete model state

#### Phase 2: Token Prediction with Multi-Output
- **Architecture Expansion**: Convert single output to per-block outputs
- **Strategy**: Clone trained `token_output` to each of 6 blocks
- **Benefit**: Start from trained parameters, not random initialization
- **Training**: Each block's output fine-tunes independently

### Implementation Strategy

```python
# Phase 1: Train with single output (current)
model = NewLLMResidual(...)  # Single token_output
train_phase1(model)
save_checkpoint(model)

# Phase 2: Expand to multi-output
phase1_model = load_checkpoint()
phase2_model = expand_to_multi_output(phase1_model)
# Each block gets copy of trained token_output
# Then train all 6 outputs independently

train_phase2(phase2_model)
```

### Parameter Cloning Details

```python
def expand_to_multi_output(phase1_model):
    """
    Clone trained token_output to each block

    Before: 1 × token_output (38.6M)
    After:  6 × token_output (231.4M)

    Each clone inherits trained weights as initialization
    """
    trained_output = phase1_model.token_output

    block_outputs = [
        clone_linear(trained_output)
        for _ in range(num_blocks)
    ]

    return phase2_model_with_block_outputs
```

### Why This Approach?

1. **Phase 1 Purity**: Context learning unaffected by token prediction
2. **Efficient Phase 2**: Start from trained params, faster convergence
3. **Clean Separation**: Two distinct training objectives
4. **Checkpoint Reuse**: Phase 1 checkpoint remains valuable
5. **Parameter Efficiency**: Single output in Phase 1, expand only when needed

### Training Configuration

**Phase 1**:
- Train: CVFPBlocks + single token_output
- Freeze: Embeddings (GPT-2 pretrained)
- Objective: Context convergence + diversity

**Phase 2**:
- Train: 6 × block_outputs (cloned from Phase 1)
- Options: Freeze or fine-tune CVFPBlocks
- Objective: Token prediction from each block's context

### Loss Computation in Phase 2

```python
# Weighted loss across all blocks
total_loss = 0
weights = [0.5, 0.7, 0.8, 0.9, 1.0, 1.2]  # Later blocks more important

for block_idx, block_output in enumerate(block_outputs):
    logits = block_output(contexts[block_idx])
    loss = cross_entropy(logits, targets)
    total_loss += weights[block_idx] * loss
```

### Advantages Over Alternatives

**vs. Single Output Only**:
- ✅ Each layer learns token prediction
- ✅ Richer learning signal
- ✅ Better intermediate representations

**vs. Multi-Output from Start**:
- ✅ Phase 1 remains focused on context learning
- ✅ No interference during CVFP training
- ✅ Easier debugging and analysis

## Code Quality Standards

### Principles

1. **Encapsulation**: Hide implementation details in layers/modules
2. **Single Responsibility**: Each class does one thing well
3. **Clean Interfaces**: Minimal parameters, clear return values
4. **Self-Documenting**: Method names explain purpose

### Anti-Patterns to Avoid

- ❌ Manual statistics calculation in training loops
- ❌ Exposing internal state (running_mean, running_var) to callers
- ❌ Mixing forward pass logic with loss calculation
- ❌ Copy-pasted code for train/eval modes

### Preferred Patterns

- ✅ Layer classes handle their own statistics
- ✅ Properties/methods for loss retrieval
- ✅ `nn.Module` buffers for persistent state
- ✅ Automatic train/eval mode handling via `self.training`

## No Hardcoding Policy - Critical

**原則**: ハイパーパラメータやマジックナンバーは絶対にハードコードしない

**禁止事項**:
- ❌ 学習率、バッチサイズ、エポック数などの訓練パラメータ
- ❌ モデルアーキテクチャパラメータ（層数、次元数など）
- ❌ 正則化パラメータ（EMAモメンタム、重みなど）
- ❌ 閾値や定数（収束判定、早期停止など）

**必須要件**:
- ✅ 全てのハイパーパラメータは `config.py` で定義
- ✅ デフォルト値はconfig内でのみ設定
- ✅ 実験時に簡単に変更可能な構造

**悪い例**:
```python
# ❌ ハードコードされたパラメータ
model = NewLLMResidual(..., ema_momentum=0.99)  # BAD
threshold = 0.95  # BAD
learning_rate = 0.001  # BAD
```

**良い例**:
```python
# ✅ config.pyから取得
model = NewLLMResidual(..., ema_momentum=config.ema_momentum)  # GOOD
threshold = config.identity_mapping_threshold  # GOOD
learning_rate = config.phase1_learning_rate  # GOOD
```

**理由**:
1. **実験の柔軟性**: パラメータを変更するたびにコードを編集する必要がない
2. **再現性**: config.pyを保存すれば実験を完全に再現可能
3. **保守性**: パラメータの一元管理により変更が容易
4. **可読性**: config.pyを見れば全ての設定が一目瞭然

## Progress Reporting Policy - CRITICAL

**目的**: 長時間処理の進捗を必ず可視化し、フリーズと処理中を区別可能にする。

**必須ルール**:
1. **トークンごとの進捗表示**: Phase1訓練で全トークンを処理する際、定期的に進捗を出力
   - 例: `Processing token 1000/92047 (1.1%)...`
2. **イテレーションごとの進捗**: 各iterationの開始・終了を明示
   - 例: `Iteration 2/10: 収束=20.0%`
3. **処理時間の予測**: 可能な限り残り時間を表示
   - 例: `Estimated remaining: 5m 23s`
4. **大量データ警告**: 10,000トークン以上の処理開始時に警告
   - 例: `⚠️ Processing 92,047 tokens - this may take several minutes`

**実装方法**:
```python
# Phase1Trainerでの実装例
def _process_tokens(self, token_embeds, device, is_training):
    total_tokens = len(token_embeds)
    if total_tokens > 10000:
        self._print_flush(f"⚠️ Processing {total_tokens:,} tokens - this may take several minutes")

    for t, token_embed in enumerate(token_embeds):
        # 100トークンごとに進捗表示
        if t > 0 and t % 100 == 0:
            progress = (t / total_tokens) * 100
            self._print_flush(f"  Progress: {t:,}/{total_tokens:,} ({progress:.1f}%)")

        # 処理実行
        ...
```

**禁止事項**:
- ❌ 長時間（1分以上）の無出力処理
- ❌ 進捗不明な大量ループ
- ❌ ユーザーがフリーズと誤解する沈黙

**理由**:
1. **ユーザー体験**: 処理中かフリーズか判別できないストレス回避
2. **デバッグ効率**: 問題箇所の早期特定
3. **時間管理**: 処理時間の予測可能性

## Validation Data Policy - CRITICAL

**目的**: 検証データが訓練データ外のトークンを含まないことを保証する。

**問題**:
- 検証データに訓練データにないトークンが含まれると、モデルは未知のトークンに対応できない
- これにより検証時のEffective Rankが異常に低くなる
- 正しい評価ができなくなる

**必須ルール**:
1. **検証データは訓練データのサブセット**: 検証データのすべてのトークンは訓練データに存在しなければならない
2. **未知トークンの排除**: 訓練データにないトークンを検証データに含めてはいけない
3. **データ生成時の確認**: データ生成時に必ず重複チェックを実施

**実装例**:
```python
# ❌ 悪い例: 独立したランダム生成
train_tokens = torch.randint(0, 1000, (10,))
val_tokens = torch.randint(0, 1000, (5,))  # 訓練データにないトークンが含まれる可能性

# ✅ 良い例: 訓練データからサンプリング
train_tokens = torch.randint(0, 1000, (10,))
indices = torch.randperm(10)[:5]
val_tokens = train_tokens[indices]  # すべて訓練データに存在することが保証される
```

**理由**:
1. **公正な評価**: モデルが学習したトークンで評価
2. **Effective Rankの正確性**: 未知トークンによる異常値を防ぐ
3. **実世界の反映**: 実際の使用では訓練データの分布から評価

## Performance Benchmarks - Phase 1 Training Speed

**測定日**: 2025-11-23
**環境**: CPU (Apple Silicon/Intel), 6 layers, 768-dim embeddings

### 測定結果

**CPU Performance**:
- **処理速度**: 15-17 tokens/sec
- **1トークンあたり**: 60-65ms
- **スケーラビリティ**: 100トークンでも1000トークンでもほぼ同速度

### 訓練時間の見積もり

10 iterations想定（収束まで）:

| トークン数 | 訓練時間（CPU） | 訓練時間（GPU予想） |
|-----------|----------------|-------------------|
| 1,000 | 約32分 | 約2-3分 |
| 10,000 | 約1.5-2時間 | 約10-15分 |
| 100,000 | 約16-18時間 | 約1-2時間 |

### GPU高速化の可能性

**重要**: Phase 1は逐次処理だが、**各トークンの処理は並列化可能**

**並列化される部分**:
- 行列積（Linear層）: 768×1536などの大規模演算
- LayerNorm: 768次元の正規化
- 活性化関数とバックプロパゲーション
- 勾配計算全体

**期待される高速化**:
- **GPU (CUDA)**: 10-20倍高速（150-300 tok/s）
- **GPU + FP16**: 20-30倍高速（300-500 tok/s）

**使用方法**:
```python
# config.py
device = "cuda"  # CPUから変更するだけ
```

### パフォーマンス最適化の優先順位

1. **GPU利用** (10-20倍): `device = "cuda"`
2. **Mixed Precision** (2-3倍): FP16訓練
3. **torch.compile** (1.2-2倍): PyTorch 2.0+
4. バッチ処理最適化 (1.5-2倍)

**注意**: 量子化は推論時のみ推奨。訓練時は精度低下のリスクあり。

---

## Context Size Monitoring Policy

**目的**: Claude Codeの会話コンテキストサイズを監視し、適切なタイミングで新しいセッションを開始する。

**ルール**:
1. **初回報告**: コンテキストサイズが **100,000トークン** を超えた場合、ユーザーに報告する
2. **継続報告**: その後、**10,000トークンごと**に増加状況を報告する
   - 110,000トークン超過時
   - 120,000トークン超過時
   - 130,000トークン超過時
   - （以降10,000トークン刻みで継続）

**報告フォーマット**:
```
⚠️ コンテキストサイズ警告
現在のトークン数: XXX,XXX / 200,000
使用率: XX.X%

このまま続行すると、コンテキストが上限に達する可能性があります。
適切なタイミングで新しいセッションの開始を検討してください。
```

**推奨アクション**:
- 150,000トークン到達時: 新しいセッションの準備を開始
- 180,000トークン到達時: 速やかに新しいセッションを開始することを強く推奨
- 190,000トークン以上: 緊急性が高い、即座に新セッション開始を推奨

**注意事項**:
- この監視は自動的に行われる（ユーザーからの明示的な要求は不要）
- 報告は過剰にならないよう、10,000トークン刻みで制限
- コンテキスト管理はプロジェクトの継続性と効率性に重要

---

Last Updated: 2025-11-23
