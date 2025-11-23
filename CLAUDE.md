# New-LLM Project Guidelines

## Diversity Regularization - Critical Design Specification ⭐ BREAKTHROUGH

### Final Approach: LayerNorm + Orthogonality Constraints

**After extensive experimentation (Nov 2025), we achieved 90%+ Effective Rank using:**

1. **LayerNorm (Value Explosion Prevention)**
   - Prevents residual connection value accumulation
   - Controls scale through normalization
   - Essential for stable training with 4+ layers

2. **Orthogonality Constraints (Diversity Enforcement)**
   - Direct enforcement of orthogonal context vectors
   - Theoretically elegant approach based on linear algebra
   - Natural diversity without artificial constraints

3. **Gradient Clipping (Training Stability)**
   - `max_norm=1.0` prevents gradient explosion
   - Ensures stable convergence

**Results:**
- Training: Effective Rank 14.45/16 (90.3%) ✅
- Validation: Effective Rank 14.04/16 (87.7%) ✅
- No value explosion, stable losses

### Why This Design?

1. **Simplicity**: No complex covariance/orthogonality constraints
2. **Effectiveness**: Dramatic improvement from 6-12% to 80%+ Effective Rank
3. **Stability**: No NaN/Inf issues, controlled norms
4. **Scalability**: Works with any sequence length

### Implementation Details

**Phase1Trainer._train_one_token() method:**

```python
# 1. Compute CVFP loss (context convergence)
cvfp_loss = F.mse_loss(
    F.normalize(new_context, p=2, dim=1),
    F.normalize(previous_context, p=2, dim=1)
)

# 2. Compute orthogonality loss
if len(self.processed_contexts) > 0:
    # Stack recent contexts
    past_contexts = torch.cat(self.processed_contexts[-10:], dim=0)

    # Normalize for cosine similarity
    new_context_norm = F.normalize(new_context, p=2, dim=1)
    past_contexts_norm = F.normalize(past_contexts, p=2, dim=1)

    # Compute similarity (should be close to 0 for orthogonal)
    similarity = torch.matmul(new_context_norm, past_contexts_norm.T)

    # Orthogonality loss
    orthogonality_loss = (similarity ** 2).mean() * self.orthogonality_weight

# 3. Combined loss
total_loss = (1 - dist_reg_weight) * cvfp_loss + dist_reg_weight * orthogonality_loss

# 4. Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Key Configuration (config.py)**:
- `layernorm_mix = 1.0` - Full LayerNorm (critical!)
- `dist_reg_weight = 0.99` - 99% diversity, 1% CVFP
- `phase1_learning_rate = 0.002` - Higher LR for faster convergence
- `num_layers = 4` - Increased from 2 for better expressiveness

### Lessons Learned from Failed Approaches

**What We Tried (and Failed):**

1. **EMA-based Covariance Regularization** ❌
   - Loss scale mismatch (158,508 vs 1.5)
   - Log scaling helped but didn't solve core issue
   - Too complex for marginal benefit

2. **Contrastive Learning with Margin Loss** ❌
   - Value explosion despite margin constraints
   - Instability with larger token counts
   - Complexity didn't justify results

3. **Fixed Dimension Assignment** ⚠️ (Worked but replaced)
   - Achieved 80.3% Effective Rank
   - Hash-based dimension assignment
   - Less theoretically elegant than orthogonality
   - Replaced by orthogonality constraints

4. **Without LayerNorm** ❌
   - Critical mistake: disabled LayerNorm initially
   - Residual connections caused value accumulation
   - Norms reached 10^23 levels
   - **User correctly identified**: "値が爆発するのはなぜですか？layernormを導入していますか？"

5. **Early Orthogonality Attempts** ❌
   - Failed due to lack of normalization
   - Matrix operations unstable without LayerNorm
   - Later succeeded when combined with LayerNorm

**Key Insights:**
- Simpler is better - complex loss functions aren't necessary
- LayerNorm is essential for residual networks
- Orthogonality constraints are theoretically elegant and effective
- Gradient clipping prevents training instability
- User insight was critical: orthogonality works WITH LayerNorm

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
