# New-LLM Project Guidelines

## Distribution Regularization - Critical Design Specification

### Philosophy: Token-wise Normalization

**IMPORTANT**: Distribution regularization MUST be applied **per-token** (online), not across all tokens (batch).

### Why Token-wise?

1. **Theoretical Correctness**: Each token processes sequentially in a language model
2. **Online Learning**: Can't wait for all 92,047 tokens to compute statistics
3. **Prevents Trivial Solutions**: Batch normalization allows identity mapping convergence
4. **Scalability**: Works with any sequence length

### Implementation Method: Exponential Moving Average (EMA)

Use running statistics updated per token:

```python
# For each token t:
running_mean = momentum * running_mean + (1 - momentum) * context[t].mean()
running_var = momentum * running_var + (1 - momentum) * context[t].var()

# Penalize deviation from N(0,1)
dist_loss = (running_mean ** 2) + ((running_var - 1.0) ** 2)
```

**Parameters**:
- Momentum: 0.99 (typical for EMA)
- Update: Every token during forward pass
- Scope: Per-dimension statistics across tokens

### Object-Oriented Design for Clean Implementation

#### Current Problem: Scattered Logic
- Distribution regularization mixed with training loop
- Forward pass doesn't handle normalization internally
- Statistics calculation exposed to caller

#### Proposed Solution: Layer-based Encapsulation

```python
class CVFPLayer(nn.Module):
    """
    Context update layer with built-in distribution tracking
    """
    def __init__(self, context_dim, embed_dim, hidden_dim, use_dist_reg=True):
        super().__init__()
        self.fnn = nn.Linear(...)

        # EMA statistics (if distribution regularization enabled)
        if use_dist_reg:
            self.register_buffer('running_mean', torch.zeros(context_dim))
            self.register_buffer('running_var', torch.ones(context_dim))
            self.momentum = 0.99

    def forward(self, context, token_embed):
        # Update context
        new_context = self._update_context(context, token_embed)

        # Update running statistics (hidden from caller)
        if self.training and self.use_dist_reg:
            self._update_statistics(new_context)

        return new_context

    def _update_statistics(self, context):
        """Hidden implementation detail"""
        with torch.no_grad():
            batch_mean = context.mean(dim=0)
            batch_var = context.var(dim=0, unbiased=False)

            self.running_mean = self.momentum * self.running_mean + \
                               (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + \
                              (1 - self.momentum) * batch_var

    def get_distribution_loss(self):
        """Calculate loss based on accumulated statistics"""
        mean_penalty = (self.running_mean ** 2).mean()
        var_penalty = ((self.running_var - 1.0) ** 2).mean()
        return mean_penalty + var_penalty
```

#### Benefits of This Design:

1. **Encapsulation**: Statistics tracking hidden inside layer
2. **Automatic Updates**: No manual `mean()` / `var()` in training loop
3. **Clean Interface**: Caller just does `context = layer(context, token)`
4. **Testable**: Easy to test distribution tracking separately
5. **Reusable**: Same pattern for other normalization schemes

### Migration Strategy

1. Create `CVFPLayer` class in `src/models/layers.py`
2. Refactor `new_llm_residual.py` to use `CVFPLayer`
3. Update `phase1.py` to use `model.get_distribution_loss()`
4. Remove manual statistics calculation from training loop

### Expected Improvements

- **Prevent Identity Mapping**: Token-wise normalization forces diversity
- **Better Convergence**: Running statistics provide stable gradients
- **Cleaner Code**: ~50 lines removed from training loop
- **Easier Debugging**: Layer-level loss inspection

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
