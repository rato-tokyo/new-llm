# New-LLM Project Guidelines

## Distribution Regularization - Critical Design Specification

### Core Requirements - MANDATORY

**必須仕様（MUST-HAVE)**:
1. **オンライン学習**: トークンを逐次処理しながら学習（バッチ処理禁止）
2. **指数移動平均（EMA）**: メモリ効率のため統計はEMAで追跡（O(d)メモリのみ）
3. **重み更新方式**: 後処理変換ではなく、モデル重みを直接学習

### Philosophy: Token-wise Normalization with EMA

**IMPORTANT**: Distribution regularization MUST be applied **per-token** (online), using Exponential Moving Average for memory efficiency.

### Why This Design?

1. **Memory Efficiency**: EMA requires only O(d) memory regardless of sequence length
2. **Online Learning**: Process tokens sequentially without waiting for full batch
3. **Weight Learning**: Model learns to output normalized distributions, not just post-process
4. **Scalability**: Works with sequences of any length without memory explosion

### Implementation Method: EMA-based Weight Learning

**核心的アプローチ**:

```python
# 1. EMA統計の追跡（メモリO(d)）
running_mean = momentum * running_mean + (1 - momentum) * current_context
running_var = momentum * running_var + (1 - momentum) * current_context.var()

# 2. 正解データ生成（現在の統計から）
target = (current_context - running_mean) / sqrt(running_var)

# 3. 重み学習（モデルが正規分布を出力するよう学習）
loss = MSE(current_context, target.detach())
loss.backward()
optimizer.step()
```

**Key Design Points**:
- **Memory**: O(d) only - dimension size, not sequence length
- **Online**: Process each token immediately, no batching required
- **Weight Update**: Model learns to output N(0,1), not just post-process
- **EMA Momentum**: 0.99 for stability, configurable in config.py

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
