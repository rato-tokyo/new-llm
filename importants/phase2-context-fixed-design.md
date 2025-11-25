# Phase 2: Context-Fixed Token Prediction Design

**作成日**: 2025-11-25
**バージョン**: v1.0 (Context Stability Loss)

## 概要

Phase 2はPhase 1で学習した文脈表現を活用して、次トークン予測を学習するフェーズです。
重要な特徴として、**文脈ベクトルをPhase 2開始時の値に固定**することで、Phase 1の学習を保護しています。

## 設計原則

### 1. Phase 1との役割分離

**Phase 1の責任**:
- 文脈表現の固定点学習
- 多様性の高い文脈空間の構築
- CVFPブロックの訓練

**Phase 2の責任**:
- 次トークン予測ヘッド（token_output）の訓練
- CVFPブロックの微調整（文脈ベクトル固定の制約付き）
- 文脈情報とトークン埋め込み両方を活用した予測

### 2. Token Output初期化方針

**Phase 1中**:
```python
# llm.py - __init__
self.token_output = nn.Linear(context_dim + embed_dim, vocab_size)

# ゼロ初期化 + 勾配無効化
with torch.no_grad():
    self.token_output.weight.fill_(0)
    self.token_output.bias.fill_(0)
self.token_output.weight.requires_grad = False
self.token_output.bias.requires_grad = False
```

**理由**:
- Phase 1ではtoken_outputは一切使用されない
- ゼロ初期化により出力も自然にゼロになる
- 文脈処理（CVFPブロック）に影響を与えない

**Phase 2開始時**:
```python
# phase2.py - __init__
self.model.token_output.weight.requires_grad = True
self.model.token_output.bias.requires_grad = True
```

### 3. 文脈ベクトル固定の実装

#### 教師データ生成

Phase 2開始時に1回だけ実行：

```python
def initialize_target_contexts(self, token_ids, device, is_training=True):
    """Phase 2開始時の文脈ベクトルを教師データとして生成"""
    self.model.eval()

    with torch.no_grad():
        # トークン埋め込み
        token_embeds = self.model.token_embedding(token_ids.unsqueeze(0).to(device))
        token_embeds = self.model.embed_norm(token_embeds).squeeze(0)

        # 文脈伝播（Phase 1と同じ）
        context = torch.zeros(1, self.model.context_dim, device=device)
        target_contexts = []

        for token_embed in token_embeds:
            for block in self.model.blocks:
                context, token_embed_out = block(token_embed.unsqueeze(0), context)
            target_contexts.append(context.squeeze(0))

        target_contexts = torch.stack(target_contexts)  # [seq_len, context_dim]

    # 保存
    if is_training:
        self.target_contexts_train = target_contexts
    else:
        self.target_contexts_val = target_contexts
```

#### 損失関数

```python
# 1. 予測損失（次トークン予測）
prediction_loss = CrossEntropyLoss(logits, target_ids)

# 2. 文脈安定性損失（Phase 2開始時の文脈と比較）
context_stability_loss = F.mse_loss(current_contexts, target_contexts)

# 総合損失
total_loss = prediction_loss + context_stability_weight * context_stability_loss
```

**context_stability_weight**:
- デフォルト値: `1.0`（予測損失と同等の重み）
- 設定: `config.py` の `phase2_context_stability_weight`

## 訓練フロー

### Phase 2開始前の状態

```
CVFPブロック: Phase 1で訓練済み
token_embedding: GPT-2事前学習済み（固定）
token_output: ゼロ初期化（未使用）
```

### Phase 2開始時

1. **token_outputを有効化**:
   ```python
   model.token_output.weight.requires_grad = True
   model.token_output.bias.requires_grad = True
   ```

2. **教師データ生成**:
   ```python
   trainer.initialize_target_contexts(train_token_ids, device, is_training=True)
   trainer.initialize_target_contexts(val_token_ids, device, is_training=False)
   ```

3. **訓練可能パラメータ**:
   - CVFPブロック: ✅ 訓練可能（文脈安定性損失による制約付き）
   - token_output: ✅ 訓練可能（新規学習）
   - token_embedding: ❌ 固定（GPT-2事前学習済み）

### 各エポックの訓練

```python
for epoch in range(epochs):
    # 順伝播
    context = torch.zeros(1, context_dim, device=device)
    all_logits = []
    all_contexts = []

    for token_embed in token_embeds:
        context = context.detach()  # 勾配遮断

        # CVFPブロック処理
        for block in model.blocks:
            context, token_embed = block(token_embed, context)

        all_contexts.append(context)

        # 予測（連結版）
        combined = torch.cat([context, token_embed], dim=-1)
        logits = model.token_output(combined)
        all_logits.append(logits)

    # 損失計算
    prediction_loss = CrossEntropyLoss(all_logits, targets)
    context_stability_loss = MSE(all_contexts, target_contexts)
    total_loss = prediction_loss + lambda * context_stability_loss

    # 最適化
    total_loss.backward()
    optimizer.step()
```

## 重要な制約と特徴

### 1. 文脈伝播

**Phase 1と同じ動作**:
- 最初のトークンはゼロベクトルから開始
- 以降のトークンは前の文脈を引き継ぐ
- 文脈は全トークンを通して連続的に伝播

### 2. 勾配遮断

```python
context = context.detach()
```

**目的**:
- 系列全体への勾配伝播を防ぐ
- 訓練の安定化
- Phase 1の固定点学習を保護

### 3. 連結予測

```python
combined = torch.cat([context, token_embed], dim=-1)  # [1, 1536]
logits = model.token_output(combined)  # [1, vocab_size]
```

**理由**:
- Context: Phase 1で学習した文脈情報（768次元）
- Token Embed: CVFPブロック出力のトークン部分（768次元）
- 両方の情報を活用して予測精度を向上

### 4. CVFPブロックの更新方針

**freeze_context = False（デフォルト）**:
- CVFPブロックは訓練可能
- ただし、出力される文脈ベクトルは固定（MSE損失による制約）
- 重みは変化できるが、最終出力は変わらない

**意図**:
- Phase 1の文脈表現を保護
- 予測タスクに合わせた微調整を許可
- 柔軟性と安定性のバランス

## パラメータ設定

### config.py

```python
# Phase 2設定
skip_phase1 = True                              # チェックポイントから続行
skip_phase2 = False                             # Phase 2を実行
freeze_context = False                          # CVFPブロックも訓練
phase2_learning_rate = 0.002                    # Phase 1と同じ
phase2_epochs = 10                              # 訓練エポック数
phase2_gradient_clip = 1.0                      # 勾配クリッピング
phase2_context_stability_weight = 1.0           # 文脈安定性損失の重み
```

### 推奨設定

**context_stability_weight**:
- `1.0`: 予測損失と文脈安定性を同等に扱う（デフォルト）
- `0.5`: 予測精度を優先
- `2.0`: 文脈固定を優先

**freeze_context**:
- `False`: CVFPブロックも微調整（推奨）
- `True`: token_outputのみ訓練（完全凍結）

## 実験結果の評価項目

### 必須指標

1. **Prediction Loss**: 次トークン予測の損失
2. **Perplexity**: 言語モデルとしての性能指標
3. **Context Stability Loss**: 文脈ベクトルの変化量
4. **Validation Accuracy**: 検証データでの予測精度

### 記録される履歴

```python
history = {
    'train_loss': [],           # 訓練予測損失
    'train_ppl': [],            # 訓練Perplexity
    'train_context_loss': [],   # 訓練文脈安定性損失
    'val_loss': [],             # 検証予測損失
    'val_ppl': [],              # 検証Perplexity
    'val_acc': []               # 検証精度
}
```

## 理論的根拠

### なぜ文脈ベクトルを固定するのか

1. **Phase 1の学習保護**:
   - Phase 1で学習した文脈表現は貴重
   - 予測タスクによって破壊されるべきではない

2. **役割の明確化**:
   - Phase 1: 文脈表現の学習
   - Phase 2: 予測ヘッドの学習
   - 各フェーズが独立した責任を持つ

3. **段階的学習**:
   - 固定点学習 → 予測学習の順序
   - 各段階で異なる目的関数を最適化
   - より安定した学習プロセス

### なぜ連結予測なのか

1. **情報の最大活用**:
   - Context: 文脈情報（Phase 1で学習）
   - Token Embed: 局所表現（CVFPブロック出力）
   - 両方を使うことで予測精度向上

2. **Phase 1の意義**:
   - Phase 1で学習した文脈を予測に活用
   - 文脈情報を捨てるのはもったいない

3. **次元数の増加**:
   - 768 + 768 = 1536次元
   - より豊富な情報で予測

## 既知の制限と将来の改善

### 現在の実装の制限

1. **文脈ベクトルの完全固定ではない**:
   - MSE損失による"緩い"制約
   - 完全にゼロにはならない

2. **メモリ使用量**:
   - target_contextsを全トークン分保存
   - 大規模データセットでは要注意

### 将来の改善案

1. **完全凍結オプション**:
   - CVFPブロックを完全に固定
   - 文脈ベクトルの完全な保護

2. **適応的重み調整**:
   - エポックごとにcontext_stability_weightを変更
   - 初期は文脈固定、後期は予測精度優先

3. **選択的固定**:
   - 特定の層のみ固定
   - 浅い層は固定、深い層は微調整

## 関連ファイル

- **実装**: `src/trainers/phase2.py`
- **モデル定義**: `src/models/llm.py`
- **設定**: `config.py`
- **訓練スクリプト**: `train.py`
- **設計ドキュメント**: `CLAUDE.md`

## 変更履歴

- **2025-11-25**: 初版作成（Context Stability Loss実装）
