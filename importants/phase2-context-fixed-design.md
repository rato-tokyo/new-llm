# Phase 2: Context-Fixed Token Prediction Design

**作成日**: 2025-11-25
**更新日**: 2025-11-26
**バージョン**: v2.0 (Complete Context Fixing)

## 概要

Phase 2はPhase 1で学習した文脈表現を活用して、次トークン予測を学習するフェーズです。
**重要な変更（v2.0）**: 文脈ベクトルは**完全固定**（MSE制約ではなく値そのものを置換）します。

## 設計原則

### 1. Phase 1との役割分離

**Phase 1の責任**:
- 文脈表現の固定点学習
- 多様性の高い文脈空間の構築
- CVFPブロックの訓練

**Phase 2の責任**:
- 次トークン予測ヘッド（token_output）の訓練
- CVFPブロックの微調整（token_out経由のみ）
- 文脈情報とトークン埋め込み両方を活用した予測

### 2. 2段階処理 (Stage 1 & Stage 2)

#### Stage 1: 初期化（パラメータ更新なし）

Phase 2開始時に1回だけ実行：

```python
def initialize_target_contexts(self, token_ids, device):
    """Phase 2開始時の固定文脈ベクトルC*を生成"""
    self.model.eval()

    with torch.no_grad():
        context = torch.zeros(1, self.model.context_dim, device=device)
        target_contexts = []  # C*[0], C*[1], ..., C*[n-1]

        for token_id in token_ids:
            token_embed = get_normalized_embedding(self.model, token_id)
            context, token_embed = process_through_blocks(self.model, context, token_embed)
            target_contexts.append(context.squeeze(0))

        self.target_contexts = torch.stack(target_contexts)  # [seq_len, context_dim]
```

**重要ポイント**:
- パラメータ更新なし（`torch.no_grad()`）
- 最初のトークンはゼロベクトルから開始
- 出力されたコンテキストをC*として固定保存
- **以降C*は絶対に変更しない**

#### Stage 2: 学習（パラメータ更新あり）

```python
def train_epoch(self, token_ids, device):
    for i, token_id in enumerate(input_ids):
        # 入力: 固定文脈C*[i-1]とトークン埋め込み
        if i == 0:
            input_context = torch.zeros(1, context_dim, device=device)
        else:
            input_context = self.target_contexts[i-1].unsqueeze(0).detach()

        # CVFPブロック処理
        context_out, token_out = process_through_blocks(model, input_context, token_embed)

        # CRITICAL: context_outをC*[i]で完全に置換（MSE制約ではない）
        fixed_context = self.target_contexts[i].unsqueeze(0).detach()

        # 予測: 固定文脈 + token_out
        combined = torch.cat([fixed_context, token_out], dim=-1)
        logits = model.token_output(combined)

        # 損失は予測損失のみ（context_stability_lossは不要）
        loss = CrossEntropy(logits, target)
```

### 3. 完全固定 vs MSE制約（v1.0からの変更点）

**v1.0（旧設計）: MSE制約**
```python
# 出力されたcontext_outとC*[i]のMSEを損失に追加
context_stability_loss = MSE(context_out, target_contexts)
total_loss = prediction_loss + λ * context_stability_loss
```

**v2.0（新設計）: 完全固定**
```python
# context_outは使わず、C*[i]で直接置換
fixed_context = target_contexts[i].detach()
combined = torch.cat([fixed_context, token_out], dim=-1)
# context_stability_lossは不要
total_loss = prediction_loss
```

**変更理由**:
- MSE制約は「緩い」固定であり、完全な一致を保証しない
- 完全固定により、文脈ベクトルが確実にC*と一致
- 勾配はtoken_out経由のみでCVFPブロックに流れる（意図的な設計）

### 4. 勾配フローの理解

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
- `loss` → `logits` → `token_output`のパラメータ ✅
- `loss` → `logits` → `combined` → `token_out` → CVFPブロック ✅
- `loss` → `logits` → `combined` → `C*[i]` → CVFPブロック ❌ (detach)
- `loss` → `logits` → `combined` → `context_out` ❌ (未使用)

**結果**:
- CVFPブロックは**token_out経由のみ**で更新される
- context_out部分への勾配はゼロ
- これは意図的な設計（context部分の学習を制限）

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

2. **Stage 1: 固定文脈C*を生成**:
   ```python
   trainer.initialize_target_contexts(train_token_ids, device, is_training=True)
   trainer.initialize_target_contexts(val_token_ids, device, is_training=False)
   ```

3. **訓練可能パラメータ**:
   - CVFPブロック: ✅ 訓練可能（ただしtoken_out経由のみ）
   - token_output: ✅ 訓練可能（新規学習）
   - token_embedding: ❌ 固定（GPT-2事前学習済み）

## 記号定義

| 記号 | 意味 |
|------|------|
| `C*[i]` | **固定目標文脈** - Stage 1で計算した、トークンiを処理した後の文脈ベクトル（不変） |
| `context_out` | **学習時の出力文脈** - Stage 2でCVFPブロックが出力する文脈（使用しない） |
| `token_out` | **学習時の出力トークン** - Stage 2でCVFPブロックが出力するトークン表現（予測に使用） |

## パラメータ設定

### config.py

```python
# Phase 2設定
skip_phase1 = True                              # チェックポイントから続行
skip_phase2 = False                             # Phase 2を実行
freeze_context = False                          # CVFPブロックも訓練（token_out経由）
phase2_learning_rate = 0.002                    # Phase 1と同じ
phase2_epochs = 10                              # 訓練エポック数
phase2_patience = 3                             # Early stopping patience
phase2_gradient_clip = 1.0                      # 勾配クリッピング
# Context-Fixed Learning: context_out = C*[i] に完全固定
```

## 実験結果の評価項目

### 必須指標

1. **Prediction Loss**: 次トークン予測の損失
2. **Perplexity**: 言語モデルとしての性能指標
3. **Validation Accuracy**: 検証データでの予測精度

### 記録される履歴

```python
history = {
    'train_loss': [],           # 訓練予測損失
    'train_ppl': [],            # 訓練Perplexity
    'val_loss': [],             # 検証予測損失
    'val_ppl': [],              # 検証Perplexity
    'val_acc': [],              # 検証精度
    'early_stopped': bool,      # Early stoppingが発動したか
    'stopped_epoch': int,       # 停止エポック
    'best_epoch': int           # ベストエポック
}
```

## 理論的根拠

### なぜ文脈ベクトルを完全固定するのか

1. **Phase 1の学習保護**:
   - Phase 1で学習した文脈表現は貴重
   - 予測タスクによって破壊されるべきではない

2. **明確な役割分離**:
   - Phase 1: 文脈表現の学習
   - Phase 2: 予測ヘッドの学習 + CVFPブロックの微調整（token_out経由）

3. **安定した学習**:
   - 文脈が固定されているため、予測タスクに集中できる
   - MSE制約のような「緩い」固定ではなく、完全固定で確実性を確保

### なぜtoken_out経由のみでCVFPを更新するのか

1. **バランスの取れた更新**:
   - context部分の学習を制限し、過度な変更を防ぐ
   - token部分の学習に集中

2. **Phase 1の保護**:
   - context経由の勾配を遮断することで、文脈表現を保護

## 関連ファイル

- **実装**: `src/trainers/phase2.py`
- **モデル定義**: `src/models/llm.py`
- **設定**: `config.py`
- **訓練スクリプト**: `train.py`, `colab.py`
- **設計ドキュメント**: `CLAUDE.md`

## 変更履歴

- **2025-11-25**: 初版作成（Context Stability Loss実装）
- **2025-11-26**: v2.0 - 完全固定方式に変更（MSE制約を廃止）
