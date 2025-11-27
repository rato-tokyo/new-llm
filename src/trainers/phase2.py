"""
Phase 2 Trainer: Next-Token Prediction with Separated Architecture (E案)

Phase 1で学習したContextBlockを固定（freeze）し、TokenBlockのみを学習する。

## 分離アーキテクチャ（E案）- レイヤー対応版

### ContextBlock（Phase 1で学習済み、Phase 2でfreeze）
- 入力: [context_in, token_embed]
- 出力: 各レイヤーのcontext出力 [context_1, context_2, ..., context_N]
- Phase 2では重みが固定されているため、同じ入力 → 同じ出力が保証される

### TokenBlock（Phase 2で学習）
- **E案**: TokenBlock Layer i は ContextBlock Layer i の出力を参照
- 入力: [context_i, token_{i-1}]
- 出力: token_i
- 最終token_Nから次トークンを予測

## E案のアーキテクチャ

```
ContextBlock (frozen):
  Layer 1: [context_0, token_embed] → context_1
  Layer 2: [context_1, token_embed] → context_2
  Layer 3: [context_2, token_embed] → context_3 (= C*)

TokenBlock (学習):
  Layer 1: [context_1, token_embed] → token_1
  Layer 2: [context_2, token_1]     → token_2
  Layer 3: [context_3, token_2]     → token_3 (= token_out)
```

## 設計の利点

1. **段階的文脈情報**: 浅いレイヤーでは浅い文脈、深いレイヤーでは深い文脈を使用
2. **C*の保持**: ContextBlockはfrozenなので、Phase 1で学習した文脈表現が維持
3. **Transformerとの類似性**: 各レイヤーで異なる深さの表現を参照
4. **物理的分離維持**: ContextBlockとTokenBlockは別の重み行列のまま

## 処理フロー（E案）

```python
# 各トークンの処理
token_embed = get_embedding(token_id)

# Step 1: ContextBlock（frozen）- 各レイヤーの出力を取得
with torch.no_grad():
    context_outputs = context_block.forward_with_intermediates(context, token_embed)
    # context_outputs = [context_1, context_2, context_3]

# Step 2: TokenBlock（学習）- 対応するレイヤーのcontextを使用
token_out = token_block.forward_with_contexts(context_outputs, token_embed)

# Step 3: 予測
logits = token_output(token_out)
loss = CrossEntropy(logits, target)
```
"""

import torch
import torch.nn as nn
import time


class Phase2Trainer:
    """
    Phase 2 Trainer: TokenBlockのみを学習

    分離アーキテクチャ:
    - ContextBlock: frozen（Phase 1で学習済み）
    - TokenBlock: 学習対象
    - token_output: 学習対象

    Args:
        model: LLMモデル（分離アーキテクチャ）
        config: ResidualConfig（設定）
    """

    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.learning_rate = config.phase2_learning_rate
        self.gradient_clip = config.phase2_gradient_clip

        # ContextBlockをfreeze
        self.model.freeze_context_block()

        # token_outputを有効化
        self.model.unfreeze_token_output()

        # 学習対象パラメータの確認
        if model.use_separated_architecture:
            # TokenBlock + token_output
            trainable_params = list(model.token_block.parameters()) + \
                              list(model.token_output.parameters())
            num_trainable = sum(p.numel() for p in trainable_params)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✓ Training TokenBlock + token_output: {num_trainable:,}/{total_params:,} parameters")
        else:
            # Legacy: token_outputのみ
            trainable_params = list(model.token_output.parameters())
            num_trainable = sum(p.numel() for p in trainable_params)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✓ Training token_output only (legacy): {num_trainable:,}/{total_params:,} parameters")

        # Optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.learning_rate
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def train_epoch(self, token_ids, device, batch_size=None):
        """
        1エポックの訓練（ミニバッチ処理）- E案

        E案の処理フロー:
        1. ContextBlock(frozen): 各レイヤーの出力を取得 [context_1, ..., context_N]
        2. TokenBlock(学習): Layer i は context_i を参照して token を更新
        3. Prediction: token_out → logits

        Args:
            token_ids: トークンID [seq_len]
            device: デバイス
            batch_size: ミニバッチサイズ

        Returns:
            avg_loss: 平均損失
            perplexity: パープレキシティ
        """
        self.model.train()

        if batch_size is None:
            batch_size = self.config.phase2_batch_size

        # Input: all tokens except last
        # Target: all tokens except first (next token prediction)
        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]

        num_tokens = len(input_ids)
        total_loss = 0.0

        # シーケンシャル処理でコンテキストを伝播
        context = torch.zeros(1, self.model.context_dim, device=device)

        for batch_start in range(0, num_tokens, batch_size):
            batch_end = min(batch_start + batch_size, num_tokens)
            batch_len = batch_end - batch_start

            # バッチ内のトークンを処理
            batch_logits_list = []
            batch_targets = target_ids[batch_start:batch_end]

            for i in range(batch_start, batch_end):
                token_id = input_ids[i]

                # トークン埋め込み
                token_embed = self.model.token_embedding(token_id.unsqueeze(0))
                if isinstance(token_embed, tuple):
                    token_embed = token_embed[0]
                token_embed = self.model.embed_norm(token_embed)

                # Step 1: ContextBlock（frozen）- 各レイヤーの出力を取得（E案）
                with torch.no_grad():
                    context_outputs = self.model.forward_context_with_intermediates(
                        context, token_embed
                    )
                    # context_outputs = [context_1, context_2, ..., context_N]

                # Step 2: TokenBlock（学習）- 対応するレイヤーのcontextを使用（E案）
                token_out = self.model.forward_token_e(context_outputs, token_embed)

                # Step 3: 予測
                logits = self.model.token_output(token_out)
                batch_logits_list.append(logits)

                # コンテキストを更新（最終レイヤーの出力を使用、detachして勾配切断）
                context = context_outputs[-1].detach()

            # バッチのlogitsを結合
            batch_logits = torch.cat(batch_logits_list, dim=0)

            # 損失計算と更新
            self.optimizer.zero_grad()
            batch_loss = self.criterion(batch_logits, batch_targets)
            batch_loss.backward()

            # 勾配クリッピング
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    self.gradient_clip
                )

            self.optimizer.step()

            # 損失を累積
            total_loss += batch_loss.item() * batch_len

        # 平均損失とパープレキシティ
        avg_loss = total_loss / num_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return avg_loss, perplexity

    def evaluate(self, token_ids, device):
        """
        評価（検証データ）- E案

        Args:
            token_ids: トークンID [seq_len]
            device: デバイス

        Returns:
            avg_loss: 平均損失
            perplexity: パープレキシティ
            accuracy: 正解率
        """
        self.model.eval()

        input_ids = token_ids[:-1]
        target_ids = token_ids[1:]
        num_tokens = len(input_ids)

        total_loss = 0.0
        correct = 0

        with torch.no_grad():
            context = torch.zeros(1, self.model.context_dim, device=device)

            for i in range(num_tokens):
                token_id = input_ids[i]

                # トークン埋め込み
                token_embed = self.model.token_embedding(token_id.unsqueeze(0))
                if isinstance(token_embed, tuple):
                    token_embed = token_embed[0]
                token_embed = self.model.embed_norm(token_embed)

                # Step 1: ContextBlock - 各レイヤーの出力を取得（E案）
                context_outputs = self.model.forward_context_with_intermediates(
                    context, token_embed
                )

                # Step 2: TokenBlock - 対応するレイヤーのcontextを使用（E案）
                token_out = self.model.forward_token_e(context_outputs, token_embed)

                # Step 3: 予測
                logits = self.model.token_output(token_out)

                # 損失
                target = target_ids[i].unsqueeze(0)
                loss = self.criterion(logits, target)
                total_loss += loss.item()

                # 正解率
                pred = torch.argmax(logits, dim=-1)
                if pred.item() == target.item():
                    correct += 1

                # コンテキスト更新（最終レイヤーの出力を使用）
                context = context_outputs[-1]

        avg_loss = total_loss / num_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        accuracy = correct / num_tokens

        return avg_loss, perplexity, accuracy

    def train_full(self, train_token_ids, val_token_ids, device, epochs=None, patience=None, batch_size=None):
        """
        フル訓練ループ（早期停止あり）

        Args:
            train_token_ids: 訓練トークンID
            val_token_ids: 検証トークンID
            device: デバイス
            epochs: エポック数
            patience: 早期停止の忍耐回数
            batch_size: ミニバッチサイズ

        Returns:
            history: 訓練履歴
        """
        if epochs is None:
            epochs = self.config.phase2_epochs
        if patience is None:
            patience = self.config.phase2_patience
        if batch_size is None:
            batch_size = self.config.phase2_batch_size

        print(f"\n{'='*70}")
        print("PHASE 2: Next-Token Prediction Training (E案 - レイヤー対応版)")
        print(f"{'='*70}\n")

        print(f"Training tokens: {len(train_token_ids):,}")
        print(f"Validation tokens: {len(val_token_ids):,}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Gradient clip: {self.gradient_clip}")
        print(f"Early stopping patience: {patience}")
        print()

        if self.model.use_separated_architecture:
            print("Architecture: E案 (ContextBlock + TokenBlock Layer-wise)")
            print("  - ContextBlock: FROZEN (Phase 1で学習済み)")
            print("  - TokenBlock: TRAINING")
            print("  - token_output: TRAINING")
            print("  - E案: TokenBlock Layer i は ContextBlock Layer i の出力を参照")
        else:
            print("Architecture: Legacy CVFP")
            print("  - CVFP blocks: FROZEN")
            print("  - token_output: TRAINING")
        print()

        history = {
            'train_loss': [],
            'train_ppl': [],
            'val_loss': [],
            'val_ppl': [],
            'val_acc': [],
            'early_stopped': False,
            'stopped_epoch': None,
            'best_epoch': None
        }

        best_val_loss = float('inf')
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss, train_ppl = self.train_epoch(
                train_token_ids, device, batch_size=batch_size
            )

            # Validate
            val_loss, val_ppl, val_acc = self.evaluate(val_token_ids, device)

            elapsed = time.time() - start_time

            # Record history
            history['train_loss'].append(train_loss)
            history['train_ppl'].append(train_ppl)
            history['val_loss'].append(val_loss)
            history['val_ppl'].append(val_ppl)
            history['val_acc'].append(val_acc)

            # Print progress
            print(f"Epoch {epoch}/{epochs} [{elapsed:.1f}s]:")
            print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
            print(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Val Acc: {val_acc*100:.2f}%")

            # Track best model and early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                print(f"  ✓ New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  ⚠️ No improvement ({patience_counter}/{patience})")

            print()

            # Early stopping check
            if patience_counter >= patience:
                print(f"⛔ Early stopping triggered at epoch {epoch}")
                print(f"   Val loss hasn't improved for {patience} epochs")
                history['early_stopped'] = True
                history['stopped_epoch'] = epoch
                break

        history['best_epoch'] = best_epoch

        print(f"{'='*70}")
        print("Phase 2 Training Complete")
        print(f"{'='*70}\n")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation PPL: {history['val_ppl'][best_epoch-1]:.2f}")
        print(f"Best validation accuracy: {history['val_acc'][best_epoch-1]*100:.2f}%")
        if history['early_stopped']:
            print(f"Early stopped at epoch: {history['stopped_epoch']}\n")
        else:
            print(f"Completed all {epochs} epochs\n")

        return history
