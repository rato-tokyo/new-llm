"""
Phase 2 Trainer: Next-Token Prediction with Separated Architecture (E案)

Phase 1で学習したContextBlockを固定（freeze）し、TokenBlockのみを学習する。

## キャッシュ方式による高速化

### 従来の問題点
- 各トークンごとにContextBlockをforward（遅い）
- バッチ処理が1トークンずつ（GPU効率が悪い）

### 新方式: ContextBlockキャッシュ
1. 全トークンのContextBlock出力を事前計算しキャッシュ
2. TokenBlockはキャッシュを参照してバッチ並列処理

### 期待される高速化
- ContextBlock計算: エポックごとに1回のみ（従来: トークン数×エポック回）
- TokenBlock: 真のバッチ並列処理（GPU効率大幅向上）

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
"""

import torch
import torch.nn as nn
import time
import sys


def print_flush(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


class Phase2Trainer:
    """
    Phase 2 Trainer: TokenBlockのみを学習（キャッシュ方式）

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

        # Embedding凍結オプション
        freeze_embedding = getattr(config, 'phase2_freeze_embedding', False)

        # token_outputを有効化（Embedding凍結オプション付き）
        self.model.unfreeze_token_output(freeze_embedding=freeze_embedding)

        # 学習対象パラメータの確認
        if model.use_separated_architecture:
            # 実際に学習されるパラメータを計算
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            num_trainable = sum(p.numel() for p in trainable_params)
            total_params = sum(p.numel() for p in model.parameters())

            if freeze_embedding:
                print_flush(f"✓ Training TokenBlock only: {num_trainable:,}/{total_params:,} parameters")
            else:
                print_flush(f"✓ Training TokenBlock + token_output: {num_trainable:,}/{total_params:,} parameters")
        else:
            # Legacy: token_outputのみ
            trainable_params = list(model.token_output.parameters())
            num_trainable = sum(p.numel() for p in trainable_params)
            total_params = sum(p.numel() for p in model.parameters())
            print_flush(f"✓ Training token_output only (legacy): {num_trainable:,}/{total_params:,} parameters")

        # Optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.learning_rate
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def _build_context_cache(self, token_ids, device):
        """
        全トークンのContextBlock出力をキャッシュ構築

        Args:
            token_ids: トークンID [seq_len]
            device: デバイス

        Returns:
            context_cache: List of context_outputs for each token
                           各要素は [context_1, ..., context_N] のリスト
            token_embeds: 全トークンの埋め込み [num_tokens, embed_dim * num_input_tokens]
        """
        num_input_tokens = getattr(self.config, 'num_input_tokens', 1)
        input_ids = token_ids[:-1]  # 最後のトークン以外
        num_tokens = len(input_ids)

        # 結果格納用
        context_cache = []  # 各トークンのcontext_outputs
        token_embeds_list = []  # combined_tokens

        with torch.no_grad():
            context = torch.zeros(1, self.model.context_dim, device=device)

            # トークン履歴を初期化（ゼロベクトルで埋める）
            token_history = [torch.zeros(1, self.model.embed_dim, device=device)
                             for _ in range(num_input_tokens - 1)]

            for i in range(num_tokens):
                token_id = input_ids[i]

                # トークン埋め込み
                token_embed = self.model.token_embedding(token_id.unsqueeze(0))
                if isinstance(token_embed, tuple):
                    token_embed = token_embed[0]
                token_embed = self.model.embed_norm(token_embed)

                # 履歴に追加
                token_history.append(token_embed)

                # 最新の num_input_tokens 個を結合
                combined_tokens = torch.cat(token_history[-num_input_tokens:], dim=-1)
                token_embeds_list.append(combined_tokens)

                # ContextBlock forward
                context_outputs = self.model.forward_context_with_intermediates(
                    context, combined_tokens
                )
                context_cache.append(context_outputs)

                # コンテキスト更新
                context = context_outputs[-1]

        # token_embedsをテンソルに変換 [num_tokens, embed_dim * num_input_tokens]
        token_embeds = torch.cat(token_embeds_list, dim=0)

        return context_cache, token_embeds

    def train_epoch(self, token_ids, device, batch_size=None):
        """
        1エポックの訓練（キャッシュ方式）

        処理フロー:
        1. ContextBlock出力を全トークン分キャッシュ（1回のみ）
        2. TokenBlockをバッチ並列処理

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

        target_ids = token_ids[1:]  # 最初のトークン以外（次トークン予測の正解）
        num_tokens = len(target_ids)

        # Step 1: ContextBlockキャッシュ構築
        context_cache, token_embeds = self._build_context_cache(token_ids, device)

        # Step 2: TokenBlockバッチ並列処理
        total_loss = 0.0
        num_layers = self.model.num_layers

        for batch_start in range(0, num_tokens, batch_size):
            batch_end = min(batch_start + batch_size, num_tokens)
            batch_len = batch_end - batch_start

            # バッチ内のデータを取得
            batch_targets = target_ids[batch_start:batch_end]
            batch_token_embeds = token_embeds[batch_start:batch_end]  # [batch, embed_dim * num_input_tokens]

            # バッチ内のcontext_outputsを構築
            # context_cache[i] = [context_1, ..., context_N] (各 [1, context_dim])
            # バッチ用に [batch, context_dim] に変換
            batch_context_list = []
            for layer_idx in range(num_layers):
                layer_contexts = torch.cat(
                    [context_cache[i][layer_idx] for i in range(batch_start, batch_end)],
                    dim=0
                )  # [batch, context_dim]
                batch_context_list.append(layer_contexts)

            # TokenBlock forward（バッチ並列）
            batch_token_out = self.model.forward_token_e(batch_context_list, batch_token_embeds)

            # 予測
            batch_logits = self.model.token_output(batch_token_out)

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
        評価（検証データ）- キャッシュ方式

        Args:
            token_ids: トークンID [seq_len]
            device: デバイス

        Returns:
            avg_loss: 平均損失
            perplexity: パープレキシティ
            accuracy: 正解率
        """
        self.model.eval()

        target_ids = token_ids[1:]
        num_tokens = len(target_ids)

        # ContextBlockキャッシュ構築
        context_cache, token_embeds = self._build_context_cache(token_ids, device)

        total_loss = 0.0
        correct = 0
        num_layers = self.model.num_layers

        with torch.no_grad():
            # 全トークンを一括処理（メモリが許す場合）
            # context_listを構築
            context_list = []
            for layer_idx in range(num_layers):
                layer_contexts = torch.cat(
                    [context_cache[i][layer_idx] for i in range(num_tokens)],
                    dim=0
                )  # [num_tokens, context_dim]
                context_list.append(layer_contexts)

            # TokenBlock forward（全トークン並列）
            token_out = self.model.forward_token_e(context_list, token_embeds)

            # 予測
            logits = self.model.token_output(token_out)  # [num_tokens, vocab_size]

            # 損失（全体）
            total_loss = self.criterion(logits, target_ids).item() * num_tokens

            # 正解率
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == target_ids).sum().item()

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

        print_flush(f"\n{'='*70}")
        print_flush("PHASE 2: Next-Token Prediction Training (キャッシュ方式)")
        print_flush(f"{'='*70}\n")

        print_flush(f"Training tokens: {len(train_token_ids):,}")
        print_flush(f"Validation tokens: {len(val_token_ids):,}")
        print_flush(f"Epochs: {epochs}")
        print_flush(f"Batch size: {batch_size}")
        print_flush(f"Learning rate: {self.learning_rate}")
        print_flush(f"Gradient clip: {self.gradient_clip}")
        print_flush(f"Early stopping patience: {patience}")
        print_flush("")

        if self.model.use_separated_architecture:
            print_flush("Architecture: E案 (ContextBlock + TokenBlock Layer-wise)")
            print_flush("  - ContextBlock: FROZEN + CACHED (エポックごとに1回計算)")
            print_flush("  - TokenBlock: TRAINING (バッチ並列処理)")
            print_flush("  - token_output: TRAINING")
            print_flush("  - E案: TokenBlock Layer i は ContextBlock Layer i の出力を参照")
        else:
            print_flush("Architecture: Legacy CVFP")
            print_flush("  - CVFP blocks: FROZEN")
            print_flush("  - token_output: TRAINING")
        print_flush("")

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
            print_flush(f"Epoch {epoch}/{epochs} [{elapsed:.1f}s]:")
            print_flush(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.2f}")
            print_flush(f"  Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} | Val Acc: {val_acc*100:.2f}%")

            # Track best model and early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                patience_counter = 0
                print_flush(f"  ✓ New best validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                print_flush(f"  ⚠️ No improvement ({patience_counter}/{patience})")

            print_flush("")

            # Early stopping check
            if patience_counter >= patience:
                print_flush(f"⛔ Early stopping triggered at epoch {epoch}")
                print_flush(f"   Val loss hasn't improved for {patience} epochs")
                history['early_stopped'] = True
                history['stopped_epoch'] = epoch
                break

        history['best_epoch'] = best_epoch

        print_flush(f"{'='*70}")
        print_flush("Phase 2 Training Complete")
        print_flush(f"{'='*70}\n")
        print_flush(f"Best epoch: {best_epoch}")
        print_flush(f"Best validation loss: {best_val_loss:.4f}")
        print_flush(f"Best validation PPL: {history['val_ppl'][best_epoch-1]:.2f}")
        print_flush(f"Best validation accuracy: {history['val_acc'][best_epoch-1]*100:.2f}%")
        if history['early_stopped']:
            print_flush(f"Early stopped at epoch: {history['stopped_epoch']}\n")
        else:
            print_flush(f"Completed all {epochs} epochs\n")

        return history
