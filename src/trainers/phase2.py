"""
Phase 2 Trainer: Next-Token Prediction with Context-Fixed Learning

Trains the prediction head to predict next tokens using fixed context vectors.

Key Design (Updated 2025-11-26):
- Stage 1 (Initialization): Compute fixed context vectors C*[i] from training data
  - Process all tokens with zero-vector start
  - NO parameter updates during this stage
  - Save output contexts as fixed targets (C*)
- Stage 2 (Training): Train with fixed context
  - Input: [C*[i-1], token_embed[i]]
  - Output: [context_out, token_out]
  - context_out is replaced with C*[i].detach() (complete fixing, no MSE constraint)
  - token_out used for prediction
  - Gradients flow through token_out only (CVFPパラメータは更新されるが、context経由のみ制限)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def get_normalized_embedding(model, token_id):
    """
    トークンIDから正規化された埋め込みを取得（共通関数）

    CRITICAL: この関数を必ず使用すること！
    - token_embedding + embed_norm の組み合わせが必須
    - この共通関数を使わないとバグの原因になる

    Args:
        model: LLMモデル
        token_id: 単一トークンID [1] or スカラー

    Returns:
        token_embed: 正規化された埋め込み [1, embed_dim]
    """
    # トークン埋め込みを取得
    token_embed = model.token_embedding(token_id.unsqueeze(0) if token_id.dim() == 0 else token_id)

    # タプルの場合の処理（一部のembeddingがタプルを返す場合）
    if isinstance(token_embed, tuple):
        token_embed = token_embed[0]

    # embed_normを適用（CRITICAL: これを忘れるとバグになる）
    token_embed = model.embed_norm(token_embed)

    return token_embed


def process_through_blocks(model, context, token_embed, freeze_context=False):
    """
    CVFPブロックを通過させる（共通関数）

    CRITICAL: 引数順序は (context, token_embed)

    Args:
        model: LLMモデル
        context: コンテキストベクトル [1, context_dim]
        token_embed: トークン埋め込み [1, embed_dim]
        freeze_context: Trueの場合、勾配を計算しない

    Returns:
        context: 更新されたコンテキスト [1, context_dim]
        token_embed: 更新されたトークン埋め込み [1, embed_dim]
    """
    for block in model.blocks:
        if freeze_context:
            with torch.no_grad():
                context, token_embed = block(context, token_embed)
        else:
            context, token_embed = block(context, token_embed)

    return context, token_embed


class Phase2Trainer:
    """
    Phase 2 Trainer for next-token prediction

    Args:
        model: LLM model with prediction head
        learning_rate: Learning rate for optimizer
        freeze_context: If True, freeze CVFP layers (only train prediction head)
        gradient_clip: Gradient clipping value (None = no clipping)
    """

    def __init__(
        self,
        model,
        learning_rate=0.0001,
        freeze_context=False,
        gradient_clip=1.0
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.freeze_context = freeze_context
        self.gradient_clip = gradient_clip

        # Phase 2用: token_outputを有効化（Phase 1ではゼロ＋無効化されている）
        self.model.token_output.weight.requires_grad = True
        self.model.token_output.bias.requires_grad = True

        # Freeze CVFP layers if requested
        if freeze_context:
            self._freeze_cvfp_layers()
        else:
            # All parameters are trainable (full fine-tuning)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✓ Full model fine-tuning: {trainable_params:,}/{total_params:,} parameters trainable")
            print(f"✓ Context-Fixed Learning: context_out replaced with C*[i] (complete fixing)")

        # Optimizer (only trainable parameters)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        # Target contexts for stability (initialized in train_full)
        self.target_contexts_train = None
        self.target_contexts_val = None

    def _freeze_cvfp_layers(self):
        """Freeze CVFP layers (keep only prediction head trainable)"""
        # Freeze embeddings
        for param in self.model.token_embedding.parameters():
            param.requires_grad = False

        # Freeze CVFP blocks
        for block in self.model.blocks:
            for param in block.parameters():
                param.requires_grad = False

        # Keep prediction head trainable
        for param in self.model.token_output.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ CVFP layers frozen: {trainable_params:,}/{total_params:,} parameters trainable (prediction head only)")

    def initialize_target_contexts(self, token_ids, device, is_training=True):
        """
        Phase 2開始時に文脈ベクトルの教師データを生成

        Args:
            token_ids: Token IDs [seq_len]
            device: Device to use
            is_training: Whether this is training data

        Returns:
            target_contexts: Target context vectors [seq_len, context_dim]
        """
        self.model.eval()

        with torch.no_grad():
            # 文脈伝播（Phase 1と同じ）
            context = torch.zeros(1, self.model.context_dim, device=device)
            target_contexts = []

            for token_id in token_ids:
                # Get normalized token embedding (共通関数を使用)
                token_embed = get_normalized_embedding(self.model, token_id)

                # Process through CVFP blocks (共通関数を使用)
                context, token_embed = process_through_blocks(
                    self.model, context, token_embed, freeze_context=False
                )
                target_contexts.append(context.squeeze(0))

            target_contexts = torch.stack(target_contexts)  # [seq_len, context_dim]

        # 保存
        if is_training:
            self.target_contexts_train = target_contexts
        else:
            self.target_contexts_val = target_contexts

        self.model.train()

        label = "Training" if is_training else "Validation"
        print(f"✓ {label} target contexts initialized: {target_contexts.shape}")

        return target_contexts

    def train_epoch(self, token_ids, device):
        """
        Train one epoch on token sequence with context-fixed learning

        CRITICAL DESIGN (Updated 2025-11-26):
        - Input: [C*[i-1], token_embed[i]] - fixed context from initialization
        - Output: [context_out, token_out] from CVFP blocks
        - context_out is replaced with C*[i].detach() (complete fixing)
        - Prediction from concatenated C*[i] + token_out
        - Gradients flow through token_out only (CVFP params still updated via token_out)

        Args:
            token_ids: Token IDs [seq_len]
            device: Device to use

        Returns:
            avg_loss: Average loss for this epoch
            perplexity: Perplexity metric
        """
        self.model.train()

        # Input: all tokens except last
        # Target: all tokens except first
        input_ids = token_ids[:-1]  # [seq_len - 1]
        target_ids = token_ids[1:]  # [seq_len - 1]

        # Process all tokens with context-fixed learning
        all_logits = []
        for i, token_id in enumerate(input_ids):
            # Get fixed input context C*[i-1] (or zero for first token)
            if i == 0:
                input_context = torch.zeros(1, self.model.context_dim, device=device)
            else:
                # Use fixed target context from previous position
                input_context = self.target_contexts_train[i-1].unsqueeze(0).detach()

            # Get normalized token embedding
            token_embed = get_normalized_embedding(self.model, token_id)

            # Process through CVFP blocks
            # Input: [C*[i-1], token_embed[i]]
            # Output: [context_out, token_out]
            context_out, token_out = process_through_blocks(
                self.model, input_context, token_embed, self.freeze_context
            )

            # CRITICAL: Replace context_out with fixed C*[i] (complete fixing)
            # Gradients flow through token_out only
            fixed_context = self.target_contexts_train[i].unsqueeze(0).detach()

            # Predict next token from concatenated fixed_context + token_out
            combined = torch.cat([fixed_context, token_out], dim=-1)  # [1, context_dim + embed_dim]
            logits = self.model.token_output(combined)  # [1, vocab_size]
            all_logits.append(logits)

        # Stack all logits
        all_logits = torch.cat(all_logits, dim=0)  # [seq_len - 1, vocab_size]

        # Compute loss (prediction loss only, no context stability loss needed)
        self.optimizer.zero_grad()
        prediction_loss = self.criterion(all_logits, target_ids)

        # Backward pass
        prediction_loss.backward()

        # Gradient clipping
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.gradient_clip
            )

        self.optimizer.step()

        # Calculate metrics
        avg_loss = prediction_loss.item()
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return avg_loss, perplexity

    def evaluate(self, token_ids, device, target_contexts=None):
        """
        Evaluate on token sequence with context-fixed learning

        Args:
            token_ids: Token IDs [seq_len]
            device: Device to use
            target_contexts: Fixed target contexts (if None, use self.target_contexts_val)

        Returns:
            avg_loss: Average loss
            perplexity: Perplexity metric
            accuracy: Token prediction accuracy
        """
        self.model.eval()

        # Use provided target_contexts or default to validation contexts
        if target_contexts is None:
            target_contexts = self.target_contexts_val

        # Input: all tokens except last
        # Target: all tokens except first
        input_ids = token_ids[:-1]  # [seq_len - 1]
        target_ids = token_ids[1:]  # [seq_len - 1]

        with torch.no_grad():
            # Process all tokens with context-fixed learning
            all_logits = []
            for i, token_id in enumerate(input_ids):
                # Get fixed input context C*[i-1] (or zero for first token)
                if i == 0:
                    input_context = torch.zeros(1, self.model.context_dim, device=device)
                else:
                    input_context = target_contexts[i-1].unsqueeze(0)

                # Get normalized token embedding
                token_embed = get_normalized_embedding(self.model, token_id)

                # Process through CVFP blocks
                context_out, token_out = process_through_blocks(
                    self.model, input_context, token_embed, freeze_context=False
                )

                # Use fixed context for prediction
                fixed_context = target_contexts[i].unsqueeze(0)

                # Predict next token from concatenated fixed_context + token_out
                combined = torch.cat([fixed_context, token_out], dim=-1)
                logits = self.model.token_output(combined)
                all_logits.append(logits)

            # Stack all logits
            all_logits = torch.cat(all_logits, dim=0)  # [seq_len - 1, vocab_size]

            # Compute loss
            loss = self.criterion(all_logits, target_ids)

            # Compute accuracy
            predictions = torch.argmax(all_logits, dim=-1)
            correct = (predictions == target_ids).sum().item()

        avg_loss = loss.item()
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        accuracy = correct / len(target_ids)

        return avg_loss, perplexity, accuracy

    def train_full(self, train_token_ids, val_token_ids, device, epochs=10, patience=3):
        """
        Full training loop with validation and early stopping

        Args:
            train_token_ids: Training token IDs [seq_len]
            val_token_ids: Validation token IDs [seq_len]
            device: Device to use
            epochs: Number of training epochs
            patience: Early stopping patience (stop if val_loss doesn't improve for this many epochs)

        Returns:
            history: Dictionary with training history
        """
        print(f"\n{'='*70}")
        print("PHASE 2: Next-Token Prediction Training")
        print("         (Context-Fixed Learning)")
        print(f"{'='*70}\n")

        print(f"Training tokens: {len(train_token_ids):,}")
        print(f"Validation tokens: {len(val_token_ids):,}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"CVFP layers frozen: {self.freeze_context}")
        print(f"Early stopping patience: {patience}")
        print(f"✓ Stage 1: Initialize fixed contexts C* from training data")
        print(f"✓ Stage 2: Train with context_out = C*[i] (complete fixing)")
        print(f"✓ Prediction from concatenated C*[i] + token_out")
        print(f"✓ Gradients flow through token_out only\n")

        # Stage 1: Initialize target contexts (Phase 2開始時の固定文脈ベクトル)
        print(f"Stage 1: Initializing fixed contexts C*...")
        self.initialize_target_contexts(train_token_ids, device, is_training=True)
        self.initialize_target_contexts(val_token_ids, device, is_training=False)
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

        print(f"Stage 2: Training with fixed contexts...")
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_ppl = self.train_epoch(
                train_token_ids, device
            )

            # Validate
            val_loss, val_ppl, val_acc = self.evaluate(
                val_token_ids, device
            )

            # Record history
            history['train_loss'].append(train_loss)
            history['train_ppl'].append(train_ppl)
            history['val_loss'].append(val_loss)
            history['val_ppl'].append(val_ppl)
            history['val_acc'].append(val_acc)

            # Print progress
            print(f"Epoch {epoch}/{epochs}:")
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
