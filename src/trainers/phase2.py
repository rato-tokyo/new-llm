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

    def train_epoch(self, token_ids, device, batch_size=512):
        """
        Train one epoch on token sequence with context-fixed learning (BATCH PARALLEL version)

        CRITICAL DESIGN (Updated 2025-11-26):
        - Input: [C*[i-1], token_embed[i]] - fixed context from initialization
        - Output: [context_out, token_out] from CVFP blocks
        - context_out is replaced with C*[i].detach() (complete fixing)
        - Prediction from concatenated C*[i] + token_out
        - Gradients flow through token_out only (CVFP params still updated via token_out)

        BATCH PARALLEL processing (Updated 2025-11-26):
        - Process all tokens in batch simultaneously (NOT sequential)
        - Each token's processing is independent (C* is fixed)
        - 5-10x speedup compared to sequential version

        Args:
            token_ids: Token IDs [seq_len]
            device: Device to use
            batch_size: Number of tokens per mini-batch (default: 512)

        Returns:
            avg_loss: Average loss for this epoch
            perplexity: Perplexity metric
        """
        self.model.train()

        # Input: all tokens except last
        # Target: all tokens except first
        input_ids = token_ids[:-1]  # [seq_len - 1]
        target_ids = token_ids[1:]  # [seq_len - 1]

        num_tokens = len(input_ids)
        total_loss = 0.0
        num_batches = 0

        # Process tokens in mini-batches (PARALLEL within each batch)
        for batch_start in range(0, num_tokens, batch_size):
            batch_end = min(batch_start + batch_size, num_tokens)
            batch_len = batch_end - batch_start

            # Get batch input IDs
            batch_input_ids = input_ids[batch_start:batch_end]  # [batch_len]

            # Build input contexts: C*[i-1] for each token in batch
            # First token uses zero vector, rest use C*[batch_start-1:batch_end-1]
            if batch_start == 0:
                # First batch: token 0 uses zero, tokens 1+ use C*[0:batch_end-1]
                zero_context = torch.zeros(1, self.model.context_dim, device=device)
                if batch_len > 1:
                    rest_contexts = self.target_contexts_train[:batch_end-1].detach()
                    batch_input_contexts = torch.cat([zero_context, rest_contexts], dim=0)
                else:
                    batch_input_contexts = zero_context
            else:
                # Other batches: use C*[batch_start-1:batch_end-1]
                batch_input_contexts = self.target_contexts_train[batch_start-1:batch_end-1].detach()

            # Get batch token embeddings (PARALLEL)
            batch_token_embeds = self.model.token_embedding(batch_input_ids)
            if isinstance(batch_token_embeds, tuple):
                batch_token_embeds = batch_token_embeds[0]
            batch_token_embeds = self.model.embed_norm(batch_token_embeds)  # [batch_len, embed_dim]

            # Process through CVFP blocks (PARALLEL - all tokens at once)
            current_contexts = batch_input_contexts
            current_tokens = batch_token_embeds
            for block in self.model.blocks:
                if self.freeze_context:
                    with torch.no_grad():
                        current_contexts, current_tokens = block(current_contexts, current_tokens)
                else:
                    current_contexts, current_tokens = block(current_contexts, current_tokens)

            # token_out is the output token embeddings after CVFP blocks
            batch_token_out = current_tokens  # [batch_len, embed_dim]

            # Get fixed contexts C*[i] for prediction (PARALLEL)
            batch_fixed_contexts = self.target_contexts_train[batch_start:batch_end].detach()

            # Predict next token from concatenated fixed_context + token_out (PARALLEL)
            combined = torch.cat([batch_fixed_contexts, batch_token_out], dim=-1)  # [batch_len, context_dim + embed_dim]
            batch_logits = self.model.token_output(combined)  # [batch_len, vocab_size]

            batch_targets = target_ids[batch_start:batch_end]

            # Compute loss and update
            self.optimizer.zero_grad()
            batch_loss = self.criterion(batch_logits, batch_targets)
            batch_loss.backward()

            # Gradient clipping
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, self.model.parameters()),
                    self.gradient_clip
                )

            self.optimizer.step()

            # Accumulate loss
            total_loss += batch_loss.item() * batch_len
            num_batches += 1

        # Calculate metrics (weighted average)
        avg_loss = total_loss / num_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return avg_loss, perplexity

    def evaluate(self, token_ids, device, target_contexts=None):
        """
        Evaluate on token sequence with context-fixed learning (BATCH PARALLEL version)

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
        num_tokens = len(input_ids)

        with torch.no_grad():
            # Build input contexts: C*[i-1] for each token (PARALLEL)
            # Token 0 uses zero vector, tokens 1+ use C*[0:num_tokens-1]
            zero_context = torch.zeros(1, self.model.context_dim, device=device)
            if num_tokens > 1:
                rest_contexts = target_contexts[:num_tokens-1]
                all_input_contexts = torch.cat([zero_context, rest_contexts], dim=0)
            else:
                all_input_contexts = zero_context

            # Get all token embeddings (PARALLEL)
            all_token_embeds = self.model.token_embedding(input_ids)
            if isinstance(all_token_embeds, tuple):
                all_token_embeds = all_token_embeds[0]
            all_token_embeds = self.model.embed_norm(all_token_embeds)  # [num_tokens, embed_dim]

            # Process through CVFP blocks (PARALLEL - all tokens at once)
            current_contexts = all_input_contexts
            current_tokens = all_token_embeds
            for block in self.model.blocks:
                current_contexts, current_tokens = block(current_contexts, current_tokens)

            # token_out is the output token embeddings after CVFP blocks
            all_token_out = current_tokens  # [num_tokens, embed_dim]

            # Get fixed contexts C*[i] for prediction (PARALLEL)
            all_fixed_contexts = target_contexts[:num_tokens]

            # Predict next token from concatenated fixed_context + token_out (PARALLEL)
            combined = torch.cat([all_fixed_contexts, all_token_out], dim=-1)  # [num_tokens, context_dim + embed_dim]
            all_logits = self.model.token_output(combined)  # [num_tokens, vocab_size]

            # Compute loss
            loss = self.criterion(all_logits, target_ids)

            # Compute accuracy
            predictions = torch.argmax(all_logits, dim=-1)
            correct = (predictions == target_ids).sum().item()

        avg_loss = loss.item()
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        accuracy = correct / len(target_ids)

        return avg_loss, perplexity, accuracy

    def train_full(self, train_token_ids, val_token_ids, device, epochs=10, patience=3, batch_size=512):
        """
        Full training loop with validation and early stopping

        Args:
            train_token_ids: Training token IDs [seq_len]
            val_token_ids: Validation token IDs [seq_len]
            device: Device to use
            epochs: Number of training epochs
            patience: Early stopping patience (stop if val_loss doesn't improve for this many epochs)
            batch_size: Mini-batch size for training (GPU memory optimization)

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
        print(f"Batch size: {batch_size}")
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
                train_token_ids, device, batch_size=batch_size
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
