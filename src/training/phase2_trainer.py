"""
Phase 2 Trainer: Next-Token Prediction with Context Propagation

Trains the prediction head to predict next tokens using contexts learned in Phase 1.

Key Design (Updated 2025-11-24):
- Context propagation: Context carries forward between tokens (like Phase 1)
- First token starts from zero-vector, subsequent tokens use previous context
- Token embed used for prediction (not context)
- Context gradient detached to prevent backprop through context history
- This ensures Phase 1 and Phase 2 consistency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Phase2Trainer:
    """
    Phase 2 Trainer for next-token prediction

    Args:
        model: NewLLMResidual model with prediction head
        learning_rate: Learning rate for optimizer
        freeze_context: If True, freeze CVFP layers (only train prediction head)
        gradient_clip: Gradient clipping value (None = no clipping)
    """

    def __init__(
        self,
        model,
        learning_rate=0.0001,
        freeze_context=True,
        gradient_clip=1.0
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.freeze_context = freeze_context
        self.gradient_clip = gradient_clip

        # Freeze CVFP layers if requested
        if freeze_context:
            self._freeze_cvfp_layers()
        else:
            # All parameters are trainable (full fine-tuning)
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✓ Full model fine-tuning: {trainable_params:,}/{total_params:,} parameters trainable")

        # Optimizer (only trainable parameters)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

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

    def train_epoch(self, token_ids, device):
        """
        Train one epoch on token sequence with context propagation

        CRITICAL DESIGN:
        - Context propagates forward (like Phase 1)
        - Token embed used for prediction (not context)
        - Context gradient detached to prevent backprop through history

        Args:
            token_ids: Token IDs [seq_len]
            device: Device to use

        Returns:
            avg_loss: Average loss for this epoch
            perplexity: Perplexity metric
        """
        print(f"  [DEBUG] train_epoch called with {len(token_ids)} tokens")
        self.model.train()

        # Input: all tokens except last
        # Target: all tokens except first
        input_ids = token_ids[:-1]  # [seq_len - 1]
        target_ids = token_ids[1:]  # [seq_len - 1]
        print(f"  [DEBUG] Processing {len(input_ids)} input tokens")

        # Initialize context (first token starts from zero)
        context = torch.zeros(1, self.model.context_dim, device=device)
        print(f"  [DEBUG] Context initialized: {context.shape}")

        # Process all tokens with context propagation
        all_logits = []
        for i, token_id in enumerate(input_ids):
            if i % 1000 == 0:
                print(f"  [DEBUG] Processing token {i}/{len(input_ids)}")
            # Detach context to prevent gradient flow through history
            context = context.detach()

            # Get token embedding
            token_embed_output = self.model.token_embedding(token_id.unsqueeze(0))
            if isinstance(token_embed_output, tuple):
                token_embed = token_embed_output[0]  # [1, embed_dim]
            else:
                token_embed = token_embed_output  # [1, embed_dim]

            # Process through CVFP blocks
            for block in self.model.blocks:
                if self.freeze_context:
                    with torch.no_grad():
                        context, token_embed = block(context, token_embed)
                else:
                    context, token_embed = block(context, token_embed)

            # Predict next token from token_embed (NOT context)
            logits = self.model.token_output(token_embed)  # [1, vocab_size]
            all_logits.append(logits)

            # Context carries forward to next token

        # Stack all logits
        print(f"  [DEBUG] Stacking {len(all_logits)} logits")
        all_logits = torch.cat(all_logits, dim=0)  # [seq_len - 1, vocab_size]
        print(f"  [DEBUG] Logits shape: {all_logits.shape}")

        # Compute loss
        print(f"  [DEBUG] Computing loss...")
        self.optimizer.zero_grad()
        loss = self.criterion(all_logits, target_ids)
        print(f"  [DEBUG] Loss: {loss.item():.4f}")

        # Backward pass
        print(f"  [DEBUG] Running backward pass...")
        loss.backward()
        print(f"  [DEBUG] Backward complete")

        # Gradient clipping
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                self.gradient_clip
            )

        self.optimizer.step()

        # Calculate metrics
        avg_loss = loss.item()
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return avg_loss, perplexity

    def evaluate(self, token_ids, device):
        """
        Evaluate on token sequence with context propagation

        Args:
            token_ids: Token IDs [seq_len]
            device: Device to use

        Returns:
            avg_loss: Average loss
            perplexity: Perplexity metric
            accuracy: Token prediction accuracy
        """
        self.model.eval()

        # Input: all tokens except last
        # Target: all tokens except first
        input_ids = token_ids[:-1]  # [seq_len - 1]
        target_ids = token_ids[1:]  # [seq_len - 1]

        with torch.no_grad():
            # Initialize context (first token starts from zero)
            context = torch.zeros(1, self.model.context_dim, device=device)

            # Process all tokens with context propagation
            all_logits = []
            for token_id in input_ids:
                # Get token embedding
                token_embed_output = self.model.token_embedding(token_id.unsqueeze(0))
                if isinstance(token_embed_output, tuple):
                    token_embed = token_embed_output[0]  # [1, embed_dim]
                else:
                    token_embed = token_embed_output  # [1, embed_dim]

                # Process through CVFP blocks
                for block in self.model.blocks:
                    context, token_embed = block(context, token_embed)

                # Predict next token from token_embed
                logits = self.model.token_output(token_embed)  # [1, vocab_size]
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

    def train_full(self, train_token_ids, val_token_ids, device, epochs=10):
        """
        Full training loop with validation

        Args:
            train_token_ids: Training token IDs [seq_len]
            val_token_ids: Validation token IDs [seq_len]
            device: Device to use
            epochs: Number of training epochs

        Returns:
            history: Dictionary with training history
        """
        print(f"\n{'='*70}")
        print("PHASE 2: Next-Token Prediction Training")
        print("         (Context Propagation + Token Embed Prediction)")
        print(f"{'='*70}\n")

        print(f"Training tokens: {len(train_token_ids):,}")
        print(f"Validation tokens: {len(val_token_ids):,}")
        print(f"Epochs: {epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Context frozen: {self.freeze_context}")
        print(f"✓ Context propagates forward (like Phase 1)")
        print(f"✓ Prediction from token_embed (not context)")
        print(f"✓ Context gradient detached (no backprop through history)\n")

        history = {
            'train_loss': [],
            'train_ppl': [],
            'val_loss': [],
            'val_ppl': [],
            'val_acc': []
        }

        best_val_loss = float('inf')

        for epoch in range(1, epochs + 1):
            print(f"\n[DEBUG] Starting Epoch {epoch}/{epochs}")
            # Train
            print(f"[DEBUG] Calling train_epoch...")
            train_loss, train_ppl = self.train_epoch(
                train_token_ids, device
            )
            print(f"[DEBUG] train_epoch complete: loss={train_loss:.4f}")

            # Validate
            print(f"[DEBUG] Calling evaluate...")
            val_loss, val_ppl, val_acc = self.evaluate(
                val_token_ids, device
            )
            print(f"[DEBUG] evaluate complete: loss={val_loss:.4f}")

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

            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  ✓ New best validation loss: {best_val_loss:.4f}")

            print()

        print(f"{'='*70}")
        print("Phase 2 Training Complete")
        print(f"{'='*70}\n")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final validation perplexity: {history['val_ppl'][-1]:.2f}")
        print(f"Final validation accuracy: {history['val_acc'][-1]*100:.2f}%\n")

        return history
