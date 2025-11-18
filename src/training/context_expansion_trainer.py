"""Context Expansion Trainer with Gradient Freezing Support

Supports two training modes for context vector expansion:
1. Freeze Base: Only train new dimensions (existing dimensions frozen)
2. Fine-tune All: Train all dimensions (standard fine-tuning)
"""

import torch
import time
from .fp16_trainer import FP16Trainer
from ..evaluation.metrics import compute_loss, compute_perplexity


class ContextExpansionTrainer(FP16Trainer):
    """Trainer for context vector expansion with optional base dimension freezing

    Supports two training strategies:
    - freeze_base_dims=True: Only new dimensions are trained (Transfer Learning)
    - freeze_base_dims=False: All dimensions are trained (Fine-tuning)

    Args:
        base_context_dim: Original context vector dimension (e.g., 256)
        freeze_base_dims: If True, freeze gradients for base dimensions
    """

    def __init__(self, *args, base_context_dim=256, freeze_base_dims=False, use_amp=True, **kwargs):
        # For testing on CPU, allow use_amp=False to skip GPU check
        if not use_amp and 'model' in kwargs:
            # Skip FP16Trainer's __init__ and go directly to base Trainer
            from .trainer import Trainer
            Trainer.__init__(self, *args, **kwargs)
            self.use_amp = False
            self.scaler = None
        else:
            super().__init__(*args, use_amp=use_amp, **kwargs)

        self.base_context_dim = base_context_dim
        self.freeze_base_dims = freeze_base_dims

        if freeze_base_dims:
            print(f"\n🔒 Freeze Base Mode: ENABLED")
            print(f"   Base dimensions (0:{base_context_dim}) will be frozen")
            print(f"   Only new dimensions ({base_context_dim}+) will be trained")
        else:
            print(f"\n🔓 Fine-tune All Mode: ENABLED")
            print(f"   All dimensions will be trained (standard fine-tuning)")

    def freeze_base_gradients(self):
        """Zero out gradients for base dimensions after backward pass

        This effectively freezes the base dimensions while allowing
        new dimensions to be trained.
        """
        if not self.freeze_base_dims:
            return

        base_dim = self.base_context_dim

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            # Freeze context_proj weights and biases
            if 'context_proj.weight' in name:
                # Shape: [new_context_dim, embed_dim + new_context_dim]
                # Freeze first base_dim rows
                param.grad[:base_dim, :] = 0

            elif 'context_proj.bias' in name:
                # Shape: [new_context_dim]
                # Freeze first base_dim elements
                param.grad[:base_dim] = 0

            # Freeze FNN input layer (first layer only)
            elif 'fnn_layers.0.weight' in name:
                # Shape: [hidden_dim, embed_dim + new_context_dim]
                # Freeze the context input part (last new_context_dim columns)
                # But only the base_dim part of those columns
                # Get context_dim from config
                context_dim = self.config.context_vector_dim
                embed_dim = param.grad.size(1) - context_dim
                param.grad[:, embed_dim:embed_dim+base_dim] = 0

            # Freeze context_update weights and biases
            elif 'context_update.weight' in name:
                # Shape: [new_context_dim, hidden_dim]
                # Freeze first base_dim rows
                param.grad[:base_dim, :] = 0

            elif 'context_update.bias' in name:
                # Shape: [new_context_dim]
                # Freeze first base_dim elements
                param.grad[:base_dim] = 0

            # Freeze forget_gate and input_gate
            elif 'forget_gate.weight' in name or 'input_gate.weight' in name:
                # Shape: [new_context_dim, hidden_dim]
                # Freeze first base_dim rows
                param.grad[:base_dim, :] = 0

            elif 'forget_gate.bias' in name or 'input_gate.bias' in name:
                # Shape: [new_context_dim]
                # Freeze first base_dim elements
                param.grad[:base_dim] = 0

            # Freeze context_norm parameters
            elif 'context_norm.weight' in name or 'context_norm.bias' in name:
                # Shape: [new_context_dim]
                # Freeze first base_dim elements
                param.grad[:base_dim] = 0

    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch with optional base dimension freezing

        Returns:
            tuple[float, float]: (average_loss, average_perplexity)
        """
        self.model.train()
        total_loss = 0.0
        total_tokens = 0
        num_batches = len(self.train_dataloader)

        print(f"  Training... ", end='', flush=True)
        start_time = time.time()

        for batch_idx, (inputs, targets) in enumerate(self.train_dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # FP16 mixed precision forward/backward (or FP32 for testing)
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(inputs)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs

                    loss = compute_loss(logits, targets, pad_idx=0)

                # Scaled backward pass
                self.scaler.scale(loss).backward()

                # Unscale gradients
                self.scaler.unscale_(self.optimizer)
            else:
                # FP32 mode (for testing on CPU)
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                loss = compute_loss(logits, targets, pad_idx=0)
                loss.backward()

            # *** CRITICAL: Freeze base dimensions if enabled ***
            self.freeze_base_gradients()

            # Adaptive gradient clipping
            if self.current_epoch < 10:
                clip_value = 0.5
            elif self.current_epoch < 30:
                clip_value = 1.0
            else:
                clip_value = 2.0

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), clip_value
            )

            # Optimizer step with scaling
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * inputs.size(0)
            total_tokens += inputs.size(0)

            # Compact progress display (every 20%)
            if (batch_idx + 1) % max(1, num_batches // 5) == 0 or batch_idx == num_batches - 1:
                progress = (batch_idx + 1) / num_batches * 100
                print(f"{progress:.0f}%", end=' ', flush=True)

        elapsed = time.time() - start_time

        avg_loss = total_loss / total_tokens
        avg_ppl = compute_perplexity(avg_loss)

        mode_indicator = "🔒" if self.freeze_base_dims else "🔓"
        print(f"| {elapsed/60:.1f}min | Loss: {avg_loss:.4f} {mode_indicator}")

        return avg_loss, avg_ppl

    def print_trainable_params(self):
        """Print information about trainable parameters

        Useful for verifying freeze mode is working correctly.
        """
        total_params = 0
        frozen_params = 0

        print(f"\n📊 Parameter Analysis:")
        print(f"   Mode: {'Freeze Base' if self.freeze_base_dims else 'Fine-tune All'}")

        if self.freeze_base_dims:
            base_dim = self.base_context_dim
            new_dim = self.config.context_vector_dim

            print(f"   Base dimensions: {base_dim} (frozen)")
            print(f"   New dimensions: {new_dim - base_dim} (trainable)")
            print(f"   Total dimensions: {new_dim}")

            # Estimate frozen parameters
            # This is approximate - actual count would require detailed analysis
            for name, param in self.model.named_parameters():
                total_params += param.numel()

                if 'context' in name or 'gate' in name:
                    # Approximate frozen count
                    if 'weight' in name and len(param.shape) == 2:
                        if param.shape[0] == new_dim:
                            frozen_params += base_dim * param.shape[1]
                    elif 'bias' in name and len(param.shape) == 1:
                        if param.shape[0] == new_dim:
                            frozen_params += base_dim

            trainable_params = total_params - frozen_params
            print(f"\n   Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
            print(f"   Frozen (approx): {frozen_params:,} ({frozen_params/1e6:.2f}M)")
            print(f"   Trainable (approx): {trainable_params:,} ({trainable_params/1e6:.2f}M)")
            print(f"   Trainable ratio: {trainable_params/total_params*100:.1f}%")
        else:
            for param in self.model.parameters():
                total_params += param.numel()

            print(f"   All parameters trainable: {total_params:,} ({total_params/1e6:.2f}M)")
