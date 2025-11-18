"""FP16 Mixed Precision Trainer

This module provides a trainer that supports PyTorch Automatic Mixed Precision (AMP)
for faster training on GPU with FP16 precision.
"""

import torch
import torch.amp
import time
from .trainer import Trainer
from ..evaluation.metrics import compute_loss, compute_perplexity


class FP16Trainer(Trainer):
    """FP16混合精度対応のTrainer

    PyTorch AMPを使ったFP16訓練
    GPU必須。
    """

    def __init__(self, *args, use_amp=True, **kwargs):
        """Initialize FP16Trainer

        Args:
            *args: Arguments passed to parent Trainer
            use_amp: Whether to use automatic mixed precision (default: True)
            **kwargs: Keyword arguments passed to parent Trainer
        """
        super().__init__(*args, **kwargs)

        # GPU必須チェック
        if not torch.cuda.is_available():
            raise RuntimeError("GPU not available! FP16 training requires CUDA GPU.")

        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler('cuda')
        print(f"\n✓ FP16 Mixed Precision enabled (AMP)")

    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch with FP16 mixed precision

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

            # FP16 mixed precision forward/backward
            with torch.amp.autocast('cuda'):
                outputs = self.model(inputs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                loss = compute_loss(logits, targets, pad_idx=0)

            # Scaled backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first)
            self.scaler.unscale_(self.optimizer)

            # Adaptive gradient clipping based on training stage
            if self.current_epoch < 10:
                clip_value = 0.5  # Early: strict clipping
            elif self.current_epoch < 30:
                clip_value = 1.0  # Middle: standard clipping
            else:
                clip_value = 2.0  # Late: relaxed clipping

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), clip_value
            )

            # Optimizer step with scaling
            self.scaler.step(self.optimizer)
            self.scaler.update()

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
        print(f"| {elapsed/60:.1f}min | Loss: {avg_loss:.4f}")

        return avg_loss, avg_ppl
