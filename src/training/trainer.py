"""Training utilities for language models"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
import os
import matplotlib.pyplot as plt

from ..evaluation.metrics import compute_loss, compute_perplexity, compute_accuracy


class Trainer:
    """Generic trainer for language models"""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config,
        model_name: str = "model",
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.model_name = model_name

        self.device = config.device
        self.model.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_ppls = []
        self.val_ppls = []

    def train_epoch(self) -> tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_tokens = 0

        pbar = tqdm(self.train_dataloader, desc=f"Training {self.model_name}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Different models may return different outputs
            outputs = self.model(inputs)

            # Handle different return types
            if isinstance(outputs, tuple):
                # New-LLM returns (logits, context_trajectory)
                logits = outputs[0]
            else:
                # Baseline returns just logits
                logits = outputs

            # Compute loss (only on token predictions)
            loss = compute_loss(logits, targets, pad_idx=0)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )

            self.optimizer.step()

            # Track metrics
            total_loss += loss.item() * inputs.size(0)
            total_tokens += inputs.size(0)

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / total_tokens
        avg_ppl = compute_perplexity(avg_loss)
        return avg_loss, avg_ppl

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float, float]:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        total_batches = 0

        for inputs, targets in self.val_dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)

            # Handle different return types
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Compute metrics
            loss = compute_loss(logits, targets, pad_idx=0)
            accuracy = compute_accuracy(logits, targets, pad_idx=0)

            total_loss += loss.item()
            total_accuracy += accuracy
            total_batches += 1

        avg_loss = total_loss / total_batches
        avg_ppl = compute_perplexity(avg_loss)
        avg_accuracy = total_accuracy / total_batches

        return avg_loss, avg_ppl, avg_accuracy

    def train(self, num_epochs: Optional[int] = None):
        """Full training loop"""
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}")

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss, train_ppl = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_ppls.append(train_ppl)

            # Evaluate
            val_loss, val_ppl, val_acc = self.evaluate()
            self.val_losses.append(val_loss)
            self.val_ppls.append(val_ppl)

            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train PPL: {train_ppl:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(f"best_{self.model_name}.pt")
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")

        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"{'='*60}\n")

        # Save training curves as image
        self.plot_and_save_training_curves()

        return self.train_losses, self.val_losses

    def plot_and_save_training_curves(self):
        """Plot and save training/validation curves as image"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        epochs = range(1, len(self.train_losses) + 1)

        # Loss curves
        axes[0].plot(epochs, self.train_losses, label='Train Loss', linewidth=2, marker='o', markersize=3)
        axes[0].plot(epochs, self.val_losses, label='Val Loss', linewidth=2, marker='s', markersize=3)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'{self.model_name} - Loss Curves', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Perplexity curves
        axes[1].plot(epochs, self.train_ppls, label='Train PPL', linewidth=2, marker='o', markersize=3)
        axes[1].plot(epochs, self.val_ppls, label='Val PPL', linewidth=2, marker='s', markersize=3)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Perplexity', fontsize=12)
        axes[1].set_title(f'{self.model_name} - Perplexity Curves', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')  # Log scale for perplexity

        plt.tight_layout()

        # Save figure
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, f"{self.model_name}_training_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved training curves to {save_path}")
        plt.close()

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config,
        }

        filepath = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        filepath = os.path.join("checkpoints", filename)
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

        print(f"Loaded checkpoint from {filepath}")
