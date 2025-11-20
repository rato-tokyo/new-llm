"""Two-Phase Training for New-LLM on UltraChat

Phase 1: Context Vector Learning
- Process each dialogue sample
- First pass: Just forward, store context vectors
- Second pass onwards: Use previous context as teacher signal
- Continue until fixed points converge

Phase 2: Token Prediction Learning
- Freeze context vectors at fixed points
- Train to predict next tokens
- Only train on assistant responses
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import math
import time
from tqdm import tqdm

from src.models.new_llm_flexible import NewLLMFlexible
from src.models.new_llm_sequential import NewLLMSequential
from src.models.new_llm_layerwise import NewLLMLayerwise
from src.utils.dialogue_config import (
    DialogueConfig,
    TinyDialogueConfig,
    Tiny256DialogueConfig,
    SmallDialogueConfig,
    Small2LayerSequentialConfig,
    Small2LayerLayerwiseConfig,
    MediumDialogueConfig,
    LargeDialogueConfig
)
from src.data.ultrachat_loader import UltraChatLoader
import argparse


class ContextCache:
    """Cache for storing fixed-point context vectors"""

    def __init__(self, cache_dir, clear_cache=False):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_dir, "context_cache.pt")

        # Clear cache if requested
        if clear_cache and os.path.exists(self.cache_file):
            os.remove(self.cache_file)
            print("✓ Cleared old context cache")

        # Load existing cache
        if os.path.exists(self.cache_file):
            self.cache = torch.load(self.cache_file)
            print(f"Loaded context cache: {len(self.cache)} samples")
        else:
            self.cache = {}

    def save_context(self, sample_idx, input_ids, contexts, converged, num_iters):
        """Save context for a sample"""
        self.cache[sample_idx] = {
            "input_ids": input_ids.cpu(),
            "contexts": contexts.cpu(),
            "converged": converged.cpu(),
            "num_iters": num_iters.cpu()
        }

    def load_context(self, sample_idx):
        """Load context for a sample"""
        if sample_idx in self.cache:
            return self.cache[sample_idx]
        return None

    def save_to_disk(self):
        """Save cache to disk"""
        torch.save(self.cache, self.cache_file)
        print(f"Saved context cache: {len(self.cache)} samples")

    def get_convergence_stats(self):
        """Get convergence statistics"""
        if len(self.cache) == 0:
            return {}

        total_tokens = 0
        converged_tokens = 0
        total_iters = 0

        for data in self.cache.values():
            converged = data["converged"]
            num_iters = data["num_iters"]

            total_tokens += converged.numel()
            converged_tokens += converged.sum().item()
            total_iters += num_iters.float().mean().item()

        return {
            "total_samples": len(self.cache),
            "total_tokens": total_tokens,
            "converged_tokens": converged_tokens,
            "convergence_rate": converged_tokens / total_tokens if total_tokens > 0 else 0,
            "avg_iterations": total_iters / len(self.cache)
        }


class TwoPhaseTrainer:
    """Two-phase trainer for dialogue"""

    def __init__(self, config, clear_cache=False):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Initialize data loader FIRST (updates config.vocab_size)
        self.data_loader = UltraChatLoader(config)

        # Initialize model AFTER data loader (uses updated vocab_size)
        # Select model based on architecture type
        architecture = getattr(config, 'architecture', 'flexible')
        if architecture == 'sequential':
            self.model = NewLLMSequential(config).to(self.device)
        elif architecture == 'layerwise':
            self.model = NewLLMLayerwise(config).to(self.device)
        else:
            self.model = NewLLMFlexible(config).to(self.device)

        print(f"Model initialized: {self.model.count_parameters():,} parameters")
        print(f"Architecture: {architecture}, {config.num_layers} layers, context_dim={config.context_dim}")

        # Initialize cache (with optional clearing)
        self.context_cache = ContextCache(
            os.path.join(config.cache_dir, "contexts"),
            clear_cache=clear_cache
        )

        # Optimizer (for Phase 1 and Phase 2)
        # Phase 1 uses phase1_learning_rate, Phase 2 will reinitialize with phase2_learning_rate
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.phase1_learning_rate
        )

    def phase1_train_sample(self, sample_idx, dataset, epoch):
        """
        Phase 1: Train model to reach fixed-point contexts for a single sample

        Args:
            sample_idx: Sample index
            dataset: UltraChat dataset
            epoch: Current epoch number

        Returns:
            avg_loss: Average fixed-point loss across tokens
            converged_ratio: Ratio of converged tokens
        """
        # Load sample
        input_ids, assistant_mask, messages = self.data_loader.get_sample(dataset, sample_idx)
        input_ids = input_ids.unsqueeze(0).to(self.device)  # [1, seq_len]
        batch_size, seq_len = input_ids.shape

        if epoch == 0:
            print(f"\nSample {sample_idx}: {seq_len} tokens, {len(messages)} messages")

        # Get token embeddings
        self.model.train()
        token_embeds = self.model.token_embedding(input_ids)
        if hasattr(self.model, 'embed_norm'):
            token_embeds = self.model.embed_norm(token_embeds)

        # Track metrics
        total_loss = 0.0
        converged_count = 0

        # Train each token to reach fixed point
        for t in range(seq_len):
            current_token = token_embeds[:, t, :]  # [batch, embed_dim]

            # Initialize context
            context = torch.zeros(batch_size, self.config.context_dim, device=self.device)

            # Fixed-point iteration with training
            for iteration in range(self.config.phase1_max_iterations):
                self.optimizer.zero_grad()

                # Forward pass - update context
                # Detach current_token to avoid gradient accumulation
                current_token_detached = current_token.detach()

                if hasattr(self.model, 'fnn_layers'):
                    # Layer-wise architecture
                    context_temp = context.detach()  # Start fresh each iteration
                    for layer_idx in range(self.model.num_layers):
                        fnn_input = torch.cat([current_token_detached, context_temp], dim=-1)
                        hidden = self.model.fnn_layers[layer_idx](fnn_input)

                        context_delta = torch.tanh(self.model.context_delta_projs[layer_idx](hidden))
                        forget = torch.sigmoid(self.model.forget_gates[layer_idx](hidden))
                        input_g = torch.sigmoid(self.model.input_gates[layer_idx](hidden))

                        context_temp = forget * context_temp + input_g * context_delta
                        context_temp = self.model.context_norms[layer_idx](context_temp)

                    context_new = context_temp
                else:
                    # Sequential architecture
                    context_detached = context.detach()  # Start fresh each iteration
                    fnn_input = torch.cat([current_token_detached, context_detached], dim=-1)
                    hidden = self.model.fnn(fnn_input)

                    context_delta = torch.tanh(self.model.context_delta_proj(hidden))
                    forget = torch.sigmoid(self.model.forget_gate(hidden))
                    input_g = torch.sigmoid(self.model.input_gate(hidden))

                    context_new = forget * context_detached + input_g * context_delta
                    context_new = self.model.context_norm(context_new)

                # Fixed-point loss
                loss = nn.functional.mse_loss(context_new, context.detach())

                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                # Update context for next iteration
                context = context_new.detach()

                # Check convergence (after warmup)
                if iteration >= self.config.phase1_warmup_iterations:
                    if loss.item() < self.config.phase1_convergence_threshold:
                        converged_count += 1
                        break

            total_loss += loss.item()

            # Progress logging
            if (t + 1) % 50 == 0 or t == seq_len - 1:
                avg_loss_so_far = total_loss / (t + 1)
                converged_ratio = converged_count / (t + 1)
                if epoch == 0:
                    print(f"  Token {t+1}/{seq_len} | Loss: {avg_loss_so_far:.6f} | Converged: {converged_ratio:.1%}", end='\r')

        avg_loss = total_loss / seq_len
        converged_ratio = converged_count / seq_len

        if epoch == 0:
            print(f"  Token {seq_len}/{seq_len} | Loss: {avg_loss:.6f} | Converged: {converged_ratio:.1%}")

        return avg_loss, converged_ratio

    def phase1_train(self, max_samples):
        """
        Phase 1: Train model to reach fixed-point contexts

        Args:
            max_samples: Maximum number of samples to process

        Returns:
            success: Whether training achieved sufficient convergence
        """
        print("\n" + "=" * 60)
        print("PHASE 1: Context Vector Fixed-Point Training")
        print("=" * 60)

        # Load dataset
        dataset = self.data_loader.load_dataset(max_samples=max_samples)

        start_time = time.time()

        # Training epochs
        num_epochs = getattr(self.config, 'phase1_epochs', 3)

        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"{'='*60}")

            epoch_loss = 0.0
            epoch_convergence = 0.0

            for idx in range(len(dataset)):
                avg_loss, converged_ratio = self.phase1_train_sample(idx, dataset, epoch)

                epoch_loss += avg_loss
                epoch_convergence += converged_ratio

            # Epoch summary
            avg_epoch_loss = epoch_loss / len(dataset)
            avg_epoch_convergence = epoch_convergence / len(dataset)

            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Average Loss: {avg_epoch_loss:.6f}")
            print(f"  Average Convergence: {avg_epoch_convergence:.1%}")

            # Save checkpoint
            checkpoint_path = os.path.join(self.config.cache_dir, "checkpoints", f"phase1_epoch{epoch+1}.pt")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": avg_epoch_loss,
                "convergence": avg_epoch_convergence
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

        total_time = time.time() - start_time

        # Final evaluation: cache fixed-point contexts
        print(f"\n{'='*60}")
        print("Caching Fixed-Point Contexts (Evaluation Mode)")
        print(f"{'='*60}")

        self.model.eval()
        with torch.no_grad():
            for idx in range(len(dataset)):
                input_ids, assistant_mask, messages = self.data_loader.get_sample(dataset, idx)
                input_ids = input_ids.unsqueeze(0).to(self.device)

                # Get fixed-point contexts (without training)
                fixed_contexts, converged, num_iters = self.model.get_fixed_point_context(
                    input_ids,
                    max_iterations=self.config.phase1_max_iterations,
                    tolerance=self.config.phase1_convergence_threshold,
                    warmup_iterations=self.config.phase1_warmup_iterations
                )

                # Save to cache
                self.context_cache.save_context(
                    idx,
                    input_ids.squeeze(0),
                    fixed_contexts.squeeze(0),
                    converged.squeeze(0),
                    num_iters.squeeze(0)
                )

                if (idx + 1) % 10 == 0:
                    print(f"  Cached {idx+1}/{len(dataset)} samples", end='\r')

        print(f"  Cached {len(dataset)}/{len(dataset)} samples")
        self.context_cache.save_to_disk()

        # Show statistics
        stats = self.context_cache.get_convergence_stats()
        print("\n" + "=" * 60)
        print("Phase 1 Training Results:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Convergence rate: {stats['convergence_rate']:.1%}")
        print(f"  Avg iterations: {stats['avg_iterations']:.1f}")
        print(f"  Total time: {total_time/60:.1f} min ({total_time:.1f}s)")
        print(f"  Avg time per sample: {total_time/max_samples:.1f}s")
        print("=" * 60)

        # Check if we can proceed to Phase 2
        if stats['convergence_rate'] < self.config.phase1_min_converged_ratio:
            print(f"\n⚠️  Convergence rate too low: {stats['convergence_rate']:.1%}")
            print(f"   Required: {self.config.phase1_min_converged_ratio:.1%}")
            print("   Consider:")
            print("   - Training for more epochs")
            print("   - Adjusting learning rate")
            print("   - Increasing num_layers")
            return False

        print("\n✓ Phase 1 training completed successfully!")
        return True

    def phase2_train_sample(self, sample_idx, dataset):
        """
        Phase 2: Train token prediction for a single sample

        Args:
            sample_idx: Sample index
            dataset: UltraChat dataset
        """
        # Load cached context
        cached = self.context_cache.load_context(sample_idx)
        if cached is None:
            print(f"Sample {sample_idx}: No cached context, skipping")
            return

        # Load sample
        input_ids, assistant_mask, messages = self.data_loader.get_sample(dataset, sample_idx)
        input_ids = input_ids.unsqueeze(0).to(self.device)  # [1, seq_len]
        assistant_mask = assistant_mask.unsqueeze(0).to(self.device)  # [1, seq_len]

        # Get fixed contexts
        fixed_contexts = cached["contexts"].unsqueeze(0).to(self.device)  # [1, seq_len, context_dim]

        print(f"\nSample {sample_idx}: Training token prediction")

        # Train for multiple epochs
        for epoch in range(self.config.phase2_epochs):
            self.model.train()

            # Forward pass with fixed contexts
            # We need to modify forward to accept pre-computed contexts
            # For now, let's just do regular forward and compute loss
            logits = self.model(input_ids)  # [1, seq_len, vocab_size]

            # Compute loss only on assistant tokens
            # Shift for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()  # [1, seq_len-1, vocab_size]
            shift_labels = input_ids[:, 1:].contiguous()  # [1, seq_len-1]
            shift_mask = assistant_mask[:, 1:].contiguous()  # [1, seq_len-1]

            # Compute loss
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss_per_token = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.shape)  # [1, seq_len-1]

            # Apply mask (only assistant tokens)
            masked_loss = loss_per_token * shift_mask.float()
            loss = masked_loss.sum() / (shift_mask.sum() + 1e-8)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.phase2_gradient_clip
            )
            self.optimizer.step()

            # Compute metrics
            with torch.no_grad():
                # Accuracy
                preds = shift_logits.argmax(dim=-1)
                correct = (preds == shift_labels) & shift_mask
                accuracy = correct.sum().float() / (shift_mask.sum() + 1e-8)

                # Perplexity
                ppl = torch.exp(loss)

            # Log every step
            if (epoch + 1) % self.config.log_every_steps == 0:
                print(f"  Epoch {epoch+1}/{self.config.phase2_epochs}: "
                      f"Loss={loss.item():.4f}, PPL={ppl.item():.2f}, Acc={accuracy.item():.1%}")

    def phase2_train(self, max_samples):
        """
        Phase 2: Train token prediction for multiple samples

        Args:
            max_samples: Maximum number of samples to process
        """
        print("\n" + "=" * 60)
        print("PHASE 2: Token Prediction Learning")
        print("=" * 60)

        # Load dataset
        dataset = self.data_loader.load_dataset(max_samples=max_samples)

        # Train each sample
        for idx in range(len(dataset)):
            self.phase2_train_sample(idx, dataset)

            # Save checkpoint periodically
            if (idx + 1) % self.config.save_every_samples == 0:
                self.save_checkpoint(f"phase2_sample_{idx+1}.pt")

        # Final checkpoint
        self.save_checkpoint("phase2_final.pt")

        print("\n✓ Phase 2 completed!")

    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config.cache_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__
        }, checkpoint_path)

        print(f"Checkpoint saved: {checkpoint_path}")


def main():
    """Main training function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="New-LLM Two-Phase Dialogue Training")
    parser.add_argument("--config", type=str, default="tiny",
                        choices=["tiny", "tiny256", "small", "seq2", "layer2", "medium", "large"],
                        help="Config: tiny=1L/128D, tiny256=1L/256D, small=2L/512D, seq2=2L/256D-Sequential, layer2=2L/256D-Layerwise, medium=3L/512D, large=4L/512D")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to use (cpu or cuda)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Clear old context cache before training")
    args = parser.parse_args()

    # Select config
    config_map = {
        "tiny": TinyDialogueConfig(),
        "tiny256": Tiny256DialogueConfig(),
        "small": SmallDialogueConfig(),
        "seq2": Small2LayerSequentialConfig(),
        "layer2": Small2LayerLayerwiseConfig(),
        "medium": MediumDialogueConfig(),
        "large": LargeDialogueConfig()
    }
    config = config_map[args.config]
    config.device = args.device

    print("=" * 60)
    print("New-LLM Two-Phase Dialogue Training")
    print("=" * 60)
    print(f"Config: {args.config.upper()}")
    print(f"Model: {config.num_layers} layers, context_dim={config.context_dim}, hidden_dim={config.hidden_dim}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Device: {config.device}")
    print(f"Phase 1: {config.phase1_max_samples} samples, max_iter={config.phase1_max_iterations}")
    print("=" * 60)

    # Initialize trainer (with optional cache clearing)
    trainer = TwoPhaseTrainer(config, clear_cache=args.clear_cache)

    # Phase 1: Context learning
    phase1_success = trainer.phase1_train(max_samples=config.phase1_max_samples)

    if not phase1_success:
        print("\n❌ Phase 1 failed. Experiment terminated.")
        print("   Please adjust architecture (num_layers, context_dim, hidden_dim)")
        return

    # Phase 2: Token prediction
    trainer.phase2_train(max_samples=config.phase1_max_samples)

    print("\n" + "=" * 60)
    print("✓ Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
