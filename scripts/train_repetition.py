#!/usr/bin/env python3
"""Repetition Training for Context Vector Convergence

This script trains New-LLM to reach stable context vector fixed points
when processing repeated phrases.

Hypothesis: context("red" * 100) ≈ context("red" * 101) when properly trained

Training Strategy:
    Stage 1: Single token repetition ("red" * 100)
    Stage 2: Two token repetition ("red apple" * 100)
    Stage 3: Three token repetition ("red apple tree" * 100)
    ...

Loss Function: MSE(context[t], context[t - cycle_length])
Goal: Minimize change in context vector across repetition cycles
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.new_llm import NewLLM
from src.utils.config import NewLLMConfig
from src.data.repetition_dataset import (
    RepetitionDataset,
    generate_repetition_phrases,
    collate_fn,
    create_staged_datasets
)
from src.training.convergence_loss import (
    ContextConvergenceLoss,
    compute_context_change_rate,
    compute_convergence_metric
)
from transformers import GPT2Tokenizer


def train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    stage: int,
    epochs: int
) -> dict:
    """Train one stage of repetition training

    Args:
        model: New-LLM model
        train_loader: DataLoader for repetition dataset
        loss_fn: ContextConvergenceLoss instance
        optimizer: Optimizer
        device: Device
        stage: Stage number (1, 2, 3, ...)
        epochs: Number of epochs for this stage

    Returns:
        metrics: Dictionary of final metrics
    """
    model.train()

    print(f"\n{'='*80}")
    print(f"🎯 Stage {stage} Training: {loss_fn.cycle_length}-token repetition")
    print(f"{'='*80}")
    print(f"Epochs: {epochs}")
    print(f"Batches: {len(train_loader)}")
    print(f"Convergence weight: {loss_fn.convergence_weight}")
    print(f"Token weight: {loss_fn.token_weight}")
    print(f"{'='*80}\n")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_convergence_loss = 0.0
        epoch_change_rate = 0.0
        epoch_convergence_metric = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Stage {stage} Epoch {epoch+1}/{epochs}",
            leave=True
        )

        for batch_idx, input_ids in enumerate(progress_bar):
            input_ids = input_ids.to(device)

            # Forward pass
            optimizer.zero_grad()
            logits, context_trajectory = model(input_ids)

            # Compute convergence loss
            loss, metrics = loss_fn(
                context_vectors=context_trajectory,
                logits=logits,
                targets=input_ids
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Metrics
            epoch_loss += metrics['loss']
            epoch_convergence_loss += metrics['convergence_loss']

            # Compute additional metrics
            change_rate = compute_context_change_rate(context_trajectory)
            convergence_metric = compute_convergence_metric(
                context_trajectory,
                loss_fn.cycle_length
            )

            epoch_change_rate += change_rate
            epoch_convergence_metric += convergence_metric

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'conv_loss': f"{metrics['convergence_loss']:.4f}",
                'change': f"{change_rate:.4f}",
                'conv_metric': f"{convergence_metric:.4f}"
            })

        # Epoch summary
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches
        avg_convergence_loss = epoch_convergence_loss / num_batches
        avg_change_rate = epoch_change_rate / num_batches
        avg_convergence_metric = epoch_convergence_metric / num_batches

        print(f"\n📊 Stage {stage} Epoch {epoch+1}/{epochs} Summary:")
        print(f"  Loss: {avg_loss:.6f}")
        print(f"  Convergence Loss: {avg_convergence_loss:.6f}")
        print(f"  Context Change Rate: {avg_change_rate:.6f}")
        print(f"  Convergence Metric: {avg_convergence_metric:.6f}")

    final_metrics = {
        'stage': stage,
        'final_loss': avg_loss,
        'final_convergence_loss': avg_convergence_loss,
        'final_change_rate': avg_change_rate,
        'final_convergence_metric': avg_convergence_metric
    }

    return final_metrics


def main():
    parser = argparse.ArgumentParser(description="Repetition training for context convergence")

    # Training parameters
    parser.add_argument('--max-stage', type=int, default=3,
                       help='Maximum stage (1=single token, 2=two tokens, ...)')
    parser.add_argument('--epochs-per-stage', type=int, default=10,
                       help='Number of epochs per stage')
    parser.add_argument('--repetitions', type=int, default=100,
                       help='Number of times to repeat each phrase')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')

    # Model parameters
    parser.add_argument('--context-dim', type=int, default=256,
                       help='Context vector dimension')
    parser.add_argument('--embed-dim', type=int, default=256,
                       help='Token embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='Hidden dimension')
    parser.add_argument('--layers', type=int, default=2,
                       help='Number of FNN layers')

    # Loss parameters
    parser.add_argument('--convergence-weight', type=float, default=1.0,
                       help='Weight for convergence loss')
    parser.add_argument('--token-weight', type=float, default=0.0,
                       help='Weight for token prediction loss (usually 0)')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='checkpoints',
                       help='Output directory')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')

    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== Initialize Tokenizer ==========
    print("\n📚 Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print(f"✓ Tokenizer loaded: {len(tokenizer):,} tokens")

    # ========== Initialize Model ==========
    print("\n🏗️  Building New-LLM model...")
    config = NewLLMConfig()
    config.vocab_size = len(tokenizer)
    config.context_vector_dim = args.context_dim
    config.embed_dim = args.embed_dim
    config.hidden_dim = args.hidden_dim
    config.num_layers = args.layers
    config.max_seq_length = args.max_length

    model = NewLLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params:,} parameters")

    # ========== Optimizer ==========
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ========== Staged Training ==========
    print(f"\n{'='*80}")
    print("🚀 Starting Staged Repetition Training")
    print(f"{'='*80}")
    print(f"Max Stage: {args.max_stage}")
    print(f"Epochs per Stage: {args.epochs_per_stage}")
    print(f"Repetitions per Phrase: {args.repetitions}")
    print(f"{'='*80}\n")

    all_stage_metrics = []

    for stage in range(1, args.max_stage + 1):
        # Generate phrases for this stage
        phrases = generate_repetition_phrases(num_tokens=stage)
        print(f"\n📝 Stage {stage} phrases: {phrases[:3]}...")

        # Create dataset
        dataset = RepetitionDataset(
            phrases=phrases,
            repetitions=args.repetitions,
            tokenizer=tokenizer,
            max_length=args.max_length
        )

        train_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        print(f"✓ Dataset ready: {len(dataset)} sequences, {len(train_loader)} batches")

        # Create loss function for this stage
        loss_fn = ContextConvergenceLoss(
            cycle_length=stage,  # Number of tokens in phrase
            convergence_weight=args.convergence_weight,
            token_weight=args.token_weight
        )

        # Train this stage
        stage_metrics = train_stage(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            stage=stage,
            epochs=args.epochs_per_stage
        )

        all_stage_metrics.append(stage_metrics)

        # Save checkpoint after each stage
        checkpoint_path = os.path.join(
            args.output_dir,
            f"new_llm_repetition_stage{stage}.pt"
        )
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'stage': stage,
            'metrics': stage_metrics
        }, checkpoint_path)
        print(f"✓ Saved checkpoint: {checkpoint_path}")

    # ========== Final Summary ==========
    print(f"\n{'='*80}")
    print("🎉 Staged Training Complete!")
    print(f"{'='*80}")
    for metrics in all_stage_metrics:
        print(f"Stage {metrics['stage']}: "
              f"Conv Loss={metrics['final_convergence_loss']:.6f}, "
              f"Change Rate={metrics['final_change_rate']:.6f}, "
              f"Conv Metric={metrics['final_convergence_metric']:.6f}")
    print(f"{'='*80}")

    # Save final model
    final_path = os.path.join(args.output_dir, "new_llm_repetition_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'all_metrics': all_stage_metrics
    }, final_path)
    print(f"\n✓ Final model saved: {final_path}")


if __name__ == "__main__":
    main()
