#!/usr/bin/env python3
"""
New-LLM Training Script

Clean, simple entry point for training the model.
"""

import torch
import os
import sys
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from config import ResidualConfig
from src.models.new_llm_residual import NewLLMResidual
from src.data.loader import load_data
from src.training.phase1 import train_phase1
from src.training.phase2 import train_phase2
from src.evaluation.metrics import analyze_fixed_points, analyze_singular_vectors


def print_flush(msg):
    """Print with immediate flush"""
    print(msg, flush=True)


def main():
    """Main training function"""
    # Load configuration
    config = ResidualConfig()

    print_flush("="*70)
    print_flush("New-LLM Training")
    print_flush("="*70)
    print_flush(f"\n📋 Configuration:")
    print_flush(f"   Architecture: {config.architecture}")
    print_flush(f"   Layers: {config.num_layers}")
    print_flush(f"   Context dim: {config.context_dim}")
    print_flush(f"   Device: {config.device}")
    print_flush(f"   Distribution Reg: {config.use_distribution_reg} (weight={config.dist_reg_weight})")
    print_flush(f"   Data: {config.num_samples} samples from {config.train_data_source}")
    print_flush("")

    # Set device
    device = torch.device(config.device)

    # Initialize model
    # layer_structure: [1] * num_layers for residual_standard
    layer_structure = [1] * config.num_layers

    model = NewLLMResidual(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        hidden_dim=config.hidden_dim,
        layer_structure=layer_structure
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print_flush(f"Model initialized: {total_params:,} parameters\n")

    # Load data
    train_token_ids, val_token_ids = load_data(config)

    # Phase 1: Fixed-Point Context Learning
    print_flush("\n" + "="*70)
    print_flush("STARTING PHASE 1")
    print_flush("="*70)

    # Train Phase 1
    train_contexts = train_phase1(
        model, train_token_ids, config, device,
        is_training=True, label="Train"
    )

    # Save Phase 1 model
    checkpoint_path = os.path.join(config.cache_dir, "phase1_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print_flush(f"Model checkpoint saved: {checkpoint_path}\n")

    # Analyze Train fixed points
    train_metrics = analyze_fixed_points(train_contexts, label="Train")

    # Val Phase 1
    val_contexts = train_phase1(
        model, val_token_ids, config, device,
        is_training=False, label="Val"
    )

    # Analyze Val fixed points
    val_metrics = analyze_fixed_points(val_contexts, label="Val")

    # Check if Phase 1 was successful
    MIN_TRAIN_RANK = config.context_dim * 0.2  # 20% minimum
    MIN_VAL_RANK = config.context_dim * 0.08   # 8% minimum

    if train_metrics["effective_rank"] < MIN_TRAIN_RANK or val_metrics["effective_rank"] < MIN_VAL_RANK:
        print_flush("\n" + "="*70)
        print_flush("⚠️  PHASE 1 FAILED - DIMENSION COLLAPSE DETECTED")
        print_flush("="*70)
        print_flush(f"\n  Train Effective Rank: {train_metrics['effective_rank']:.2f}/{config.context_dim} (required: >= {MIN_TRAIN_RANK:.1f})")
        print_flush(f"  Val Effective Rank:   {val_metrics['effective_rank']:.2f}/{config.context_dim} (required: >= {MIN_VAL_RANK:.1f})")
        print_flush(f"\n  ❌ Phase 2 skipped. Fix dimension collapse first.")
        print_flush(f"  ❌ See documentation for solutions.\n")
        print_flush("="*70 + "\n")
        return

    # Optional: Singular vector analysis
    if config.context_dim <= 16:  # Only for small models
        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            cache_dir=os.path.join(config.cache_dir, "tokenizer")
        )
        analyze_singular_vectors(train_contexts, train_token_ids, tokenizer)

    # Phase 2: Token Prediction (if not skipped)
    if not config.skip_phase2:
        print_flush("\n" + "="*70)
        print_flush("STARTING PHASE 2")
        print_flush("="*70)

        train_phase2(
            model, train_token_ids, val_token_ids,
            train_contexts, val_contexts,
            config, device
        )

        # Save final model
        final_path = os.path.join(config.cache_dir, "final_model.pt")
        torch.save(model.state_dict(), final_path)
        print_flush(f"\nFinal model saved: {final_path}")
    else:
        print_flush("\n⚠️ Phase 2 skipped (skip_phase2=True in config)")

    print_flush("\n" + "="*70)
    print_flush("TRAINING COMPLETE")
    print_flush("="*70 + "\n")


if __name__ == "__main__":
    main()