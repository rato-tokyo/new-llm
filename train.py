"""
New-LLM Training Script (Refactored Version)

Uses the new CVFPLayer-based architecture with clean encapsulation.

Usage:
    python3 train.py           # Full training with config.py settings
    python3 train.py --test    # Quick test with 10 tokens only
"""

import os
import sys
import torch
import time
import argparse
import random
import numpy as np
from tokenizers import Tokenizer

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import ResidualConfig
from src.models.llm import LLM
from src.data.loader import load_data
from src.trainers.phase1 import phase1_train
from src.trainers.phase2 import Phase2Trainer
from src.evaluation.metrics import analyze_fixed_points
from src.evaluation.diagnostics import check_identity_mapping, print_identity_mapping_warning


def print_flush(msg):
    """Print with immediate flush"""
    print(msg)
    sys.stdout.flush()


def set_seed(seed=42):
    """全ての乱数生成器のシードを固定（完全な再現性保証）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 決定的動作を保証
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function"""

    # Fix random seed for reproducibility
    set_seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser(description='New-LLM Training')
    parser.add_argument('--test', action='store_true', help='Run quick test with 100 tokens')
    args = parser.parse_args()

    # Configuration
    config = ResidualConfig()
    device = torch.device(config.device)
    test_mode = args.test

    print_flush("\n✅ Random seed fixed: 42 (完全な再現性保証)")
    print_flush(f"\n{'='*70}")
    if test_mode:
        print_flush("New-LLM Quick Test Mode (100 tokens)")
    else:
        print_flush("New-LLM Training (Refactored Architecture)")
    print_flush(f"{'='*70}\n")

    print_flush("📋 Configuration:")
    print_flush(f"   Architecture: {config.architecture}")
    print_flush(f"   Layers: {config.num_layers}")
    print_flush(f"   Context dim: {config.context_dim}")
    print_flush(f"   Device: {config.device}")
    print_flush(f"   Diversity weight: {config.dist_reg_weight}")
    if not test_mode:
        print_flush(f"   Data: {config.num_samples} samples from {config.train_data_source}")

    # Load tokenizer
    tokenizer_path = os.path.join(config.cache_dir, "tokenizer", "tokenizer.json")
    tokenizer_dir = os.path.dirname(tokenizer_path)

    if not os.path.exists(tokenizer_path):
        print_flush("\n📥 Downloading GPT-2 tokenizer...")
        from transformers import GPT2TokenizerFast
        gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=config.cache_dir)
        os.makedirs(tokenizer_dir, exist_ok=True)
        gpt2_tokenizer.save_pretrained(tokenizer_dir)

    # Create model with E案 architecture
    model = LLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        context_layers=config.context_layers,
        token_layers=config.token_layers,
        use_pretrained_embeddings=config.use_pretrained_embeddings
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_flush(f"\nModel initialized: {total_params:,} parameters")

    # Load checkpoint if available
    checkpoint_loaded = False
    checkpoint_epoch = None
    if config.load_checkpoint and os.path.exists(config.checkpoint_path):
        print_flush(f"\n📥 Loading checkpoint from {config.checkpoint_path}")
        try:
            checkpoint = torch.load(config.checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint_epoch = checkpoint.get('epoch', 'unknown')
            checkpoint_loaded = True
            print_flush(f"✓ Loaded checkpoint (epoch {checkpoint_epoch})")
        except Exception as e:
            print_flush(f"⚠️ Failed to load checkpoint: {e}")
            print_flush("Starting training from scratch...")

    # Load data
    if test_mode:
        print_flush("\nGenerating test data (100 tokens for stability test)...")
        # 訓練データ: ランダムな100トークン
        train_token_ids = torch.randint(0, 1000, (100,), device=device)
        # 検証データ: 訓練データから50トークンをランダムサンプリング（重複なし）
        indices = torch.randperm(100)[:50]
        val_token_ids = train_token_ids[indices]
    else:
        print_flush("\nLoading training data...")
        train_token_ids, val_token_ids = load_data(config)

    # Phase 1: Fixed-Point Learning
    should_skip_phase1 = config.skip_phase1 and checkpoint_loaded and checkpoint_epoch in ['phase1_complete', 'phase2_complete']

    if should_skip_phase1:
        print_flush(f"\n{'='*70}")
        print_flush("SKIPPING PHASE 1 (Using checkpoint)")
        print_flush(f"{'='*70}\n")
        print_flush("✓ Phase 1 already completed, loading from checkpoint")
        print_flush("  Proceeding directly to Phase 2...\n")

        # Skip identity mapping check when using checkpoint
        is_identity = False

    else:
        print_flush(f"\n{'='*70}")
        print_flush("STARTING PHASE 1")
        print_flush(f"{'='*70}\n")

        phase1_start = time.time()

        # Phase 1: CVFP Learning with parallel processing
        train_contexts = phase1_train(
            model=model,
            token_ids=train_token_ids,
            device=device,
            learning_rate=config.phase1_learning_rate,
            max_iterations=config.phase1_max_iterations,
            convergence_threshold=config.phase1_convergence_threshold,
            dist_reg_weight=config.dist_reg_weight,
            min_converged_ratio=config.phase1_min_converged_ratio,
            label="Train"
        )

        # Validation: Use forward_all_tokens_sequential for evaluation
        print_flush("\n" + "="*70)
        print_flush("Evaluating on validation data...")
        print_flush("="*70 + "\n")

        from src.trainers.phase1 import forward_all_tokens_sequential

        # Prepare validation token embeddings
        val_token_embeds = model.token_embedding(val_token_ids.unsqueeze(0).to(device))
        if isinstance(val_token_embeds, tuple):
            val_token_embeds = val_token_embeds[0]
        val_token_embeds = model.embed_norm(val_token_embeds).squeeze(0)

        # Single forward pass (no training)
        model.eval()
        with torch.no_grad():
            val_contexts = forward_all_tokens_sequential(
                model=model,
                token_embeds=val_token_embeds,
                previous_contexts=None,
                device=device
            )
        model.train()

        phase1_time = time.time() - phase1_start
        print_flush(f"\nPhase 1 completed in {phase1_time:.1f}s")

        # Save checkpoint after Phase 1
        if config.save_checkpoint:
            print_flush(f"\n💾 Saving checkpoint to {config.checkpoint_path}")
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            try:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'epoch': 'phase1_complete',
                    'config': {
                        'context_layers': config.context_layers,
                        'token_layers': config.token_layers,
                        'embed_dim': config.embed_dim,
                        'context_dim': config.context_dim,
                        'vocab_size': config.vocab_size
                    }
                }
                torch.save(checkpoint, config.checkpoint_path)
                print_flush(f"✓ Checkpoint saved successfully")
            except Exception as e:
                print_flush(f"⚠️ Failed to save checkpoint: {e}")

        # Analyze fixed points
        print_flush(f"\n{'='*70}")
        print_flush("FIXED-POINT ANALYSIS")
        print_flush(f"{'='*70}\n")

        train_metrics = analyze_fixed_points(train_contexts, label="Train")
        val_metrics = analyze_fixed_points(val_contexts, label="Val")

        # Check for identity mapping
        identity_check = check_identity_mapping(
            model=model,
            context_dim=config.context_dim,
            device=device,
            num_samples=config.identity_check_samples,
            threshold=config.identity_mapping_threshold
        )
        is_identity = print_identity_mapping_warning(identity_check)

    # Phase 2: Token Prediction
    should_skip_phase2 = config.skip_phase2 or is_identity

    if is_identity:
        print_flush("\n⚠️  恒等写像が検出されたため、Phase 2をスキップします。")
        print_flush("    モデルの設定を見直してから再訓練することを推奨します。\n")

    if not should_skip_phase2:
        print_flush(f"\n{'='*70}")
        print_flush("STARTING PHASE 2")
        print_flush(f"{'='*70}\n")

        # Phase 2: Next-Token Prediction
        phase2_trainer = Phase2Trainer(
            model=model,
            learning_rate=config.phase2_learning_rate,
            gradient_clip=config.phase2_gradient_clip
        )

        # Train Phase 2
        phase2_history = phase2_trainer.train_full(
            train_token_ids=train_token_ids,
            val_token_ids=val_token_ids,
            device=device,
            epochs=config.phase2_epochs
        )

        # Save checkpoint after Phase 2
        if config.save_checkpoint:
            print_flush(f"\n💾 Saving Phase 2 checkpoint to {config.checkpoint_path}")
            try:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'epoch': 'phase2_complete',
                    'phase2_history': phase2_history,
                    'config': {
                        'context_layers': config.context_layers,
                        'token_layers': config.token_layers,
                        'embed_dim': config.embed_dim,
                        'context_dim': config.context_dim,
                        'vocab_size': config.vocab_size
                    }
                }
                torch.save(checkpoint, config.checkpoint_path)
                print_flush(f"✓ Phase 2 checkpoint saved successfully")
            except Exception as e:
                print_flush(f"⚠️ Failed to save Phase 2 checkpoint: {e}")

    # Final summary
    print_flush(f"\n{'='*70}")
    print_flush("TRAINING COMPLETE")
    print_flush(f"{'='*70}\n")

    print_flush("✅ All training phases completed successfully")


if __name__ == "__main__":
    main()
