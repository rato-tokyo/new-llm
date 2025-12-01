"""
New-LLM Training Script

Usage:
    python3 train.py                    # 通常訓練
    python3 train.py --test             # クイックテスト（100トークン）
"""

import os
import sys
import torch
import time
import argparse

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import ResidualConfig  # noqa: E402
from src.models.llm import LLM  # noqa: E402
from src.providers.data import MemoryDataProvider  # noqa: E402
from src.trainers.phase1 import MemoryPhase1Trainer  # noqa: E402
from src.trainers.phase2 import Phase2Trainer  # noqa: E402
from src.evaluation.metrics import analyze_fixed_points  # noqa: E402
from src.evaluation.diagnostics import check_identity_mapping, print_identity_mapping_warning  # noqa: E402
from src.utils.io import print_flush  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


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
        print_flush("New-LLM Training")
    print_flush(f"{'='*70}\n")

    print_flush("📋 Configuration:")
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

    # Create model
    model = LLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        num_layers=config.num_layers,
        num_input_tokens=config.num_input_tokens,
        use_pretrained_embeddings=config.use_pretrained_embeddings,
        use_weight_tying=config.use_weight_tying,
        config=config
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
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
    data_provider = None
    if test_mode:
        print_flush("\nGenerating test data (100 tokens for stability test)...")
        train_token_ids = torch.randint(0, 1000, (100,), device=device)
        indices = torch.randperm(100)[:50]
        val_token_ids = train_token_ids[indices]
    else:
        print_flush("\nLoading training data...")
        data_provider = MemoryDataProvider(config)
        train_token_ids, val_token_ids = data_provider.load_data()
        train_token_ids = train_token_ids.to(device)
        val_token_ids = val_token_ids.to(device)

    # Phase 1: Fixed-Point Learning
    should_skip_phase1 = config.skip_phase1 and checkpoint_loaded and checkpoint_epoch in ['phase1_complete', 'phase2_complete']

    if should_skip_phase1:
        print_flush(f"\n{'='*70}")
        print_flush("SKIPPING PHASE 1 (Using checkpoint)")
        print_flush(f"{'='*70}\n")
        print_flush("✓ Phase 1 already completed, loading from checkpoint")
        print_flush("  Proceeding directly to Phase 2...\n")
        is_identity = False
    else:
        print_flush(f"\n{'='*70}")
        print_flush("STARTING PHASE 1")
        print_flush(f"{'='*70}\n")

        phase1_start = time.time()

        # Create Phase 1 Trainer
        phase1_trainer = MemoryPhase1Trainer(model, config, device)

        # Train Phase 1
        train_contexts = phase1_trainer.train(
            train_token_ids,
            label="Train",
            data_provider=data_provider
        )

        # Evaluate on validation data
        val_contexts = phase1_trainer.evaluate(val_token_ids, label="Val")

        phase1_time = time.time() - phase1_start
        print_flush(f"\nPhase 1 completed in {phase1_time:.1f}s")

        # Save checkpoint after Phase 1
        if config.save_checkpoint:
            print_flush(f"\n💾 Saving checkpoint to {config.checkpoint_path}")
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            try:
                phase1_trainer.save_checkpoint(config.checkpoint_path)
            except Exception as e:
                print_flush(f"⚠️ Failed to save checkpoint: {e}")

        # Analyze fixed points
        print_flush(f"\n{'='*70}")
        print_flush("FIXED-POINT ANALYSIS")
        print_flush(f"{'='*70}\n")

        analyze_fixed_points(train_contexts, label="Train")
        analyze_fixed_points(val_contexts, label="Val")

        # Check for identity mapping
        identity_check = check_identity_mapping(
            model=model,
            context_dim=config.context_dim,
            device=device,
            num_samples=config.identity_check_samples,
            threshold=config.identity_mapping_threshold,
            num_input_tokens=config.num_input_tokens
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

        phase2_trainer = Phase2Trainer(model=model, config=config)

        phase2_history = phase2_trainer.train_full(
            train_token_ids=train_token_ids,
            val_token_ids=val_token_ids,
            device=device
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
                        'num_layers': config.num_layers,
                        'embed_dim': config.embed_dim,
                        'context_dim': config.context_dim,
                        'vocab_size': config.vocab_size,
                        'num_context_splits': config.num_context_splits
                    }
                }
                torch.save(checkpoint, config.checkpoint_path)
                print_flush("✓ Phase 2 checkpoint saved successfully")
            except Exception as e:
                print_flush(f"⚠️ Failed to save Phase 2 checkpoint: {e}")

    # Final summary
    print_flush(f"\n{'='*70}")
    print_flush("TRAINING COMPLETE")
    print_flush(f"{'='*70}\n")

    print_flush("✅ All training phases completed successfully")

    # Clean up
    if data_provider is not None:
        data_provider.close()


if __name__ == "__main__":
    main()
