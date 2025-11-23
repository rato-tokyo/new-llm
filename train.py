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
from tokenizers import Tokenizer

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import ResidualConfig
from src.models.new_llm_residual import NewLLMResidual
from src.data.loader import load_data
from src.training.phase1_trainer import Phase1Trainer
from src.training.phase2 import train_phase2
from src.evaluation.metrics import analyze_fixed_points
from src.evaluation.diagnostics import check_identity_mapping, print_identity_mapping_warning


def print_flush(msg):
    """Print with immediate flush"""
    print(msg)
    sys.stdout.flush()


def main():
    """Main training function"""

    # Parse arguments
    parser = argparse.ArgumentParser(description='New-LLM Training')
    parser.add_argument('--test', action='store_true', help='Run quick test with 100 tokens')
    args = parser.parse_args()

    # Configuration
    config = ResidualConfig()
    device = torch.device(config.device)
    test_mode = args.test

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
    print_flush(f"   Distribution Reg: {config.use_distribution_reg} (weight={config.dist_reg_weight})")
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

    # Create model with refactored architecture
    layer_structure = [1] * config.num_layers
    model = NewLLMResidual(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        hidden_dim=config.hidden_dim,
        layer_structure=layer_structure,
        use_dist_reg=config.use_distribution_reg,
        ema_momentum=config.ema_momentum,
        layernorm_mix=1.0,  # Enabled to prevent value explosion
        enable_cvfp_learning=True  # トークンごとのオンライン学習を有効化
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_flush(f"\nModel initialized: {total_params:,} parameters")

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
    print_flush(f"\n{'='*70}")
    print_flush("STARTING PHASE 1")
    print_flush(f"{'='*70}\n")

    phase1_start = time.time()

    # Create Phase1Trainer
    trainer = Phase1Trainer(
        model=model,
        max_iterations=config.phase1_max_iterations,
        convergence_threshold=config.phase1_convergence_threshold,
        min_converged_ratio=config.phase1_min_converged_ratio,
        learning_rate=config.phase1_learning_rate,
        dist_reg_weight=config.dist_reg_weight
    )

    # Train
    train_contexts = trainer.train(train_token_ids, device, label="Train")

    # Validation
    val_contexts = trainer.evaluate(val_token_ids, device, label="Val")

    phase1_time = time.time() - phase1_start
    print_flush(f"\nPhase 1 completed in {phase1_time:.1f}s")

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

        phase2_start = time.time()

        train_phase2(
            model=model,
            train_contexts=train_contexts,
            train_token_ids=train_token_ids,
            val_contexts=val_contexts,
            val_token_ids=val_token_ids,
            config=config,
            device=device
        )

        phase2_time = time.time() - phase2_start
        print_flush(f"\nPhase 2 completed in {phase2_time:.1f}s")

    # Final summary
    print_flush(f"\n{'='*70}")
    print_flush("TRAINING COMPLETE")
    print_flush(f"{'='*70}\n")

    print_flush("✅ All training phases completed successfully")


if __name__ == "__main__":
    main()
