"""
New-LLM Colab Training Script

Google Colabで実行するための完全な訓練スクリプト。
Phase 1からPhase 2まで一貫して実行し、すべての数値を表示します。

使い方:
    # Google Colabで
    !git clone https://github.com/your-repo/new-llm.git
    %cd new-llm
    !python colab.py

    # または
    !python colab.py --epochs 20  # Phase 2のエポック数を変更

出力される数値:
    - Phase 1: Effective Rank, 収束率, CVFP損失, 多様性損失
    - Phase 2: Perplexity (PPL), Loss, Accuracy, Context Stability Loss
"""

import os
import sys
import torch
import time
import argparse
import random
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def print_flush(msg):
    """Print with immediate flush for Colab compatibility"""
    print(msg)
    sys.stdout.flush()


def set_seed(seed=42):
    """全ての乱数生成器のシードを固定（完全な再現性保証）"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function for Colab"""

    # Parse arguments
    parser = argparse.ArgumentParser(description='New-LLM Colab Training')
    parser.add_argument('--epochs', type=int, default=10, help='Phase 2 epochs')
    parser.add_argument('--skip-phase1', action='store_true', help='Skip Phase 1 (use checkpoint)')
    parser.add_argument('--no-cache', action='store_true', help='Regenerate data cache')
    args = parser.parse_args()

    # Fix random seed
    set_seed(42)

    # Import after path setup
    from config import ResidualConfig
    from src.models.llm import LLM
    from src.data.loader import load_data
    from src.trainers.phase1 import phase1_train, forward_all_tokens_sequential
    from src.trainers.phase2 import Phase2Trainer
    from src.evaluation.metrics import analyze_fixed_points
    from src.evaluation.diagnostics import check_identity_mapping, print_identity_mapping_warning

    # Configuration
    config = ResidualConfig()
    device = torch.device(config.device)

    print_flush("\n" + "=" * 70)
    print_flush("New-LLM Training for Google Colab")
    print_flush("=" * 70 + "\n")

    print_flush("✅ Random seed fixed: 42 (完全な再現性保証)")
    print_flush(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        print_flush(f"   GPU: {torch.cuda.get_device_name(0)}")
        print_flush(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print_flush(f"\n📋 Configuration:")
    print_flush(f"   Architecture: {config.architecture}")
    print_flush(f"   Layers: {config.num_layers}")
    print_flush(f"   Context dim: {config.context_dim}")
    print_flush(f"   Diversity weight: {config.dist_reg_weight}")
    print_flush(f"   Phase 2 epochs: {args.epochs}")
    print_flush(f"   Context stability weight: {config.phase2_context_stability_weight}")

    # Delete cache if requested
    if args.no_cache:
        cache_file = f"./cache/ultrachat_{config.num_samples}samples_{config.max_seq_length}len.pt"
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print_flush(f"\n🗑️  Deleted cache: {cache_file}")

    # Setup tokenizer
    tokenizer_path = os.path.join(config.cache_dir, "tokenizer", "tokenizer.json")
    tokenizer_dir = os.path.dirname(tokenizer_path)

    if not os.path.exists(tokenizer_path):
        print_flush("\n📥 Downloading GPT-2 tokenizer...")
        from transformers import GPT2TokenizerFast
        gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=config.cache_dir)
        os.makedirs(tokenizer_dir, exist_ok=True)
        gpt2_tokenizer.save_pretrained(tokenizer_dir)
        print_flush("✓ Tokenizer saved")

    # Create model
    print_flush("\n📦 Creating model...")
    layer_structure = [1] * config.num_layers
    model = LLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        hidden_dim=config.hidden_dim,
        layer_structure=layer_structure,
        layernorm_mix=1.0,
        use_pretrained_embeddings=config.use_pretrained_embeddings
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print_flush(f"✓ Model created: {total_params:,} parameters")

    # Load data
    print_flush("\n📊 Loading data...")
    train_token_ids, val_token_ids = load_data(config)
    train_token_ids = train_token_ids.to(device)
    val_token_ids = val_token_ids.to(device)
    print_flush(f"   Train: {len(train_token_ids):,} tokens")
    print_flush(f"   Val:   {len(val_token_ids):,} tokens")

    # Load checkpoint if skipping Phase 1
    if args.skip_phase1 and os.path.exists(config.checkpoint_path):
        print_flush(f"\n📥 Loading checkpoint: {config.checkpoint_path}")
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print_flush("✓ Checkpoint loaded")

    # ========== PHASE 1 ==========
    if not args.skip_phase1:
        print_flush(f"\n{'=' * 70}")
        print_flush("PHASE 1: 固定点コンテキスト学習 (CVFP)")
        print_flush(f"{'=' * 70}\n")

        phase1_start = time.time()

        # Phase 1 Training
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

        # Validation evaluation
        print_flush("\n" + "=" * 70)
        print_flush("Evaluating on validation data...")
        print_flush("=" * 70 + "\n")

        val_token_embeds = model.token_embedding(val_token_ids.unsqueeze(0).to(device))
        if isinstance(val_token_embeds, tuple):
            val_token_embeds = val_token_embeds[0]
        val_token_embeds = model.embed_norm(val_token_embeds).squeeze(0)

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
        print_flush(f"\n⏱️  Phase 1 completed in {phase1_time:.1f}s")

        # Save checkpoint
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': 'phase1_complete',
            'config': {
                'num_layers': config.num_layers,
                'embed_dim': config.embed_dim,
                'context_dim': config.context_dim,
                'hidden_dim': config.hidden_dim,
                'vocab_size': config.vocab_size
            }
        }
        torch.save(checkpoint, config.checkpoint_path)
        print_flush(f"💾 Checkpoint saved: {config.checkpoint_path}")

        # Analyze fixed points
        print_flush(f"\n{'=' * 70}")
        print_flush("FIXED-POINT ANALYSIS")
        print_flush(f"{'=' * 70}\n")

        train_metrics = analyze_fixed_points(train_contexts, label="Train")
        val_metrics = analyze_fixed_points(val_contexts, label="Val")

        # Identity mapping check
        identity_check = check_identity_mapping(
            model=model,
            context_dim=config.context_dim,
            device=device,
            num_samples=config.identity_check_samples,
            threshold=config.identity_mapping_threshold
        )
        is_identity = print_identity_mapping_warning(identity_check)

        if is_identity:
            print_flush("\n⚠️  恒等写像が検出されたため、Phase 2をスキップします。")
            return
    else:
        is_identity = False

    # ========== PHASE 2 ==========
    print_flush(f"\n{'=' * 70}")
    print_flush("PHASE 2: トークン予測学習")
    print_flush(f"{'=' * 70}\n")

    phase2_start = time.time()

    # Create Phase 2 trainer
    phase2_trainer = Phase2Trainer(
        model=model,
        learning_rate=config.phase2_learning_rate,
        freeze_context=config.freeze_context,
        gradient_clip=config.phase2_gradient_clip,
        context_stability_weight=config.phase2_context_stability_weight
    )

    # Train Phase 2
    phase2_history = phase2_trainer.train_full(
        train_token_ids=train_token_ids,
        val_token_ids=val_token_ids,
        device=device,
        epochs=args.epochs
    )

    phase2_time = time.time() - phase2_start
    print_flush(f"\n⏱️  Phase 2 completed in {phase2_time:.1f}s")

    # Save final checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': 'phase2_complete',
        'phase2_history': phase2_history,
        'config': {
            'num_layers': config.num_layers,
            'embed_dim': config.embed_dim,
            'context_dim': config.context_dim,
            'hidden_dim': config.hidden_dim,
            'vocab_size': config.vocab_size
        }
    }
    torch.save(checkpoint, config.checkpoint_path)
    print_flush(f"💾 Final checkpoint saved: {config.checkpoint_path}")

    # ========== FINAL SUMMARY ==========
    print_flush(f"\n{'=' * 70}")
    print_flush("TRAINING COMPLETE - FINAL RESULTS")
    print_flush(f"{'=' * 70}\n")

    print_flush("📊 Phase 2 Results:")
    print_flush(f"   Final Train Loss: {phase2_history['train_loss'][-1]:.4f}")
    print_flush(f"   Final Train PPL:  {phase2_history['train_ppl'][-1]:.2f}")
    print_flush(f"   Final Val Loss:   {phase2_history['val_loss'][-1]:.4f}")
    print_flush(f"   Final Val PPL:    {phase2_history['val_ppl'][-1]:.2f}")
    print_flush(f"   Final Val Acc:    {phase2_history['val_acc'][-1] * 100:.2f}%")

    print_flush(f"\n📈 Best Results:")
    best_val_idx = np.argmin(phase2_history['val_loss'])
    print_flush(f"   Best Epoch:       {best_val_idx + 1}")
    print_flush(f"   Best Val Loss:    {phase2_history['val_loss'][best_val_idx]:.4f}")
    print_flush(f"   Best Val PPL:     {phase2_history['val_ppl'][best_val_idx]:.2f}")
    print_flush(f"   Best Val Acc:     {phase2_history['val_acc'][best_val_idx] * 100:.2f}%")

    print_flush(f"\n📉 Training Progress (Loss):")
    for i, (tl, vl) in enumerate(zip(phase2_history['train_loss'], phase2_history['val_loss'])):
        print_flush(f"   Epoch {i+1:2d}: Train={tl:.4f}, Val={vl:.4f}")

    print_flush(f"\n📉 Training Progress (PPL):")
    for i, (tp, vp) in enumerate(zip(phase2_history['train_ppl'], phase2_history['val_ppl'])):
        print_flush(f"   Epoch {i+1:2d}: Train={tp:.2f}, Val={vp:.2f}")

    print_flush(f"\n📊 Context Stability Loss:")
    for i, cl in enumerate(phase2_history['train_context_loss']):
        print_flush(f"   Epoch {i+1:2d}: {cl:.6f}")

    print_flush("\n✅ All training complete!")


if __name__ == "__main__":
    main()
