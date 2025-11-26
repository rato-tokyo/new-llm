"""
New-LLM Colab Training Script (E案アーキテクチャ対応版)

Google Colabで実行するための完全な訓練スクリプト。
Phase 1（固定点学習）からPhase 2（トークン予測）まで一貫して実行し、すべての数値を表示します。

使い方:
    # Google Colabで
    !git clone https://github.com/your-repo/new-llm.git
    %cd new-llm
    !pip install torch transformers tokenizers datasets
    !python colab.py

    # オプション
    !python colab.py --epochs 20      # Phase 2のエポック数を変更
    !python colab.py --skip-phase1    # Phase 1をスキップ（チェックポイント使用）

出力される数値:
    - Phase 1: Effective Rank, 収束率, CVFP損失, 多様性損失
    - Phase 2: Perplexity (PPL), Loss, Accuracy
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

    # Parse arguments (実行時オプションのみ、設定値はconfig.pyから)
    parser = argparse.ArgumentParser(description='New-LLM Colab Training')
    parser.add_argument('--skip-phase1', action='store_true', help='Skip Phase 1 (use checkpoint)')
    parser.add_argument('--no-cache', action='store_true', help='Regenerate data cache')
    args = parser.parse_args()

    # Fix random seed
    set_seed(42)

    # Import after path setup
    from config import ResidualConfig
    from src.models.llm import LLM
    from src.trainers.phase1 import phase1_train, forward_all_tokens_sequential
    from src.trainers.phase2 import Phase2Trainer
    from src.evaluation.metrics import analyze_fixed_points
    from src.evaluation.diagnostics import check_identity_mapping, print_identity_mapping_warning
    from transformers import AutoTokenizer
    from datasets import load_dataset

    # Configuration
    config = ResidualConfig()
    device = torch.device(config.device)

    print_flush("\n" + "=" * 70)
    print_flush("New-LLM Training for Google Colab (E案アーキテクチャ)")
    print_flush("=" * 70 + "\n")

    print_flush("✅ Random seed fixed: 42 (完全な再現性保証)")
    print_flush(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        print_flush(f"   GPU: {torch.cuda.get_device_name(0)}")
        print_flush(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print_flush(f"\n📋 Configuration:")
    print_flush(f"   Architecture: E案 (Separated ContextBlock + TokenBlock)")
    print_flush(f"   Num layers: {config.num_layers}")
    print_flush(f"   Context dim: {config.context_dim}")
    print_flush(f"   Embed dim: {config.embed_dim}")
    print_flush(f"   Diversity weight: {config.dist_reg_weight}")
    print_flush(f"   Phase 2 epochs: {config.phase2_epochs}")
    print_flush(f"   Early stopping patience: {config.phase2_patience}")

    # 結果を格納する変数
    total_start_time = time.time()
    phase1_results = None
    phase2_results = None

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

    # Create model with E案 architecture
    print_flush("\n📦 Creating model (E案 architecture)...")
    model = LLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        num_layers=config.num_layers,
        use_pretrained_embeddings=config.use_pretrained_embeddings
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print_flush(f"✓ Model created: {total_params:,} parameters")

    # Load data directly from UltraChat (auto-generate validation from training)
    print_flush("\n📊 Loading data from UltraChat...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        cache_dir=os.path.join(config.cache_dir, "tokenizer")
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Check for cached data
    cache_file = os.path.join(
        config.cache_dir,
        f"ultrachat_{config.num_samples}samples_{config.max_seq_length}len.pt"
    )

    if os.path.exists(cache_file) and not args.no_cache:
        print_flush(f"   Loading from cache: {cache_file}")
        all_token_ids = torch.load(cache_file)
    else:
        print_flush(f"   Downloading UltraChat dataset...")
        dataset = load_dataset(
            config.dataset_name,
            split=config.dataset_split,
            cache_dir=os.path.join(config.cache_dir, "datasets")
        )

        # Process samples
        all_tokens = []
        for idx in range(min(config.num_samples, len(dataset))):
            messages = dataset[idx]["messages"]
            text = "\n".join([msg["content"] for msg in messages])

            tokens = tokenizer(
                text,
                max_length=config.max_seq_length,
                truncation=True,
                return_tensors="pt"
            )
            all_tokens.append(tokens["input_ids"].squeeze(0))

        all_token_ids = torch.cat(all_tokens)

        # Save cache
        os.makedirs(config.cache_dir, exist_ok=True)
        torch.save(all_token_ids, cache_file)
        print_flush(f"   Cached to: {cache_file}")

    print_flush(f"   Total tokens: {len(all_token_ids):,}")

    # Split into train/val with FIXED validation size
    fixed_val_size = 1280  # 検証データは常に1280トークン固定

    if len(all_token_ids) <= fixed_val_size:
        raise ValueError(
            f"❌ ERROR: Total tokens ({len(all_token_ids)}) must be greater than "
            f"fixed validation size ({fixed_val_size}). Increase num_samples."
        )

    train_size = len(all_token_ids) - fixed_val_size
    val_size = fixed_val_size

    train_token_ids = all_token_ids[:train_size].to(device)
    val_token_ids = all_token_ids[train_size:].to(device)

    print_flush(f"   Train: {len(train_token_ids):,} tokens")
    print_flush(f"   Val:   {len(val_token_ids):,} tokens (fixed size)")
    print_flush(f"   ✓ Validation auto-generated from training data")

    # Load checkpoint if skipping Phase 1
    if args.skip_phase1 and os.path.exists(config.checkpoint_path):
        print_flush(f"\n📥 Loading checkpoint: {config.checkpoint_path}")
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print_flush("✓ Checkpoint loaded")

    # ========== PHASE 1 ==========
    is_identity = False
    if not args.skip_phase1:
        print_flush(f"\n{'=' * 70}")
        print_flush("PHASE 1: 固定点コンテキスト学習 (CVFP) - ContextBlock")
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
                'context_layers': config.context_layers,
                'token_layers': config.token_layers,
                'embed_dim': config.embed_dim,
                'context_dim': config.context_dim,
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

        # Phase 1結果を保存
        train_er = train_metrics.get('effective_rank', 0)
        val_er = val_metrics.get('effective_rank', 0)
        phase1_results = {
            'train_effective_rank': train_er,
            'train_effective_rank_pct': train_er / config.context_dim * 100,
            'val_effective_rank': val_er,
            'val_effective_rank_pct': val_er / config.context_dim * 100,
            'time': phase1_time
        }

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

    # ========== PHASE 2 ==========
    print_flush(f"\n{'=' * 70}")
    print_flush("PHASE 2: トークン予測学習 (TokenBlock)")
    print_flush(f"{'=' * 70}\n")

    phase2_start = time.time()

    # Create Phase 2 trainer
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
        epochs=config.phase2_epochs,
        patience=config.phase2_patience,
        batch_size=config.phase2_batch_size
    )

    phase2_time = time.time() - phase2_start
    print_flush(f"\n⏱️  Phase 2 completed in {phase2_time:.1f}s")

    # Phase 2結果を保存
    best_epoch = phase2_history['best_epoch']
    phase2_results = {
        'best_epoch': best_epoch,
        'best_val_loss': phase2_history['val_loss'][best_epoch - 1],
        'best_val_ppl': phase2_history['val_ppl'][best_epoch - 1],
        'best_val_acc': phase2_history['val_acc'][best_epoch - 1],
        'final_val_ppl': phase2_history['val_ppl'][-1],
        'final_val_acc': phase2_history['val_acc'][-1],
        'early_stopped': phase2_history['early_stopped'],
        'stopped_epoch': phase2_history['stopped_epoch'],
        'total_epochs': len(phase2_history['train_loss']),
        'time': phase2_time
    }

    # Save final checkpoint
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
    print_flush(f"💾 Final checkpoint saved: {config.checkpoint_path}")

    # ========== FINAL SUMMARY ==========
    total_time = time.time() - total_start_time

    print_flush("\n")
    print_flush("=" * 70)
    print_flush("                    NEW-LLM TRAINING RESULTS                         ")
    print_flush("=" * 70)

    # Phase 1 結果
    print_flush("\n[PHASE 1: Context Learning (CVFP) - ContextBlock]")
    if phase1_results:
        print_flush(f"  Effective Rank (Train): {phase1_results['train_effective_rank_pct']:.1f}% ({phase1_results['train_effective_rank']:.2f}/{config.context_dim})")
        print_flush(f"  Effective Rank (Val):   {phase1_results['val_effective_rank_pct']:.1f}% ({phase1_results['val_effective_rank']:.2f}/{config.context_dim})")
        print_flush(f"  Time: {phase1_results['time']:.1f}s")
        print_flush(f"  Status: ✅ PASSED")
    else:
        print_flush(f"  Status: ⏭️  SKIPPED (using checkpoint)")

    # Phase 2 結果
    print_flush("\n[PHASE 2: Token Prediction - TokenBlock]")
    print_flush(f"  Best Val PPL:    {phase2_results['best_val_ppl']:.2f} (Epoch {phase2_results['best_epoch']})")
    print_flush(f"  Best Val Acc:    {phase2_results['best_val_acc'] * 100:.2f}%")
    print_flush(f"  Final Val PPL:   {phase2_results['final_val_ppl']:.2f}")
    print_flush(f"  Final Val Acc:   {phase2_results['final_val_acc'] * 100:.2f}%")
    print_flush(f"  Epochs Run:      {phase2_results['total_epochs']}/{args.epochs}")
    print_flush(f"  Time: {phase2_results['time']:.1f}s")

    if phase2_results['early_stopped']:
        print_flush(f"  Status: ⚠️  EARLY STOPPED at epoch {phase2_results['stopped_epoch']}")
    else:
        print_flush(f"  Status: ✅ COMPLETED all epochs")

    # 総合結果
    print_flush("\n" + "-" * 70)
    print_flush(f"  TOTAL TIME: {total_time:.1f}s")
    print_flush("=" * 70)

    # 詳細ログ
    print_flush("\n📉 Epoch-by-Epoch Progress:")
    print_flush("-" * 50)
    print_flush(f"{'Epoch':>6} | {'Train PPL':>10} | {'Val PPL':>10} | {'Val Acc':>8}")
    print_flush("-" * 50)
    for i in range(len(phase2_history['train_ppl'])):
        marker = " ⭐" if i + 1 == best_epoch else ""
        print_flush(f"{i+1:>6} | {phase2_history['train_ppl'][i]:>10.2f} | {phase2_history['val_ppl'][i]:>10.2f} | {phase2_history['val_acc'][i]*100:>7.2f}%{marker}")
    print_flush("-" * 50)

    print_flush("\n✅ Training complete!")


if __name__ == "__main__":
    main()
