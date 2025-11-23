"""
10,000 Token Experiment

Purpose:
1. Measure training time for Phase 1
2. Analyze convergence behavior
3. Evaluate Effective Rank and diversity
4. Provide data for config.py parameter tuning

This experiment will take approximately 10-11 hours on CPU.
"""

import torch
import sys
import time
import os
from datetime import datetime

sys.path.insert(0, '.')

from config import ResidualConfig
from src.models.new_llm_residual import NewLLMResidual
from src.training.phase1_trainer import Phase1Trainer
from src.evaluation.metrics import analyze_fixed_points

def print_flush(msg):
    """Print with timestamp and flush"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def run_10k_experiment():
    """Run 10,000 token experiment with detailed logging"""

    print_flush("="*70)
    print_flush("10,000 Token Experiment - Started")
    print_flush("="*70)

    config = ResidualConfig()
    device = torch.device(config.device)

    # Log configuration
    print_flush(f"\nConfiguration:")
    print_flush(f"  Device: {config.device}")
    print_flush(f"  Layers: {config.num_layers}")
    print_flush(f"  Embed dim: {config.embed_dim}")
    print_flush(f"  Context dim: {config.context_dim}")
    print_flush(f"  Max iterations: {config.phase1_max_iterations}")
    print_flush(f"  Learning rate: {config.phase1_learning_rate}")
    print_flush(f"  Diversity weight: {config.dist_reg_weight}")
    print_flush(f"  Convergence threshold: {config.phase1_convergence_threshold}")

    # Create model
    print_flush("\nCreating model...")
    layer_structure = [1] * config.num_layers
    model = NewLLMResidual(
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_flush(f"  Total parameters: {total_params:,}")
    print_flush(f"  Trainable parameters: {trainable_params:,}")

    # Load checkpoint if exists
    if config.load_checkpoint and os.path.exists(config.checkpoint_path):
        print_flush(f"\nLoading checkpoint from {config.checkpoint_path}")
        checkpoint = torch.load(config.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print_flush(f"✓ Checkpoint loaded")

    # Generate data
    print_flush("\nGenerating 10,000 token dataset...")
    num_train = 10000
    num_val = 2000

    train_tokens = torch.randint(0, 1000, (num_train,), device=device)
    val_tokens = torch.randint(0, 1000, (num_val,), device=device)

    print_flush(f"  Training tokens: {num_train:,}")
    print_flush(f"  Validation tokens: {num_val:,}")

    # Create trainer
    print_flush("\nInitializing Phase 1 trainer...")
    trainer = Phase1Trainer(
        model=model,
        max_iterations=config.phase1_max_iterations,
        convergence_threshold=config.phase1_convergence_threshold,
        min_converged_ratio=config.phase1_min_converged_ratio,
        learning_rate=config.phase1_learning_rate,
        dist_reg_weight=config.dist_reg_weight
    )

    # Training
    print_flush("\n" + "="*70)
    print_flush("PHASE 1 TRAINING - 10,000 TOKENS")
    print_flush("="*70)

    experiment_start = time.time()

    # Train
    train_start = time.time()
    train_contexts = trainer.train(train_tokens, device, label="Train")
    train_time = time.time() - train_start

    # Validation
    val_start = time.time()
    val_contexts = trainer.evaluate(val_tokens, device, label="Val")
    val_time = time.time() - val_start

    total_time = time.time() - experiment_start

    # Performance metrics
    print_flush("\n" + "="*70)
    print_flush("PERFORMANCE METRICS")
    print_flush("="*70)

    train_tokens_per_sec = num_train * trainer.current_iteration / train_time
    val_tokens_per_sec = num_val * trainer.current_iteration / val_time

    print_flush(f"\nTiming:")
    print_flush(f"  Training time: {train_time:.1f}s ({train_time/60:.1f}min)")
    print_flush(f"  Validation time: {val_time:.1f}s ({val_time/60:.1f}min)")
    print_flush(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")

    print_flush(f"\nThroughput:")
    print_flush(f"  Training: {train_tokens_per_sec:.2f} tokens/sec")
    print_flush(f"  Validation: {val_tokens_per_sec:.2f} tokens/sec")
    print_flush(f"  Average: {(train_tokens_per_sec + val_tokens_per_sec)/2:.2f} tokens/sec")

    print_flush(f"\nConvergence:")
    print_flush(f"  Iterations completed: {trainer.current_iteration}/{config.phase1_max_iterations}")
    print_flush(f"  Training convergence rate: {trainer.train_convergence_rate*100:.1f}%")

    # Analyze fixed points
    print_flush("\n" + "="*70)
    print_flush("FIXED-POINT ANALYSIS")
    print_flush("="*70)

    train_metrics = analyze_fixed_points(train_contexts, label="Train")
    val_metrics = analyze_fixed_points(val_contexts, label="Val")

    # Save detailed results
    print_flush("\n" + "="*70)
    print_flush("SAVING RESULTS")
    print_flush("="*70)

    results = {
        'config': {
            'num_tokens_train': num_train,
            'num_tokens_val': num_val,
            'num_layers': config.num_layers,
            'embed_dim': config.embed_dim,
            'context_dim': config.context_dim,
            'learning_rate': config.phase1_learning_rate,
            'dist_reg_weight': config.dist_reg_weight,
            'convergence_threshold': config.phase1_convergence_threshold,
        },
        'performance': {
            'train_time_sec': train_time,
            'val_time_sec': val_time,
            'total_time_sec': total_time,
            'train_tokens_per_sec': train_tokens_per_sec,
            'val_tokens_per_sec': val_tokens_per_sec,
            'iterations': trainer.current_iteration,
            'train_convergence_rate': trainer.train_convergence_rate,
        },
        'metrics': {
            'train': train_metrics,
            'val': val_metrics,
        },
        'timestamp': datetime.now().isoformat(),
    }

    results_path = os.path.join(config.checkpoint_dir, "experiment_10k_results.pt")
    torch.save(results, results_path)
    print_flush(f"  Results saved to: {results_path}")

    # Save checkpoint
    if config.save_checkpoint:
        print_flush(f"\n  Saving model checkpoint...")
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': 'experiment_10k',
            'config': {
                'num_layers': config.num_layers,
                'embed_dim': config.embed_dim,
                'context_dim': config.context_dim,
                'hidden_dim': config.hidden_dim,
                'vocab_size': config.vocab_size
            }
        }
        torch.save(checkpoint, config.checkpoint_path)
        print_flush(f"  ✓ Checkpoint saved to: {config.checkpoint_path}")

    # Summary for parameter tuning
    print_flush("\n" + "="*70)
    print_flush("PARAMETER TUNING RECOMMENDATIONS")
    print_flush("="*70)

    print_flush(f"\n📊 Current Configuration Performance:")
    print_flush(f"  • Processing speed: {train_tokens_per_sec:.1f} tokens/sec")
    print_flush(f"  • Convergence: {trainer.train_convergence_rate*100:.1f}% in {trainer.current_iteration} iterations")
    print_flush(f"  • Train Effective Rank: {train_metrics['effective_rank']:.2f}/{config.context_dim} ({train_metrics['effective_rank']/config.context_dim*100:.1f}%)")
    print_flush(f"  • Val Effective Rank: {val_metrics['effective_rank']:.2f}/{config.context_dim} ({val_metrics['effective_rank']/config.context_dim*100:.1f}%)")

    print_flush(f"\n💡 Analysis:")

    # Convergence analysis
    if trainer.train_convergence_rate < 0.5:
        print_flush(f"  ⚠️ Low convergence rate ({trainer.train_convergence_rate*100:.1f}%)")
        print_flush(f"     → Consider: Increase learning_rate or max_iterations")
    elif trainer.train_convergence_rate >= 0.95:
        print_flush(f"  ✓ Excellent convergence ({trainer.train_convergence_rate*100:.1f}%)")
        print_flush(f"     → Current settings are effective")

    # Effective Rank analysis
    train_rank_ratio = train_metrics['effective_rank'] / config.context_dim
    val_rank_ratio = val_metrics['effective_rank'] / config.context_dim

    if train_rank_ratio < 0.5:
        print_flush(f"  ⚠️ Low Effective Rank ({train_rank_ratio*100:.1f}%)")
        print_flush(f"     → Consider: Increase dist_reg_weight")
    elif train_rank_ratio >= 0.8:
        print_flush(f"  ✓ High diversity achieved ({train_rank_ratio*100:.1f}%)")
        print_flush(f"     → Diversity regularization working well")

    # Generalization analysis
    rank_gap = train_rank_ratio - val_rank_ratio
    if abs(rank_gap) > 0.1:
        print_flush(f"  ⚠️ Train/Val Effective Rank gap: {rank_gap*100:.1f}%")
        print_flush(f"     → May indicate overfitting or underfitting")

    # Time projection
    print_flush(f"\n⏱️ Time Projections (based on {train_tokens_per_sec:.1f} tok/s):")
    for tokens in [50000, 100000, 500000, 1000000]:
        estimated_hours = (tokens * trainer.current_iteration) / train_tokens_per_sec / 3600
        print_flush(f"  • {tokens:,} tokens: ~{estimated_hours:.1f} hours")

    print_flush("\n" + "="*70)
    print_flush("EXPERIMENT COMPLETE")
    print_flush("="*70)
    print_flush(f"\nTotal elapsed time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print_flush(f"Results and checkpoint saved for future analysis.")

    return results

if __name__ == "__main__":
    try:
        results = run_10k_experiment()
    except Exception as e:
        print_flush(f"\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
