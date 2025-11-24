"""
Optuna-based Hyperparameter Optimization for dist_reg_weight

このスクリプトは、Optunaを使用してdist_reg_weight（多様性正則化の重み）を自動調整します。

評価方法:
1. Phase 1（固定点学習）を実行
2. Phase 2（トークン予測）を実行
3. Validation Perplexityで評価
4. Phase 2に到達できない場合は、そのパラメータを不適格として扱う

使い方:
    python3 scripts/optimize_dist_reg_weight.py --n-trials 20 --timeout 3600
"""

import os
import sys
import time
import argparse
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import ResidualConfig
from src.models.new_llm_residual import NewLLMResidual
from src.models.new_llm_phase2 import expand_to_phase2
from src.data.loader import load_data
from src.training.phase1_trainer import Phase1Trainer
from src.training.phase2 import train_phase2_multioutput
from src.evaluation.metrics import analyze_fixed_points
from src.evaluation.diagnostics import check_identity_mapping


def print_flush(msg):
    """Print with immediate flush"""
    print(msg, flush=True)


def compute_perplexity(model, token_ids, device):
    """
    Compute perplexity on validation data

    Args:
        model: Trained Phase 2 model
        token_ids: Validation token IDs
        device: torch device

    Returns:
        perplexity: float
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        # Create input/target pairs
        input_ids = token_ids[:-1].unsqueeze(0).to(device)
        target_ids = token_ids[1:].to(device)

        # Get logits (use final block output)
        logits = model(input_ids)  # [1, seq_len, vocab_size]
        logits = logits.squeeze(0)  # [seq_len, vocab_size]

        # Compute cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_ids.reshape(-1),
            reduction='sum'
        )

        total_loss += loss.item()
        total_tokens += target_ids.numel()

    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return perplexity


def objective(trial, config, device, train_token_ids, val_token_ids, trial_times):
    """
    Optuna objective function

    Args:
        trial: Optuna trial
        config: Base configuration
        device: torch device
        train_token_ids: Training token IDs
        val_token_ids: Validation token IDs
        trial_times: List to track trial execution times

    Returns:
        perplexity: float (lower is better)
    """
    # Suggest dist_reg_weight (0.5 ~ 0.999)
    dist_reg_weight = trial.suggest_float('dist_reg_weight', 0.5, 0.999)

    print_flush(f"\n{'='*70}")
    print_flush(f"Trial {trial.number}: dist_reg_weight = {dist_reg_weight:.3f}")
    print_flush(f"{'='*70}\n")

    trial_start_time = time.time()

    try:
        # ========== Phase 1: Fixed-Point Learning ==========
        print_flush("PHASE 1: Fixed-Point Learning")

        # Create model
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

        # Create Phase1Trainer with trial's dist_reg_weight
        trainer = Phase1Trainer(
            model=model,
            max_iterations=config.phase1_max_iterations,
            convergence_threshold=config.phase1_convergence_threshold,
            min_converged_ratio=config.phase1_min_converged_ratio,
            learning_rate=config.phase1_learning_rate,
            dist_reg_weight=dist_reg_weight,  # Use trial's value
            ema_momentum=0.99
        )

        # Train
        train_contexts = trainer.train(train_token_ids, device, label="Train")
        val_contexts = trainer.evaluate(val_token_ids, device, label="Val")

        # Analyze fixed points
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

        if identity_check['is_identity']:
            print_flush("\n⚠️  Identity mapping detected - pruning trial")
            raise optuna.TrialPruned()

        # Check effective rank
        train_eff_rank_ratio = train_metrics['effective_rank'] / config.context_dim
        if train_eff_rank_ratio < 0.5:  # Less than 50% effective rank
            print_flush(f"\n⚠️  Low effective rank ({train_eff_rank_ratio*100:.1f}%) - pruning trial")
            raise optuna.TrialPruned()

        # ========== Phase 2: Token Prediction ==========
        print_flush("\nPHASE 2: Token Prediction")

        # Expand to Phase 2
        phase2_model = expand_to_phase2(model)
        phase2_model.to(device)

        # Create a temporary config for Phase 2
        class Phase2Config:
            freeze_context = config.freeze_context
            phase2_learning_rate = config.phase2_learning_rate
            phase2_epochs = min(config.phase2_epochs, 5)  # Limit epochs for speed
            phase2_batch_size = config.phase2_batch_size
            phase2_gradient_clip = config.phase2_gradient_clip
            checkpoint_dir = config.checkpoint_dir

        phase2_config = Phase2Config()

        # Train Phase 2
        train_phase2_multioutput(
            phase2_model,
            train_token_ids,
            val_token_ids,
            phase2_config,
            device
        )

        # Compute validation perplexity
        val_perplexity = compute_perplexity(phase2_model, val_token_ids, device)

        trial_time = time.time() - trial_start_time
        trial_times.append(trial_time)

        # Calculate ETA
        avg_trial_time = sum(trial_times) / len(trial_times)
        completed_trials = len(trial_times)

        print_flush(f"\n{'='*70}")
        print_flush(f"Trial {trial.number} Complete")
        print_flush(f"  dist_reg_weight: {dist_reg_weight:.3f}")
        print_flush(f"  Train Effective Rank: {train_metrics['effective_rank']:.1f}/{config.context_dim} ({train_eff_rank_ratio*100:.1f}%)")
        print_flush(f"  Val Perplexity: {val_perplexity:.2f}")
        print_flush(f"  Trial Time: {trial_time:.1f}s")
        print_flush(f"  Average Trial Time: {avg_trial_time:.1f}s")
        print_flush(f"{'='*70}\n")

        # Report intermediate value for pruning
        trial.report(val_perplexity, step=0)

        # Check if trial should be pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

        return val_perplexity

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print_flush(f"\n⚠️  Trial {trial.number} failed with error: {e}")
        print_flush("Treating as pruned trial\n")
        raise optuna.TrialPruned()


def main():
    parser = argparse.ArgumentParser(description='Optimize dist_reg_weight using Optuna')
    parser.add_argument('--n-trials', type=int, default=20, help='Number of trials')
    parser.add_argument('--timeout', type=int, default=None, help='Timeout in seconds')
    parser.add_argument('--study-name', type=str, default='dist_reg_weight_optimization', help='Study name')
    parser.add_argument('--storage', type=str, default=None, help='Database URL for study storage')
    args = parser.parse_args()

    print_flush(f"\n{'='*70}")
    print_flush("Optuna Hyperparameter Optimization: dist_reg_weight")
    print_flush(f"{'='*70}\n")

    print_flush(f"Configuration:")
    print_flush(f"  Number of trials: {args.n_trials}")
    print_flush(f"  Timeout: {args.timeout}s" if args.timeout else "  Timeout: None")
    print_flush(f"  Study name: {args.study_name}\n")

    # Load configuration
    config = ResidualConfig()
    device = torch.device(config.device)

    # Load data (once)
    print_flush("Loading data...")
    train_token_ids, val_token_ids = load_data(config)
    print_flush(f"  Train tokens: {len(train_token_ids)}")
    print_flush(f"  Val tokens: {len(val_token_ids)}\n")

    # Create Optuna study
    sampler = TPESampler(seed=config.random_seed)
    pruner = MedianPruner(
        n_startup_trials=5,  # Don't prune first 5 trials
        n_warmup_steps=0,
        interval_steps=1
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='minimize',  # Minimize perplexity
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True
    )

    optimization_start = time.time()

    # Track trial times for ETA calculation
    trial_times = []

    # Define callback for progress tracking
    def callback(study, trial):
        if len(trial_times) > 0:
            avg_time = sum(trial_times) / len(trial_times)
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            remaining = args.n_trials - completed
            eta_seconds = remaining * avg_time
            eta_minutes = eta_seconds / 60

            print_flush(f"\n📊 Progress: {completed}/{args.n_trials} trials complete")
            print_flush(f"⏱️  Estimated time remaining: {eta_minutes:.1f} minutes ({eta_seconds:.0f} seconds)\n")

    # Run optimization
    study.optimize(
        lambda trial: objective(trial, config, device, train_token_ids, val_token_ids, trial_times),
        n_trials=args.n_trials,
        timeout=args.timeout,
        callbacks=[callback],
        show_progress_bar=True
    )

    optimization_time = time.time() - optimization_start

    # Print results
    print_flush(f"\n{'='*70}")
    print_flush("OPTIMIZATION COMPLETE")
    print_flush(f"{'='*70}\n")

    print_flush(f"Total optimization time: {optimization_time/60:.1f} minutes")
    print_flush(f"Completed trials: {len(study.trials)}")
    print_flush(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print_flush(f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}\n")

    print_flush("Best trial:")
    best_trial = study.best_trial
    print_flush(f"  Trial number: {best_trial.number}")
    print_flush(f"  Validation Perplexity: {best_trial.value:.2f}")
    print_flush(f"  Best dist_reg_weight: {best_trial.params['dist_reg_weight']:.3f}\n")

    print_flush("Top 5 trials:")
    for i, trial in enumerate(sorted(study.trials, key=lambda t: t.value if t.value else float('inf'))[:5]):
        if trial.value is not None:
            print_flush(f"  {i+1}. Trial {trial.number}: dist_reg_weight={trial.params['dist_reg_weight']:.3f}, perplexity={trial.value:.2f}")

    # Save best config
    best_config_path = os.path.join(project_root, "config_optimized.py")
    print_flush(f"\n💾 Saving optimized config to {best_config_path}")

    with open(best_config_path, 'w') as f:
        f.write(f"""# Optimized Configuration (Generated by Optuna)
#
# Optimization Results:
# - Best dist_reg_weight: {best_trial.params['dist_reg_weight']:.3f}
# - Validation Perplexity: {best_trial.value:.2f}
# - Trial number: {best_trial.number}
# - Optimization time: {optimization_time/60:.1f} minutes
#
# Usage:
#   from config_optimized import OptimizedConfig
#   config = OptimizedConfig()

from config import ResidualConfig

class OptimizedConfig(ResidualConfig):
    # Optimized parameter
    dist_reg_weight = {best_trial.params['dist_reg_weight']:.3f}
""")

    print_flush(f"✅ Optimization complete! Use OptimizedConfig from config_optimized.py")


if __name__ == "__main__":
    main()
