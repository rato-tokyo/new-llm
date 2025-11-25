"""
Hyperparameter Optimization with Optuna

Searches for optimal hyperparameters to balance:
1. Convergence Rate (target: >50%)
2. Effective Rank (target: ~89%)

Uses Optuna with continuous logging to allow interruption and resumption.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import optuna
from optuna.storages import RDBStorage
import logging
from datetime import datetime
from config import ResidualConfig
from src.models.llm import LLM
from src.data.loader import load_data
from src.trainers.phase1 import Phase1Trainer
from src.evaluation.metrics import analyze_fixed_points

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"{log_dir}/optuna_optimization_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def objective(trial, config, train_token_ids, device, timeout_seconds=None):
    """
    Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object
        config: Base configuration
        train_token_ids: Training tokens
        device: torch device
        timeout_seconds: Optional timeout per trial

    Returns:
        score: Combined score (higher is better)
    """
    # Suggest hyperparameters
    dist_reg_weight = trial.suggest_float("dist_reg_weight", 0.1, 0.9)
    learning_rate = trial.suggest_float("phase1_learning_rate", 0.0005, 0.005, log=True)
    convergence_threshold = trial.suggest_float("phase1_convergence_threshold", 0.05, 0.5)
    layernorm_mix = trial.suggest_float("layernorm_mix", 0.5, 1.0)

    logger.info(f"\nTrial {trial.number} started:")
    logger.info(f"  dist_reg_weight={dist_reg_weight:.3f}")
    logger.info(f"  learning_rate={learning_rate:.6f}")
    logger.info(f"  convergence_threshold={convergence_threshold:.3f}")
    logger.info(f"  layernorm_mix={layernorm_mix:.3f}")

    try:
        # Create model with trial parameters
        layer_structure = [1] * config.num_layers
        model = LLM(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            context_dim=config.context_dim,
            hidden_dim=config.hidden_dim,
            layer_structure=layer_structure,
            layernorm_mix=layernorm_mix,
            use_pretrained_embeddings=True
        )
        model.to(device)

        # Create trainer with trial parameters
        trainer = Phase1Trainer(
            model=model,
            max_iterations=config.phase1_max_iterations,
            convergence_threshold=convergence_threshold,
            min_converged_ratio=config.phase1_min_converged_ratio,
            learning_rate=learning_rate,
            dist_reg_weight=dist_reg_weight
        )

        # Train
        contexts = trainer.train(train_token_ids, device, label=f"Trial-{trial.number}")

        # Calculate metrics
        convergence_rate = trainer.num_converged_tokens / len(train_token_ids)

        # Analyze effective rank (verbose=False for speed)
        metrics = analyze_fixed_points(contexts, label=f"Trial-{trial.number}", verbose=False)
        effective_rank_ratio = metrics['effective_rank'] / config.context_dim

        # Combined score: balance convergence and diversity
        # Weight: 50% convergence, 50% effective rank
        score = 0.5 * convergence_rate + 0.5 * effective_rank_ratio

        logger.info(f"Trial {trial.number} completed:")
        logger.info(f"  Convergence Rate: {convergence_rate*100:.1f}%")
        logger.info(f"  Effective Rank: {metrics['effective_rank']:.2f}/{config.context_dim} ({effective_rank_ratio*100:.1f}%)")
        logger.info(f"  Combined Score: {score:.4f}")

        # Report intermediate values for pruning
        trial.report(score, step=0)

        return score

    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}")
        raise optuna.TrialPruned()


def optimize(n_trials=50, timeout=None, n_jobs=1, study_name="cvfp_optimization"):
    """
    Run Optuna optimization.

    Args:
        n_trials: Number of trials to run
        timeout: Total timeout in seconds (None = no limit)
        n_jobs: Number of parallel jobs
        study_name: Name of the study (for resuming)
    """
    logger.info("="*70)
    logger.info("OPTUNA HYPERPARAMETER OPTIMIZATION")
    logger.info("="*70)
    logger.info(f"Study name: {study_name}")
    logger.info(f"Number of trials: {n_trials}")
    logger.info(f"Timeout: {timeout}s" if timeout else "Timeout: None")
    logger.info(f"Parallel jobs: {n_jobs}")
    logger.info(f"Log file: {log_file}")

    # Load configuration and data
    config = ResidualConfig()
    device = torch.device("cpu")  # Use CPU for stability across trials

    logger.info("\nLoading data...")
    train_token_ids, _ = load_data(config)

    # Use subset for faster trials
    subset_size = 1600  # 1/4 of full dataset for speed
    train_token_ids = train_token_ids[:subset_size]
    logger.info(f"Using {len(train_token_ids)} tokens for optimization")

    # Create or load study with SQLite storage for persistence
    storage_url = f"sqlite:///optuna_{study_name}.db"
    storage = RDBStorage(url=storage_url)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=0,
            interval_steps=1
        )
    )

    logger.info(f"\nStorage: {storage_url}")
    logger.info(f"Study has {len(study.trials)} trials from previous runs")

    # Run optimization
    try:
        study.optimize(
            lambda trial: objective(trial, config, train_token_ids, device),
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("\n⚠️  Optimization interrupted by user")

    # Report results
    logger.info("\n" + "="*70)
    logger.info("OPTIMIZATION RESULTS")
    logger.info("="*70)

    logger.info(f"\nTotal trials: {len(study.trials)}")
    logger.info(f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    logger.info(f"Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    logger.info(f"Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")

    if len(study.trials) > 0:
        logger.info("\n--- Best Trial ---")
        best_trial = study.best_trial
        logger.info(f"  Trial number: {best_trial.number}")
        logger.info(f"  Best score: {best_trial.value:.4f}")
        logger.info(f"\n  Best parameters:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")

        # Show top 5 trials
        logger.info("\n--- Top 5 Trials ---")
        sorted_trials = sorted(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
            key=lambda t: t.value,
            reverse=True
        )[:5]

        for i, trial in enumerate(sorted_trials, 1):
            logger.info(f"\n{i}. Trial {trial.number}: score={trial.value:.4f}")
            logger.info(f"   dist_reg_weight={trial.params['dist_reg_weight']:.3f}")
            logger.info(f"   learning_rate={trial.params['phase1_learning_rate']:.6f}")
            logger.info(f"   convergence_threshold={trial.params['phase1_convergence_threshold']:.3f}")
            logger.info(f"   layernorm_mix={trial.params['layernorm_mix']:.3f}")

    logger.info("\n" + "="*70)
    logger.info(f"Results saved to: {log_file}")
    logger.info(f"Study database: {storage_url}")
    logger.info("="*70)

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize CVFP hyperparameters with Optuna")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials to run")
    parser.add_argument("--timeout", type=int, default=None, help="Total timeout in seconds")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs")
    parser.add_argument("--study-name", type=str, default="cvfp_optimization", help="Study name")
    parser.add_argument("--quick-test", action="store_true", help="Run single trial for testing")

    args = parser.parse_args()

    if args.quick_test:
        logger.info("🔍 QUICK TEST MODE: Running single trial")
        args.n_trials = 1
        args.timeout = 300  # 5 minutes max

    study = optimize(
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        study_name=args.study_name
    )
