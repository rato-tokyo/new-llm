#!/usr/bin/env python3
"""
KA-Attention vs Pythia Comparison Experiment

KA-Attention方式とBaselineのPythiaを比較する実験。

Usage:
    python3 scripts/experiment_ka_comparison.py --samples 10000 --epochs 10
"""

import argparse
import sys
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, ".")

from config.pythia import PythiaConfig
from src.models.pythia import PythiaModel
from src.models.ka_attention import KAPythiaModel
from src.utils.io import print_flush
from src.utils.training import prepare_data_loaders, train_model, get_device
from src.utils.device import clear_gpu_cache


def run_experiment(
    num_samples: int = 10000,
    seq_length: int = 128,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    skip_baseline: bool = False,
) -> Dict[str, Any]:
    """
    Run KA-Attention vs Pythia comparison experiment.

    Args:
        num_samples: Number of samples
        seq_length: Sequence length
        num_epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        skip_baseline: Skip baseline training

    Returns:
        Results dict
    """
    device = get_device()
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("KA-ATTENTION vs PYTHIA COMPARISON")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples:,}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Epochs: {num_epochs}")
    print_flush(f"Batch size: {batch_size}")
    print_flush(f"Learning rate: {lr}")
    print_flush("=" * 70)

    print_flush("\nArchitecture comparison:")
    print_flush(f"  Pythia:      hidden_size={config.hidden_size}, intermediate={config.intermediate_size}")
    print_flush(f"  KA-Pythia:   hidden_size={config.hidden_size}, intermediate={config.intermediate_size}")
    print_flush("  (Same architecture, different attention mechanism)")
    print_flush("=" * 70)

    # Load data
    print_flush("\n[Data] Loading Pile data...")
    train_loader, val_loader = prepare_data_loaders(
        num_samples=num_samples,
        seq_length=seq_length,
        tokenizer_name=config.tokenizer_name,
        batch_size=batch_size,
    )

    results: Dict[str, Any] = {}

    # 1. Baseline Pythia
    if not skip_baseline:
        print_flush("\n" + "=" * 70)
        print_flush("1. PYTHIA-70M (Baseline)")
        print_flush("=" * 70)

        pythia_model = PythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rotary_pct=config.rotary_pct,
        )

        print_flush("\n[Pythia-70M] Training...")
        pythia_results = train_model(
            pythia_model,
            train_loader,
            val_loader,
            device,
            num_epochs=num_epochs,
            learning_rate=lr,
            model_name="Pythia",
        )
        results["Pythia"] = pythia_results

        del pythia_model
        clear_gpu_cache(device)

    # 2. KA-Pythia
    print_flush("\n" + "=" * 70)
    print_flush("2. KA-PYTHIA (KA-Attention)")
    print_flush("=" * 70)

    ka_model = KAPythiaModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        rotary_pct=config.rotary_pct,
    )

    print_flush("\n[KA-Pythia] Training...")
    ka_results = train_model(
        ka_model,
        train_loader,
        val_loader,
        device,
        num_epochs=num_epochs,
        learning_rate=lr,
        model_name="KA-Pythia",
    )
    results["KA-Pythia"] = ka_results

    # Summary
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    for name, res in results.items():
        print_flush(f"  {name}: val_ppl={res['best_val_ppl']:.1f}")

    if "Pythia" in results and "KA-Pythia" in results:
        diff = results["KA-Pythia"]["best_val_ppl"] - results["Pythia"]["best_val_ppl"]
        pct = (diff / results["Pythia"]["best_val_ppl"]) * 100
        print_flush(f"\n  Difference: {diff:+.1f} ppl ({pct:+.1f}%)")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="KA-Attention vs Pythia Comparison")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline training")
    args = parser.parse_args()

    run_experiment(
        num_samples=args.samples,
        seq_length=args.seq_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        skip_baseline=args.skip_baseline,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
