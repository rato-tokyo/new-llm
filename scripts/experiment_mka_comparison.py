#!/usr/bin/env python3
"""
MKA-Attention vs KA-Attention vs Pythia Comparison Experiment

Mini-Context KA-Attention方式とKA-Attention、Baselineを比較する実験。

MKA-Attention:
- Stage 1: 直近mini_context_lengthトークンの純粋KV attention → local A
- Stage 2: 全過去の純粋KA attention → global output

Usage:
    python3 scripts/experiment_mka_comparison.py --samples 10000 --epochs 10
    python3 scripts/experiment_mka_comparison.py --samples 10000 --mini-context 16
"""

import argparse
import sys
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, ".")

from config.pythia import PythiaConfig
from src.models.pythia import PythiaModel
from src.models.ka_attention import KAPythiaModel
from src.models.mka_attention import MKAPythiaModel
from src.utils.io import print_flush
from src.utils.training import prepare_data_loaders, train_model, get_device
from src.utils.device import clear_gpu_cache


def run_experiment(
    num_samples: int = 10000,
    seq_length: int = 128,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    mini_context_length: int = 16,
    skip_baseline: bool = False,
    skip_ka: bool = False,
) -> Dict[str, Any]:
    """
    Run MKA vs KA vs Pythia comparison experiment

    Args:
        num_samples: Number of training samples
        seq_length: Sequence length
        num_epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        mini_context_length: Mini context length for MKA
        skip_baseline: Skip Pythia baseline
        skip_ka: Skip KA-Attention

    Returns:
        Results dict
    """
    device = get_device()
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("MKA-ATTENTION vs KA-ATTENTION vs PYTHIA COMPARISON")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples:,}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Epochs: {num_epochs}")
    print_flush(f"Learning rate: {lr}")
    print_flush(f"Mini context length: {mini_context_length}")
    print_flush("=" * 70)

    print_flush("\nArchitecture comparison:")
    print_flush("  Pythia:     Standard attention")
    print_flush("  KA-Pythia:  V->A replacement (mixed cache)")
    print_flush("  MKA-Pythia: Two-stage (pure KV + pure KA)")
    print_flush("=" * 70)

    # Load data
    print_flush("\n[Data] Loading Pile data...")
    train_loader, val_loader = prepare_data_loaders(
        num_samples=num_samples,
        seq_length=seq_length,
        tokenizer_name=config.tokenizer_name,
        val_split=0.1,
        batch_size=batch_size,
    )

    results: Dict[str, Any] = {}

    # ===== 1. Pythia (Baseline) =====
    if skip_baseline:
        print_flush("\n" + "=" * 70)
        print_flush("1. PYTHIA-70M (Baseline) - SKIPPED")
        print_flush("=" * 70)
        results["pythia"] = None
    else:
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
            patience=config.early_stopping_patience,
            model_name="Pythia-70M",
        )
        results["pythia"] = pythia_results

        del pythia_model
        clear_gpu_cache(device)

    # ===== 2. KA-Pythia =====
    if skip_ka:
        print_flush("\n" + "=" * 70)
        print_flush("2. KA-PYTHIA (KA-Attention) - SKIPPED")
        print_flush("=" * 70)
        results["ka_pythia"] = None
    else:
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
            patience=config.early_stopping_patience,
            model_name="KA-Pythia",
        )
        results["ka_pythia"] = ka_results

        del ka_model
        clear_gpu_cache(device)

    # ===== 3. MKA-Pythia =====
    print_flush("\n" + "=" * 70)
    print_flush(f"3. MKA-PYTHIA (Mini-Context={mini_context_length})")
    print_flush("=" * 70)

    mka_model = MKAPythiaModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        mini_context_length=mini_context_length,
        max_position_embeddings=config.max_position_embeddings,
        rotary_pct=config.rotary_pct,
    )

    print_flush("\n[MKA-Pythia] Training...")
    mka_results = train_model(
        mka_model,
        train_loader,
        val_loader,
        device,
        num_epochs=num_epochs,
        learning_rate=lr,
        patience=config.early_stopping_patience,
        model_name="MKA-Pythia",
    )
    results["mka_pythia"] = mka_results

    del mka_model
    clear_gpu_cache(device)

    # ===== Summary =====
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | Best PPL | Best Epoch |")
    print_flush("|-------|----------|------------|")

    if results["pythia"] is not None:
        print_flush(
            f"| Pythia-70M | {results['pythia']['best_val_ppl']:.1f} | "
            f"{results['pythia']['best_epoch']} |"
        )
    else:
        print_flush("| Pythia-70M | (skipped) | - |")

    if results["ka_pythia"] is not None:
        print_flush(
            f"| KA-Pythia | {results['ka_pythia']['best_val_ppl']:.1f} | "
            f"{results['ka_pythia']['best_epoch']} |"
        )
    else:
        print_flush("| KA-Pythia | (skipped) | - |")

    print_flush(
        f"| MKA-Pythia (w={mini_context_length}) | {results['mka_pythia']['best_val_ppl']:.1f} | "
        f"{results['mka_pythia']['best_epoch']} |"
    )

    # Differences
    if results["pythia"] is not None:
        mka_diff = results["mka_pythia"]["best_val_ppl"] - results["pythia"]["best_val_ppl"]
        print_flush(f"\nMKA vs Pythia: {mka_diff:+.1f} ppl")

    if results["ka_pythia"] is not None:
        mka_ka_diff = results["mka_pythia"]["best_val_ppl"] - results["ka_pythia"]["best_val_ppl"]
        print_flush(f"MKA vs KA: {mka_ka_diff:+.1f} ppl")

    return results


def main() -> None:
    # Get defaults from config
    config = PythiaConfig()

    parser = argparse.ArgumentParser(description="MKA-Attention vs KA-Attention vs Pythia")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=config.num_epochs, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.learning_rate, help="Learning rate")
    parser.add_argument("--mini-context", type=int, default=16, help="Mini context length")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip Pythia baseline")
    parser.add_argument("--skip-ka", action="store_true", help="Skip KA-Attention")
    args = parser.parse_args()

    run_experiment(
        num_samples=args.samples,
        seq_length=args.seq_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        mini_context_length=args.mini_context,
        skip_baseline=args.skip_baseline,
        skip_ka=args.skip_ka,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
