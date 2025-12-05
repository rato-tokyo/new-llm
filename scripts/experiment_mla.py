#!/usr/bin/env python3
"""
MLA Experiment: KV Cache Compression with ALiBi

MLAによるKVキャッシュ圧縮実験。
Pythia (RoPE) vs MLA-Pythia (ALiBi) の比較。

Usage:
    python3 scripts/experiment_mla.py --samples 10000 --epochs 30
    python3 scripts/experiment_mla.py --samples 10000 --skip-baseline
"""

import argparse
import sys
import time
from typing import Any

import torch

# Add project root to path
sys.path.insert(0, ".")

from config.pythia import PythiaConfig  # noqa: E402
from src.config.experiment_defaults import EARLY_STOPPING_PATIENCE  # noqa: E402
from src.models.pythia import PythiaModel  # noqa: E402
from src.models.mla_pythia import MLAPythiaModel  # noqa: E402
from src.data.reversal_pairs import get_reversal_pairs  # noqa: E402
from src.utils.io import print_flush  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.training import (  # noqa: E402
    prepare_data_loaders,
    get_device,
    get_tokenizer,
    train_epoch,
    evaluate,
)
from src.utils.evaluation import (  # noqa: E402
    evaluate_position_wise_ppl,
    evaluate_reversal_curse,
)
from src.utils.device import clear_gpu_cache  # noqa: E402


def run_experiment(
    num_samples: int = 10000,
    seq_length: int = 128,
    num_epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    kv_dim: int = 128,
    alibi_slope: float = 0.0625,
    skip_baseline: bool = False,
) -> dict[str, Any]:
    """Run MLA experiment."""
    set_seed(42)

    device = get_device()
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("MLA EXPERIMENT: KV Cache Compression with ALiBi")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples:,}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Epochs: {num_epochs}")
    print_flush(f"Learning rate: {lr}")
    print_flush(f"KV dim: {kv_dim} (from {config.hidden_size})")
    print_flush(f"ALiBi slope: {alibi_slope}")
    print_flush(f"Skip baseline: {skip_baseline}")
    print_flush("=" * 70)

    # KV Cache reduction calculation
    standard_kv = config.hidden_size * 2  # K + V
    mla_kv = kv_dim  # c_kv only
    reduction = (standard_kv - mla_kv) / standard_kv * 100
    print_flush(f"\nKV Cache reduction: {reduction:.1f}%")
    print_flush("=" * 70)

    print_flush("\n[Data] Loading Pile data...")
    train_loader, val_loader = prepare_data_loaders(
        num_samples=num_samples,
        seq_length=seq_length,
        tokenizer_name=config.tokenizer_name,
        val_split=0.1,
        batch_size=batch_size,
    )

    results: dict[str, Any] = {}

    # ===== 1. Pythia Baseline (RoPE) =====
    if skip_baseline:
        print_flush("\n" + "=" * 70)
        print_flush("1. PYTHIA-70M (Baseline, RoPE) - SKIPPED")
        print_flush("=" * 70)
        results["pythia"] = None
    else:
        print_flush("\n" + "=" * 70)
        print_flush("1. PYTHIA-70M (Baseline, RoPE)")
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
        pythia_model = pythia_model.to(device)

        param_info = pythia_model.num_parameters()
        print_flush(f"  Total parameters: {param_info['total']:,}")

        optimizer = torch.optim.AdamW(pythia_model.parameters(), lr=lr)

        print_flush("\n[Pythia] Training...")
        best_val_ppl = float("inf")
        best_epoch = 0
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            train_loss = train_epoch(pythia_model, train_loader, optimizer, device)
            train_ppl = torch.exp(torch.tensor(train_loss)).item()
            _, val_ppl = evaluate(pythia_model, val_loader, device)
            elapsed = time.time() - start_time

            improved = val_ppl < best_val_ppl
            if improved:
                best_val_ppl = val_ppl
                best_epoch = epoch
                patience_counter = 0
                marker = "*"
            else:
                patience_counter += 1
                marker = ""

            print_flush(
                f"  Epoch {epoch:2d}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
                f"[{elapsed:.1f}s] {marker}"
            )

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print_flush("  -> Early stop")
                break

        print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

        print_flush("\n  Position-wise PPL:")
        pythia_pos_ppl = evaluate_position_wise_ppl(pythia_model, val_loader, device)
        for pos_range, ppl in pythia_pos_ppl.items():
            print_flush(f"    Position {pos_range}: {ppl:.1f}")

        # Reversal Curse evaluation
        print_flush("\n  Reversal Curse evaluation:")
        tokenizer = get_tokenizer(config.tokenizer_name)
        reversal_pairs = get_reversal_pairs()
        pythia_reversal = evaluate_reversal_curse(
            pythia_model, tokenizer, reversal_pairs, device
        )
        print_flush(f"    Forward PPL: {pythia_reversal['forward_ppl']:.1f}")
        print_flush(f"    Backward PPL: {pythia_reversal['backward_ppl']:.1f}")
        print_flush(f"    Reversal Ratio: {pythia_reversal['reversal_ratio']:.3f}")
        print_flush(f"    Reversal Gap: {pythia_reversal['reversal_gap']:+.1f}")

        results["pythia"] = {
            "best_val_ppl": best_val_ppl,
            "best_epoch": best_epoch,
            "position_wise_ppl": pythia_pos_ppl,
            "reversal_curse": pythia_reversal,
        }

        del pythia_model
        clear_gpu_cache(device)

    # ===== 2. MLA-Pythia (ALiBi) =====
    print_flush("\n" + "=" * 70)
    print_flush(f"2. MLA-PYTHIA (kv_dim={kv_dim}, ALiBi)")
    print_flush("=" * 70)

    mla_model = MLAPythiaModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        kv_dim=kv_dim,
        q_compressed=False,  # KV圧縮のみ（Q圧縮なし）
        alibi_slope=alibi_slope,
    )
    mla_model = mla_model.to(device)

    param_info = mla_model.num_parameters()
    print_flush(f"  Total parameters: {param_info['total']:,}")
    print_flush(f"  MLA attention: {param_info['mla_attention']:,}")

    cache_info = mla_model.kv_cache_size(seq_length)
    print_flush(f"  KV Cache reduction: {cache_info['reduction_percent']:.1f}%")

    optimizer = torch.optim.AdamW(mla_model.parameters(), lr=lr)

    print_flush("\n[MLA] Training...")
    best_val_ppl = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(mla_model, train_loader, optimizer, device)
        train_ppl = torch.exp(torch.tensor(train_loss)).item()
        _, val_ppl = evaluate(mla_model, val_loader, device)
        elapsed = time.time() - start_time

        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            best_epoch = epoch
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
            f"[{elapsed:.1f}s] {marker}"
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print_flush("  -> Early stop")
            break

    print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

    print_flush("\n  Position-wise PPL:")
    mla_pos_ppl = evaluate_position_wise_ppl(mla_model, val_loader, device)
    for pos_range, ppl in mla_pos_ppl.items():
        print_flush(f"    Position {pos_range}: {ppl:.1f}")

    # Reversal Curse evaluation
    print_flush("\n  Reversal Curse evaluation:")
    tokenizer = get_tokenizer(config.tokenizer_name)
    reversal_pairs = get_reversal_pairs()
    mla_reversal = evaluate_reversal_curse(
        mla_model, tokenizer, reversal_pairs, device
    )
    print_flush(f"    Forward PPL: {mla_reversal['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {mla_reversal['backward_ppl']:.1f}")
    print_flush(f"    Reversal Ratio: {mla_reversal['reversal_ratio']:.3f}")
    print_flush(f"    Reversal Gap: {mla_reversal['reversal_gap']:+.1f}")

    results["mla"] = {
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
        "position_wise_ppl": mla_pos_ppl,
        "reversal_curse": mla_reversal,
        "kv_dim": kv_dim,
        "cache_reduction": cache_info["reduction_percent"],
    }

    del mla_model
    clear_gpu_cache(device)

    # ===== Summary =====
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | PPL | Epoch | KV Reduction |")
    print_flush("|-------|-----|-------|--------------|")

    if results["pythia"] is not None:
        print_flush(
            f"| Pythia (RoPE) | {results['pythia']['best_val_ppl']:.1f} | "
            f"{results['pythia']['best_epoch']} | 0% |"
        )

    print_flush(
        f"| MLA (ALiBi) | {results['mla']['best_val_ppl']:.1f} | "
        f"{results['mla']['best_epoch']} | "
        f"{results['mla']['cache_reduction']:.1f}% |"
    )

    if results["pythia"] is not None:
        diff = results["mla"]["best_val_ppl"] - results["pythia"]["best_val_ppl"]
        print_flush(f"\nDifference: {diff:+.1f} ppl")

        print_flush("\n" + "=" * 70)
        print_flush("POSITION-WISE PPL")
        print_flush("=" * 70)

        pythia_pos = results["pythia"]["position_wise_ppl"]
        mla_pos = results["mla"]["position_wise_ppl"]

        print_flush("\n| Position | Pythia | MLA | Diff |")
        print_flush("|----------|--------|-----|------|")
        for pos_range in pythia_pos:
            pythia_val = pythia_pos[pos_range]
            mla_val = mla_pos[pos_range]
            pos_diff = mla_val - pythia_val
            print_flush(
                f"| {pos_range} | {pythia_val:.1f} | {mla_val:.1f} | {pos_diff:+.1f} |"
            )

        # Reversal Curse comparison
        print_flush("\n" + "=" * 70)
        print_flush("REVERSAL CURSE")
        print_flush("=" * 70)

        pythia_rev = results["pythia"]["reversal_curse"]
        mla_rev = results["mla"]["reversal_curse"]

        print_flush("\n| Model | Forward PPL | Backward PPL | Ratio | Gap |")
        print_flush("|-------|-------------|--------------|-------|-----|")
        print_flush(
            f"| Pythia | {pythia_rev['forward_ppl']:.1f} | "
            f"{pythia_rev['backward_ppl']:.1f} | "
            f"{pythia_rev['reversal_ratio']:.3f} | "
            f"{pythia_rev['reversal_gap']:+.1f} |"
        )
        print_flush(
            f"| MLA | {mla_rev['forward_ppl']:.1f} | "
            f"{mla_rev['backward_ppl']:.1f} | "
            f"{mla_rev['reversal_ratio']:.3f} | "
            f"{mla_rev['reversal_gap']:+.1f} |"
        )

        print_flush("\n(Reversal Ratio closer to 1.0 = less reversal curse)")

    return results


def main() -> None:
    config = PythiaConfig()

    parser = argparse.ArgumentParser(description="MLA Experiment")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument(
        "--epochs", type=int, default=config.num_epochs, help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=config.batch_size, help="Batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=config.learning_rate, help="Learning rate"
    )
    parser.add_argument(
        "--kv-dim", type=int, default=128, help="KV compression dimension"
    )
    parser.add_argument(
        "--alibi-slope", type=float, default=0.0625, help="ALiBi slope (uniform)"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip Pythia baseline"
    )
    args = parser.parse_args()

    run_experiment(
        num_samples=args.samples,
        seq_length=args.seq_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        kv_dim=args.kv_dim,
        alibi_slope=args.alibi_slope,
        skip_baseline=args.skip_baseline,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
