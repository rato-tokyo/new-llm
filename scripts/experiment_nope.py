#!/usr/bin/env python3
"""
NoPE Experiment: Position Encoding Ablation Study

位置エンコーディングの重要性を測定するアブレーション実験。
Pythia (RoPE) vs NoPE-Pythia (位置情報なし) の比較。

Usage:
    python3 scripts/experiment_nope.py --samples 10000 --epochs 30
    python3 scripts/experiment_nope.py --samples 10000 --skip-baseline
"""

import argparse
import sys
import time
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, ".")

from config.pythia import PythiaConfig  # noqa: E402
from src.config.experiment_defaults import (  # noqa: E402
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP,
)
from src.models.pythia import PythiaModel  # noqa: E402
from src.models.nope_pythia import NoPEPythiaModel  # noqa: E402
from src.data.reversal_pairs import get_reversal_pairs  # noqa: E402
from src.utils.io import print_flush  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.training import prepare_data_loaders, get_device, get_tokenizer  # noqa: E402
from src.utils.evaluation import (  # noqa: E402
    evaluate_ppl,
    evaluate_position_wise_ppl,
    evaluate_reversal_curse,
)
from src.utils.device import clear_gpu_cache  # noqa: E402


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch. Returns train PPL."""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in train_loader:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)

        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl


def run_experiment(
    num_samples: int = 10000,
    seq_length: int = 128,
    num_epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    skip_baseline: bool = False,
) -> dict[str, Any]:
    """Run NoPE experiment."""
    set_seed(42)

    device = get_device()
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("NoPE EXPERIMENT: Position Encoding Ablation Study")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples:,}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Epochs: {num_epochs}")
    print_flush(f"Learning rate: {lr}")
    print_flush(f"Skip baseline: {skip_baseline}")
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

            train_ppl = train_epoch(pythia_model, train_loader, optimizer, device)
            val_ppl_result = evaluate_ppl(pythia_model, val_loader, device)
            val_ppl = float(val_ppl_result) if isinstance(val_ppl_result, (int, float)) else val_ppl_result["ppl"]
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

    # ===== 2. NoPE-Pythia (No Position Encoding) =====
    print_flush("\n" + "=" * 70)
    print_flush("2. NoPE-PYTHIA (No Position Encoding)")
    print_flush("=" * 70)

    nope_model = NoPEPythiaModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
    )
    nope_model = nope_model.to(device)

    param_info = nope_model.num_parameters()
    print_flush(f"  Total parameters: {param_info['total']:,}")
    print_flush("  Position encoding: NONE")

    optimizer = torch.optim.AdamW(nope_model.parameters(), lr=lr)

    print_flush("\n[NoPE] Training...")
    best_val_ppl = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_ppl = train_epoch(nope_model, train_loader, optimizer, device)
        val_ppl_result = evaluate_ppl(nope_model, val_loader, device)
        val_ppl = float(val_ppl_result) if isinstance(val_ppl_result, (int, float)) else val_ppl_result["ppl"]
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
    nope_pos_ppl = evaluate_position_wise_ppl(nope_model, val_loader, device)
    for pos_range, ppl in nope_pos_ppl.items():
        print_flush(f"    Position {pos_range}: {ppl:.1f}")

    # Reversal Curse evaluation
    print_flush("\n  Reversal Curse evaluation:")
    tokenizer = get_tokenizer(config.tokenizer_name)
    reversal_pairs = get_reversal_pairs()
    nope_reversal = evaluate_reversal_curse(
        nope_model, tokenizer, reversal_pairs, device
    )
    print_flush(f"    Forward PPL: {nope_reversal['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {nope_reversal['backward_ppl']:.1f}")
    print_flush(f"    Reversal Ratio: {nope_reversal['reversal_ratio']:.3f}")
    print_flush(f"    Reversal Gap: {nope_reversal['reversal_gap']:+.1f}")

    results["nope"] = {
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
        "position_wise_ppl": nope_pos_ppl,
        "reversal_curse": nope_reversal,
    }

    del nope_model
    clear_gpu_cache(device)

    # ===== Summary =====
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | Position Encoding | PPL | Epoch |")
    print_flush("|-------|-------------------|-----|-------|")

    if results["pythia"] is not None:
        print_flush(
            f"| Pythia | RoPE (25%) | {results['pythia']['best_val_ppl']:.1f} | "
            f"{results['pythia']['best_epoch']} |"
        )

    print_flush(
        f"| NoPE-Pythia | None | {results['nope']['best_val_ppl']:.1f} | "
        f"{results['nope']['best_epoch']} |"
    )

    if results["pythia"] is not None:
        diff = results["nope"]["best_val_ppl"] - results["pythia"]["best_val_ppl"]
        diff_pct = (diff / results["pythia"]["best_val_ppl"]) * 100
        print_flush(f"\nPPL Degradation without Position Encoding: {diff:+.1f} ({diff_pct:+.1f}%)")

        print_flush("\n" + "=" * 70)
        print_flush("POSITION-WISE PPL")
        print_flush("=" * 70)

        pythia_pos = results["pythia"]["position_wise_ppl"]
        nope_pos = results["nope"]["position_wise_ppl"]

        print_flush("\n| Position | Pythia (RoPE) | NoPE | Diff |")
        print_flush("|----------|---------------|------|------|")
        for pos_range in pythia_pos:
            pythia_val = pythia_pos[pos_range]
            nope_val = nope_pos[pos_range]
            pos_diff = nope_val - pythia_val
            print_flush(
                f"| {pos_range} | {pythia_val:.1f} | {nope_val:.1f} | {pos_diff:+.1f} |"
            )

        # Reversal Curse comparison
        print_flush("\n" + "=" * 70)
        print_flush("REVERSAL CURSE")
        print_flush("=" * 70)

        pythia_rev = results["pythia"]["reversal_curse"]
        nope_rev = results["nope"]["reversal_curse"]

        print_flush("\n| Model | Forward PPL | Backward PPL | Ratio | Gap |")
        print_flush("|-------|-------------|--------------|-------|-----|")
        print_flush(
            f"| Pythia (RoPE) | {pythia_rev['forward_ppl']:.1f} | "
            f"{pythia_rev['backward_ppl']:.1f} | "
            f"{pythia_rev['reversal_ratio']:.3f} | "
            f"{pythia_rev['reversal_gap']:+.1f} |"
        )
        print_flush(
            f"| NoPE | {nope_rev['forward_ppl']:.1f} | "
            f"{nope_rev['backward_ppl']:.1f} | "
            f"{nope_rev['reversal_ratio']:.3f} | "
            f"{nope_rev['reversal_gap']:+.1f} |"
        )

        print_flush("\n(Reversal Ratio closer to 1.0 = less reversal curse)")

    print_flush("\n" + "=" * 70)
    print_flush("ANALYSIS")
    print_flush("=" * 70)
    print_flush("\nExpected observations:")
    print_flush("- NoPE should have significantly higher PPL (worse)")
    print_flush("- Position-wise PPL should be more uniform for NoPE")
    print_flush("  (since it cannot distinguish position)")
    print_flush("- NoPE may show less 'reversal curse' pattern")
    print_flush("  (position-agnostic = direction-agnostic)")

    return results


def main() -> None:
    config = PythiaConfig()

    parser = argparse.ArgumentParser(description="NoPE Experiment")
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
        "--skip-baseline", action="store_true", help="Skip Pythia baseline"
    )
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
