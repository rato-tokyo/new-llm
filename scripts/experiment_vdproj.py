#!/usr/bin/env python3
"""
V-DProj Experiment: Value Compression with Invertible Projection

V（Value）を圧縮して復元する方式でKVキャッシュを削減。
Pythiaベースラインとの比較実験。

Usage:
    python3 scripts/experiment_vdproj.py --samples 10000 --epochs 30
    python3 scripts/experiment_vdproj.py --samples 10000 --recon-weight 0.1
"""

import argparse
import sys
import time
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, ".")

from config.pythia import PythiaConfig  # noqa: E402
from src.config.experiment_defaults import (  # noqa: E402
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP,
    DEFAULT_RECON_WEIGHT,
)
from src.models.pythia import PythiaModel  # noqa: E402
from src.models.vdproj_pythia import VDProjPythiaModel  # noqa: E402
from src.utils.io import print_flush  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.training import prepare_data_loaders, get_device  # noqa: E402
from src.utils.evaluation import evaluate_ppl, evaluate_position_wise_ppl  # noqa: E402
from src.utils.device import clear_gpu_cache  # noqa: E402


def train_epoch_with_recon(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    recon_weight: float = 0.1,
) -> Dict[str, float]:
    """Train for one epoch with reconstruction loss."""
    model.train()
    total_lm_loss = 0.0
    total_recon_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for batch in train_loader:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, recon_loss = model(input_ids, return_reconstruction_loss=True)

        lm_loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        if recon_loss is not None:
            loss = lm_loss + recon_weight * recon_loss
            total_recon_loss += recon_loss.item()
        else:
            loss = lm_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

        total_lm_loss += lm_loss.item() * labels.numel()
        total_tokens += labels.numel()
        num_batches += 1

    avg_lm_loss = total_lm_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_lm_loss)).item()
    avg_recon_loss = total_recon_loss / num_batches if num_batches > 0 else 0.0

    return {"ppl": ppl, "recon_loss": avg_recon_loss}


def run_experiment(
    num_samples: int = 10000,
    seq_length: int = 128,
    num_epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    v_proj_dim: int = 320,
    recon_weight: float = DEFAULT_RECON_WEIGHT,
    skip_baseline: bool = False,
) -> Dict[str, Any]:
    """Run V-DProj experiment."""
    set_seed(42)

    device = get_device()
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("V-DPROJ EXPERIMENT: Value Compression")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples:,}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Epochs: {num_epochs}")
    print_flush(f"Learning rate: {lr}")
    print_flush(f"V proj dim: {v_proj_dim} (from {config.hidden_size})")
    print_flush(f"Reconstruction weight: {recon_weight}")
    print_flush(f"Skip baseline: {skip_baseline}")
    print_flush("=" * 70)

    reduction = (1.0 - v_proj_dim / config.hidden_size) / 2 * 100
    print_flush(f"\nKV Cache reduction (V only): {reduction:.1f}%")
    print_flush("=" * 70)

    print_flush("\n[Data] Loading Pile data...")
    train_loader, val_loader = prepare_data_loaders(
        num_samples=num_samples,
        seq_length=seq_length,
        tokenizer_name=config.tokenizer_name,
        val_split=0.1,
        batch_size=batch_size,
    )

    results: Dict[str, Any] = {}

    # ===== 1. Pythia Baseline =====
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

            # Train
            pythia_model.train()
            total_loss = 0.0
            total_tokens = 0
            for batch in train_loader:
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = pythia_model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pythia_model.parameters(), GRADIENT_CLIP)
                optimizer.step()

                total_loss += loss.item() * labels.numel()
                total_tokens += labels.numel()

            train_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()

            # Evaluate
            val_ppl = evaluate_ppl(pythia_model, val_loader, device)
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

        results["pythia"] = {
            "best_val_ppl": best_val_ppl,
            "best_epoch": best_epoch,
            "position_wise_ppl": pythia_pos_ppl,
        }

        del pythia_model
        clear_gpu_cache(device)

    # ===== 2. V-DProj Pythia =====
    print_flush("\n" + "=" * 70)
    print_flush(f"2. V-DPROJ PYTHIA (V: {config.hidden_size} → {v_proj_dim})")
    print_flush("=" * 70)

    vdproj_model = VDProjPythiaModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        v_proj_dim=v_proj_dim,
        max_position_embeddings=config.max_position_embeddings,
        rotary_pct=config.rotary_pct,
    )
    vdproj_model = vdproj_model.to(device)

    param_info = vdproj_model.num_parameters()
    print_flush(f"  Total parameters: {param_info['total']:,}")
    print_flush(f"  V projection: {param_info['v_projection']:,}")

    cache_info = vdproj_model.kv_cache_size(seq_length)
    print_flush(f"  KV Cache reduction: {cache_info['reduction_percent']:.1f}%")

    optimizer = torch.optim.AdamW(vdproj_model.parameters(), lr=lr)

    print_flush(f"\n[V-DProj] Training (recon_weight={recon_weight})...")
    best_val_ppl = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_metrics = train_epoch_with_recon(
            vdproj_model, train_loader, optimizer, device, recon_weight
        )
        val_metrics = evaluate_ppl(vdproj_model, val_loader, device, return_recon_loss=True)

        elapsed = time.time() - start_time

        improved = val_metrics["ppl"] < best_val_ppl
        if improved:
            best_val_ppl = val_metrics["ppl"]
            best_epoch = epoch
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train_ppl={train_metrics['ppl']:.1f} "
            f"val_ppl={val_metrics['ppl']:.1f} "
            f"recon={val_metrics['recon_loss']:.4f} "
            f"[{elapsed:.1f}s] {marker}"
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print_flush("  -> Early stop")
            break

    print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

    print_flush("\n  Position-wise PPL:")
    vdproj_pos_ppl = evaluate_position_wise_ppl(
        vdproj_model, val_loader, device, return_recon_loss=True
    )
    for pos_range, ppl in vdproj_pos_ppl.items():
        print_flush(f"    Position {pos_range}: {ppl:.1f}")

    results["vdproj"] = {
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
        "position_wise_ppl": vdproj_pos_ppl,
        "v_proj_dim": v_proj_dim,
        "recon_weight": recon_weight,
        "cache_reduction": cache_info["reduction_percent"],
    }

    del vdproj_model
    clear_gpu_cache(device)

    # ===== Summary =====
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | PPL | Epoch | KV Reduction |")
    print_flush("|-------|-----|-------|--------------|")

    if results["pythia"] is not None:
        print_flush(
            f"| Pythia | {results['pythia']['best_val_ppl']:.1f} | "
            f"{results['pythia']['best_epoch']} | 0% |"
        )

    print_flush(
        f"| V-DProj | {results['vdproj']['best_val_ppl']:.1f} | "
        f"{results['vdproj']['best_epoch']} | "
        f"{results['vdproj']['cache_reduction']:.1f}% |"
    )

    if results["pythia"] is not None:
        diff = results["vdproj"]["best_val_ppl"] - results["pythia"]["best_val_ppl"]
        print_flush(f"\nDifference: {diff:+.1f} ppl")

        print_flush("\n" + "=" * 70)
        print_flush("POSITION-WISE PPL")
        print_flush("=" * 70)

        pythia_pos = results["pythia"]["position_wise_ppl"]
        vdproj_pos = results["vdproj"]["position_wise_ppl"]

        print_flush("\n| Position | Pythia | V-DProj | Diff |")
        print_flush("|----------|--------|---------|------|")
        for pos_range in pythia_pos:
            pythia_val = pythia_pos[pos_range]
            vdproj_val = vdproj_pos[pos_range]
            pos_diff = vdproj_val - pythia_val
            print_flush(
                f"| {pos_range} | {pythia_val:.1f} | {vdproj_val:.1f} | {pos_diff:+.1f} |"
            )

    return results


def main() -> None:
    config = PythiaConfig()

    parser = argparse.ArgumentParser(description="V-DProj Experiment")
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
        "--v-proj-dim", type=int, default=320, help="V projection dimension"
    )
    parser.add_argument(
        "--recon-weight", type=float, default=DEFAULT_RECON_WEIGHT,
        help="Reconstruction loss weight"
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
        v_proj_dim=args.v_proj_dim,
        recon_weight=args.recon_weight,
        skip_baseline=args.skip_baseline,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
