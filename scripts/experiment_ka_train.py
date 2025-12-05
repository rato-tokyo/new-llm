#!/usr/bin/env python3
"""
KA Training Experiment (案2)

学習時からKAキャッシュを使用するモデルの実験。
Baseline（KV方式）との比較。

Usage:
    python3 scripts/experiment_ka_train.py --samples 5000 --epochs 30

    # Baselineスキップ
    python3 scripts/experiment_ka_train.py --samples 5000 --skip-baseline
"""

import argparse
import sys
import time
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from config.pythia import PythiaConfig
from src.config.experiment_defaults import EARLY_STOPPING_PATIENCE
from src.models.pythia import PythiaModel
from src.models.ka_train import KATrainPythiaModel
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.training import (
    prepare_data_loaders,
    get_device,
    get_tokenizer,
    train_epoch,
    evaluate,
)
from src.utils.evaluation import evaluate_position_wise_ppl, evaluate_reversal_curse
from src.data.reversal_pairs import get_reversal_pairs
from src.utils.device import clear_gpu_cache


def train_model_ka(
    model: KATrainPythiaModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    lr: float,
) -> Dict[str, Any]:
    """
    KA方式でモデルを学習

    forward_parallel_kaを使用するためカスタム学習ループ。
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_ppl = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    print_flush("\n  Training KA model...")

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Train
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # KA方式でforward（forward_parallel_ka使用）
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

        train_loss = total_loss / total_tokens
        train_ppl = torch.exp(torch.tensor(train_loss)).item()

        # Validate
        model.eval()
        val_loss = 0.0
        val_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="sum",
                )
                val_loss += loss.item()
                val_tokens += labels.numel()

        val_ppl = torch.exp(torch.tensor(val_loss / val_tokens)).item()
        elapsed = time.time() - start_time

        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"    Epoch {epoch:2d}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
            f"[{elapsed:.1f}s] {marker}"
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print_flush("    -> Early stop")
            break

    print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
    }


def run_experiment(
    num_samples: int = 5000,
    seq_length: int = 128,
    num_epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    skip_baseline: bool = False,
) -> Dict[str, Any]:
    """Run KA training experiment"""
    set_seed(42)
    device = get_device()
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("KA TRAINING EXPERIMENT (案2)")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples:,}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Epochs: {num_epochs}")
    print_flush(f"Learning rate: {lr}")
    print_flush(f"Skip baseline: {skip_baseline}")
    print_flush("=" * 70)

    # Prepare data
    print_flush("\n[Data] Loading Pile data...")
    train_loader, val_loader = prepare_data_loaders(
        num_samples=num_samples,
        seq_length=seq_length,
        tokenizer_name=config.tokenizer_name,
        val_split=0.1,
        batch_size=batch_size,
    )

    results: Dict[str, Any] = {}

    # ===== Baseline (Pythia with KV) =====
    if not skip_baseline:
        print_flush("\n" + "=" * 70)
        print_flush("1. BASELINE (Pythia with KV Cache)")
        print_flush("=" * 70)

        pythia_model = PythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            rotary_pct=config.rotary_pct,
        )
        pythia_model = pythia_model.to(device)

        param_info = pythia_model.num_parameters()
        print_flush(f"  Parameters: {param_info['total']:,}")

        optimizer = torch.optim.AdamW(pythia_model.parameters(), lr=lr)

        print_flush("\n  Training Pythia (KV)...")
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
                f"    Epoch {epoch:2d}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
                f"[{elapsed:.1f}s] {marker}"
            )

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print_flush("    -> Early stop")
                break

        print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

        # Position-wise PPL
        print_flush("\n  Position-wise PPL:")
        pos_ppl = evaluate_position_wise_ppl(pythia_model, val_loader, device)
        for pos_range, ppl in pos_ppl.items():
            print_flush(f"    Position {pos_range}: {ppl:.1f}")

        # Reversal Curse evaluation
        print_flush("\n  Reversal Curse evaluation:")
        tokenizer = get_tokenizer(config.tokenizer_name)
        reversal_pairs = get_reversal_pairs()
        reversal = evaluate_reversal_curse(pythia_model, tokenizer, reversal_pairs, device)
        print_flush(f"    Forward PPL: {reversal['forward_ppl']:.1f}")
        print_flush(f"    Backward PPL: {reversal['backward_ppl']:.1f}")
        print_flush(f"    Reversal Ratio: {reversal['reversal_ratio']:.4f}")
        print_flush(f"    Reversal Gap: {reversal['reversal_gap']:.1f}")

        results["baseline"] = {
            "best_val_ppl": best_val_ppl,
            "best_epoch": best_epoch,
            "params": param_info["total"],
            "position_wise_ppl": pos_ppl,
            "reversal_curse": reversal,
        }

        del pythia_model
        clear_gpu_cache(device)

    # ===== KA Training Model =====
    print_flush("\n" + "=" * 70)
    print_flush("2. KA TRAINING MODEL (案2)")
    print_flush("=" * 70)

    ka_model = KATrainPythiaModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        rotary_pct=config.rotary_pct,
    )

    param_info = ka_model.num_parameters()
    print_flush(f"  Parameters: {param_info['total']:,}")

    ka_result = train_model_ka(
        model=ka_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
    )
    ka_model = ka_result["model"]

    # Position-wise PPL
    print_flush("\n  Position-wise PPL:")
    pos_ppl = evaluate_position_wise_ppl(ka_model, val_loader, device)
    for pos_range, ppl in pos_ppl.items():
        print_flush(f"    Position {pos_range}: {ppl:.1f}")

    # Reversal Curse evaluation
    print_flush("\n  Reversal Curse evaluation:")
    tokenizer = get_tokenizer(config.tokenizer_name)
    reversal_pairs = get_reversal_pairs()
    reversal = evaluate_reversal_curse(ka_model, tokenizer, reversal_pairs, device)
    print_flush(f"    Forward PPL: {reversal['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {reversal['backward_ppl']:.1f}")
    print_flush(f"    Reversal Ratio: {reversal['reversal_ratio']:.4f}")
    print_flush(f"    Reversal Gap: {reversal['reversal_gap']:.1f}")

    results["ka_train"] = {
        "best_val_ppl": ka_result["best_val_ppl"],
        "best_epoch": ka_result["best_epoch"],
        "params": param_info["total"],
        "position_wise_ppl": pos_ppl,
        "reversal_curse": reversal,
    }

    # ===== Summary =====
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | PPL | Epoch | Params |")
    print_flush("|-------|-----|-------|--------|")

    if "baseline" in results:
        r = results["baseline"]
        print_flush(f"| Pythia (KV) | {r['best_val_ppl']:.1f} | {r['best_epoch']} | {r['params']:,} |")

    r = results["ka_train"]
    print_flush(f"| KA Train (案2) | {r['best_val_ppl']:.1f} | {r['best_epoch']} | {r['params']:,} |")

    # Comparison
    if "baseline" in results:
        baseline_ppl = results["baseline"]["best_val_ppl"]
        ka_ppl = results["ka_train"]["best_val_ppl"]
        diff = ka_ppl - baseline_ppl
        diff_pct = (diff / baseline_ppl) * 100

        print_flush("\n" + "=" * 70)
        print_flush("ANALYSIS")
        print_flush("=" * 70)
        print_flush(f"\nKA Train vs Baseline: {diff:+.1f} ppl ({diff_pct:+.1f}%)")

        # Reversal Curse comparison
        print_flush("\n| Model | Forward PPL | Backward PPL | Ratio | Gap |")
        print_flush("|-------|-------------|--------------|-------|-----|")

        r = results["baseline"]
        rev = r["reversal_curse"]
        print_flush(
            f"| Pythia (KV) | {rev['forward_ppl']:.1f} | "
            f"{rev['backward_ppl']:.1f} | "
            f"{rev['reversal_ratio']:.4f} | "
            f"{rev['reversal_gap']:.1f} |"
        )

        r = results["ka_train"]
        rev = r["reversal_curse"]
        print_flush(
            f"| KA Train (案2) | {rev['forward_ppl']:.1f} | "
            f"{rev['backward_ppl']:.1f} | "
            f"{rev['reversal_ratio']:.4f} | "
            f"{rev['reversal_gap']:.1f} |"
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="KA Training Experiment")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
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
