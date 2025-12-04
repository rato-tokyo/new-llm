#!/usr/bin/env python3
"""
Pretrained MKA-Attention Experiment

事前学習済みPythiaを凍結し、KA部分のみを学習する実験。

3つのモデルを比較:
1. Pythia-70M (pretrained, frozen): ベースライン
2. PretrainedMKA V2: Pythia全体凍結 + KA attention学習

Usage:
    python3 scripts/experiment_pretrained_mka.py --samples 10000 --epochs 10
    python3 scripts/experiment_pretrained_mka.py --samples 10000 --eval-only
"""

import argparse
import sys
import time
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM

# Add project root to path
sys.path.insert(0, ".")

from config.pythia import PythiaConfig  # noqa: E402
from src.config.experiment_defaults import (  # noqa: E402
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP,
)
from src.models.pretrained_mka import PretrainedMKAModelV2  # noqa: E402
from src.utils.io import print_flush  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.training import prepare_data_loaders, get_device  # noqa: E402
from src.utils.evaluation import evaluate_ppl, evaluate_position_wise_ppl  # noqa: E402
from src.utils.device import clear_gpu_cache  # noqa: E402


class PythiaWrapper(nn.Module):
    """Wrapper to make HuggingFace model compatible with evaluation functions."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids).logits


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
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
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    eval_only: bool = False,
) -> Dict[str, Any]:
    """Run pretrained MKA experiment."""
    set_seed(42)

    device = get_device()
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("PRETRAINED MKA-ATTENTION EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples:,}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Epochs: {num_epochs}")
    print_flush(f"Learning rate: {lr}")
    print_flush(f"Eval only: {eval_only}")
    print_flush("=" * 70)

    print_flush("\nArchitecture:")
    print_flush("  Pythia-70M (pretrained): Frozen baseline")
    print_flush("  PretrainedMKA V2: Pythia frozen + KA attention trainable")
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

    # ===== 1. Pretrained Pythia-70M (Baseline) =====
    print_flush("\n" + "=" * 70)
    print_flush("1. PYTHIA-70M (Pretrained, Frozen)")
    print_flush("=" * 70)

    pretrained_pythia = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    pretrained_pythia = pretrained_pythia.to(device)
    pretrained_pythia.eval()

    pythia_wrapper = PythiaWrapper(pretrained_pythia)

    print_flush("\n[Pythia-70M] Evaluating pretrained model...")
    pythia_ppl = evaluate_ppl(pythia_wrapper, val_loader, device)
    print_flush(f"  Pretrained val_ppl: {pythia_ppl:.1f}")

    print_flush("\n  Position-wise PPL (long-range dependency):")
    pythia_pos_ppl = evaluate_position_wise_ppl(pythia_wrapper, val_loader, device)
    for pos_range, ppl in pythia_pos_ppl.items():
        print_flush(f"    Position {pos_range}: {ppl:.1f}")

    results["pythia_pretrained"] = {
        "val_ppl": pythia_ppl,
        "position_wise_ppl": pythia_pos_ppl,
    }

    del pretrained_pythia, pythia_wrapper
    clear_gpu_cache(device)

    if eval_only:
        print_flush("\n[Eval only mode] Skipping training.")
        return results

    # ===== 2. PretrainedMKA V2 =====
    print_flush("\n" + "=" * 70)
    print_flush("2. PRETRAINED-MKA V2 (Pythia frozen + KA trainable)")
    print_flush("=" * 70)

    mka_model = PretrainedMKAModelV2(pretrained_model_name="EleutherAI/pythia-70m")
    mka_model = mka_model.to(device)

    param_info = mka_model.num_parameters()
    print_flush(f"  Total parameters: {param_info['total']:,}")
    print_flush(f"  Trainable: {param_info['trainable']:,}")
    print_flush(f"  Frozen: {param_info['frozen']:,}")

    trainable_params = [p for p in mka_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    print_flush("\n[PretrainedMKA V2] Training...")

    best_val_ppl = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_ppl = train_epoch(mka_model, train_loader, optimizer, device)
        val_ppl = evaluate_ppl(mka_model, val_loader, device)

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

    print_flush("\n  Position-wise PPL (long-range dependency):")
    mka_pos_ppl = evaluate_position_wise_ppl(mka_model, val_loader, device)
    for pos_range, ppl in mka_pos_ppl.items():
        print_flush(f"    Position {pos_range}: {ppl:.1f}")

    results["pretrained_mka_v2"] = {
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
        "position_wise_ppl": mka_pos_ppl,
    }

    del mka_model
    clear_gpu_cache(device)

    # ===== Summary =====
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | PPL | Note |")
    print_flush("|-------|-----|------|")
    print_flush(
        f"| Pythia-70M (pretrained) | {results['pythia_pretrained']['val_ppl']:.1f} | "
        "frozen baseline |"
    )
    print_flush(
        f"| PretrainedMKA V2 | {results['pretrained_mka_v2']['best_val_ppl']:.1f} | "
        f"epoch {results['pretrained_mka_v2']['best_epoch']} |"
    )

    diff = (
        results["pretrained_mka_v2"]["best_val_ppl"]
        - results["pythia_pretrained"]["val_ppl"]
    )
    print_flush(f"\nDifference: {diff:+.1f} ppl")

    print_flush("\n" + "=" * 70)
    print_flush("POSITION-WISE PPL (Long-Range Dependency)")
    print_flush("=" * 70)

    pythia_pos = results["pythia_pretrained"]["position_wise_ppl"]
    mka_pos = results["pretrained_mka_v2"]["position_wise_ppl"]

    print_flush("\n| Position | Pythia | MKA | Diff |")
    print_flush("|----------|--------|-----|------|")
    for pos_range in pythia_pos:
        pythia_val = pythia_pos[pos_range]
        mka_val = mka_pos[pos_range]
        pos_diff = mka_val - pythia_val
        print_flush(f"| {pos_range} | {pythia_val:.1f} | {mka_val:.1f} | {pos_diff:+.1f} |")

    return results


def main() -> None:
    config = PythiaConfig()

    parser = argparse.ArgumentParser(description="Pretrained MKA-Attention Experiment")
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
        "--eval-only", action="store_true", help="Only evaluate pretrained baseline"
    )
    args = parser.parse_args()

    run_experiment(
        num_samples=args.samples,
        seq_length=args.seq_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_only=args.eval_only,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
