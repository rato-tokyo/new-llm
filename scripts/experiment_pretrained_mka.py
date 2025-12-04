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
import random
import sys
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM

# Add project root to path
sys.path.insert(0, ".")

from config.pythia import PythiaConfig  # noqa: E402
from src.models.pretrained_mka import PretrainedMKAModelV2  # noqa: E402
from src.utils.io import print_flush  # noqa: E402
from src.utils.training import prepare_data_loaders, get_device  # noqa: E402
from src.utils.device import clear_gpu_cache  # noqa: E402


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model and return perplexity"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)

            # labels are already shifted (labels[i] = next token of input_ids[i])
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += labels.numel()

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl


def evaluate_position_wise_ppl(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    position_ranges: list[tuple[int, int]] | None = None,
) -> Dict[str, float]:
    """
    Evaluate position-wise perplexity.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device
        position_ranges: List of (start, end) tuples for position ranges.
                        If None, uses default ranges for seq_len=128.

    Returns:
        Dictionary with position range keys and PPL values
    """
    model.eval()

    # Get sequence length from first batch
    first_batch = next(iter(val_loader))
    seq_len = first_batch[0].shape[1]

    # Default position ranges
    if position_ranges is None:
        position_ranges = [
            (0, 16),
            (16, 32),
            (32, 64),
            (64, 96),
            (96, seq_len),
        ]

    # Initialize accumulators for each range
    range_losses: Dict[str, float] = {}
    range_tokens: Dict[str, int] = {}
    for start, end in position_ranges:
        key = f"{start}-{end}"
        range_losses[key] = 0.0
        range_tokens[key] = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)  # [batch, seq_len, vocab]

            # Compute per-position loss
            # logits: [batch, seq_len, vocab]
            # labels: [batch, seq_len]
            for start, end in position_ranges:
                key = f"{start}-{end}"

                # Get logits and labels for this position range
                range_logits = logits[:, start:end, :]  # [batch, range_len, vocab]
                range_labels = labels[:, start:end]  # [batch, range_len]

                # Compute loss for this range
                loss = nn.functional.cross_entropy(
                    range_logits.reshape(-1, range_logits.size(-1)),
                    range_labels.reshape(-1),
                    reduction="sum",
                )

                range_losses[key] += loss.item()
                range_tokens[key] += range_labels.numel()

    # Compute PPL for each range
    results: Dict[str, float] = {}
    for key in range_losses:
        if range_tokens[key] > 0:
            avg_loss = range_losses[key] / range_tokens[key]
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            results[key] = ppl
        else:
            results[key] = float("inf")

    return results


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in train_loader:
        input_ids, labels = batch
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)

        # labels are already shifted (labels[i] = next token of input_ids[i])
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    """
    Run pretrained MKA experiment

    Args:
        num_samples: Number of training samples
        seq_length: Sequence length
        num_epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        eval_only: Only evaluate pretrained baseline

    Returns:
        Results dict
    """
    # Set seed for reproducibility
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

    # ===== 1. Pretrained Pythia-70M (Baseline) =====
    print_flush("\n" + "=" * 70)
    print_flush("1. PYTHIA-70M (Pretrained, Frozen)")
    print_flush("=" * 70)

    pretrained_pythia = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    pretrained_pythia = pretrained_pythia.to(device)
    pretrained_pythia.eval()

    # Wrap for our evaluation function
    class PythiaWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, input_ids):
            return self.model(input_ids).logits

    pythia_wrapper = PythiaWrapper(pretrained_pythia)

    print_flush("\n[Pythia-70M] Evaluating pretrained model...")
    pythia_ppl = evaluate_model(pythia_wrapper, val_loader, device)
    print_flush(f"  Pretrained val_ppl: {pythia_ppl:.1f}")

    # Position-wise PPL
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

    # Only optimize trainable parameters
    trainable_params = [p for p in mka_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    print_flush("\n[PretrainedMKA V2] Training...")

    best_val_ppl = float("inf")
    best_epoch = 0
    patience_counter = 0
    patience = config.early_stopping_patience

    for epoch in range(1, num_epochs + 1):
        import time
        start_time = time.time()

        train_ppl = train_epoch(mka_model, train_loader, optimizer, device)
        val_ppl = evaluate_model(mka_model, val_loader, device)

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

        if patience_counter >= patience:
            print_flush(f"  -> Early stop: val_ppl worsened for {patience} epochs")
            break

    print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

    # Position-wise PPL for MKA model
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

    # Difference
    diff = (
        results["pretrained_mka_v2"]["best_val_ppl"]
        - results["pythia_pretrained"]["val_ppl"]
    )
    print_flush(f"\nDifference: {diff:+.1f} ppl")

    # Position-wise PPL comparison
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
