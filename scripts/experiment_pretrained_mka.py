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
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPTNeoXForCausalLM

# Add project root to path
sys.path.insert(0, ".")

from config.pythia import PythiaConfig
from src.models.pretrained_mka import PretrainedMKAModelV2
from src.utils.io import print_flush
from src.utils.training import prepare_data_loaders, get_device
from src.utils.device import clear_gpu_cache


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
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids)

            # Shift for next token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return ppl


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
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids)

        # Shift for next token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * shift_labels.numel()
        total_tokens += shift_labels.numel()

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
    results["pythia_pretrained"] = {"val_ppl": pythia_ppl}

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
    results["pretrained_mka_v2"] = {
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
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
