#!/usr/bin/env python3
"""
Selective Output LM Experiment (Simplified)

仮説: LLMの全出力がトークン化されるべきではない
- 固定パターンで持ち越し（例: 2回に1回）
- 持ち越し位置では損失なし、出力位置でのみ損失計算

実験:
1. SelectiveOutputLM を訓練（固定スキップパターン）
2. 通常のPPLで評価
3. Pythiaベースラインとの比較

Usage:
    # Selective only (default, skip_interval=2)
    python3 scripts/experiment_selective.py

    # With different skip interval
    python3 scripts/experiment_selective.py --skip-interval 3

    # With baseline comparison
    python3 scripts/experiment_selective.py --models pythia selective
"""

import argparse
import sys
import time
from typing import Optional

sys.path.insert(0, ".")

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.pythia import PythiaConfig
from src.config.experiment_defaults import GRADIENT_CLIP
from src.data.reversal_pairs import get_reversal_pairs
from src.models import create_model
from src.utils.device import clear_gpu_cache
from src.utils.evaluation import evaluate_position_wise_ppl, evaluate_reversal_curse
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.tokenizer_utils import get_tokenizer
from src.utils.training import get_device, prepare_data_loaders


def train_pythia(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int,
    gradient_clip: float = 1.0,
) -> tuple[float, int, Optional[dict]]:
    """Pythia標準訓練"""
    best_val_ppl = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            total_loss += loss.item()
            total_tokens += labels.numel()

        train_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()

        # Validation
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
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train={train_ppl:7.1f}, val={val_ppl:7.1f} "
            f"({elapsed:.1f}s) {marker}"
        )

        if patience_counter >= patience:
            print_flush("  Early stopping")
            break

    return best_val_ppl, best_epoch, best_state_dict


def train_selective(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int,
    gradient_clip: float = 1.0,
) -> tuple[float, int, Optional[dict]]:
    """
    SelectiveOutputLM訓練（固定スキップパターン）

    skip_interval回に1回のみ損失計算。
    残りは持ち越し（損失なし）。
    """
    best_val_ppl = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()

        epoch_lm_loss = 0.0
        epoch_output_count = 0.0
        epoch_carryover_ratio = 0.0
        batch_count = 0

        for batch in train_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward (returns logits, final_hidden)
            logits, _ = model(input_ids, use_selective=False)

            # 固定スキップパターンによる損失計算
            loss, stats = model.compute_fixed_skip_loss(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            epoch_lm_loss += stats["lm_loss"] * stats["output_count"]
            epoch_output_count += stats["output_count"]
            epoch_carryover_ratio += stats["carryover_ratio"]
            batch_count += 1

        # Epoch統計
        if epoch_output_count > 0:
            train_ppl = torch.exp(torch.tensor(epoch_lm_loss / epoch_output_count)).item()
        else:
            train_ppl = float("inf")
        avg_carryover_ratio = epoch_carryover_ratio / batch_count

        # Validation
        model.eval()
        val_lm_loss = 0.0
        val_output_count = 0.0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                logits, _ = model(input_ids, use_selective=False)
                _, stats = model.compute_fixed_skip_loss(logits, labels)

                val_lm_loss += stats["lm_loss"] * stats["output_count"]
                val_output_count += stats["output_count"]

        if val_output_count > 0:
            val_ppl = torch.exp(torch.tensor(val_lm_loss / val_output_count)).item()
        else:
            val_ppl = float("inf")
        elapsed = time.time() - start_time

        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            best_epoch = epoch
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train={train_ppl:7.1f}, val={val_ppl:7.1f}, "
            f"skip={avg_carryover_ratio:.1%} ({elapsed:.1f}s) {marker}"
        )

        if patience_counter >= patience:
            print_flush("  Early stopping")
            break

    return best_val_ppl, best_epoch, best_state_dict


def main():
    parser = argparse.ArgumentParser(description="Selective Output LM Experiment")
    parser.add_argument(
        "--models", nargs="+", default=["selective"],
        choices=["pythia", "selective"],
        help="Models to run (default: selective only)"
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--skip-interval", type=int, default=2,
                        help="Output every N positions (default: 2 = 50% skip)")

    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    config = PythiaConfig()

    # Print experiment info
    print_flush("=" * 70)
    print_flush("SELECTIVE OUTPUT LM EXPERIMENT (fixed skip pattern)")
    print_flush("=" * 70)
    print_flush(f"Models: {args.models}")
    print_flush(f"Samples: {args.samples:,}")
    print_flush(f"Sequence length: {args.seq_length}")
    print_flush(f"Epochs: {args.epochs}")
    print_flush(f"Skip interval: {args.skip_interval} (output every {args.skip_interval} positions)")
    print_flush("=" * 70)

    # Load data
    print_flush("\n[Data] Loading Pile data...")
    train_loader, val_loader = prepare_data_loaders(
        num_samples=args.samples,
        seq_length=args.seq_length,
        tokenizer_name=config.tokenizer_name,
        val_split=0.1,
        batch_size=args.batch_size,
    )

    results = {}

    # =========================================================================
    # Pythia (baseline)
    # =========================================================================
    if "pythia" in args.models:
        print_flush("\n" + "=" * 70)
        print_flush("PYTHIA (baseline)")
        print_flush("=" * 70)

        model = create_model("pythia", config)
        model = model.to(device)

        param_info = model.num_parameters()
        print_flush(f"  Parameters: {param_info['total']:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        print_flush("\n[Pythia] Training...")
        best_ppl, best_epoch, best_state = train_pythia(
            model, train_loader, val_loader, optimizer, device,
            args.epochs, args.patience, GRADIENT_CLIP,
        )
        print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

        if best_state:
            model.load_state_dict(best_state)

        # Evaluation
        print_flush("\n  Position-wise PPL:")
        model.eval()
        pos_ppl = evaluate_position_wise_ppl(model, val_loader, device)
        for pos_range, ppl in pos_ppl.items():
            print_flush(f"    {pos_range}: {ppl:.1f}")

        print_flush("\n  Reversal Curse:")
        tokenizer = get_tokenizer(config.tokenizer_name)
        reversal_pairs = get_reversal_pairs()
        reversal = evaluate_reversal_curse(model, tokenizer, reversal_pairs, device)
        print_flush(f"    Forward PPL: {reversal['forward_ppl']:.1f}")
        print_flush(f"    Backward PPL: {reversal['backward_ppl']:.1f}")
        print_flush(f"    Gap: {reversal['reversal_gap']:+.1f}")

        results["pythia"] = {
            "best_val_ppl": best_ppl,
            "best_epoch": best_epoch,
            "position_wise_ppl": pos_ppl,
            "reversal_curse": reversal,
        }

        del model
        clear_gpu_cache(device)

    # =========================================================================
    # Selective Output LM
    # =========================================================================
    if "selective" in args.models:
        print_flush("\n" + "=" * 70)
        print_flush(f"SELECTIVE OUTPUT LM (skip_interval={args.skip_interval})")
        print_flush("=" * 70)

        model = create_model("selective", config, skip_interval=args.skip_interval)
        model = model.to(device)

        param_info = model.num_parameters()
        print_flush(f"  Parameters: {param_info['total']:,}")
        print_flush(f"    - hidden_proj: {param_info['hidden_proj']:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        print_flush("\n[Selective] Training (fixed skip pattern)...")
        best_ppl, best_epoch, best_state = train_selective(
            model, train_loader, val_loader, optimizer, device,
            args.epochs, args.patience, GRADIENT_CLIP,
        )
        print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

        if best_state:
            model.load_state_dict(best_state)

        # Standard evaluation (without selective mode)
        print_flush("\n  Position-wise PPL (standard mode):")
        model.eval()

        # Need wrapper for standard evaluation
        class SelectiveWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids):
                logits, _ = self.model(input_ids, use_selective=False)
                return logits

        wrapper = SelectiveWrapper(model)
        pos_ppl = evaluate_position_wise_ppl(wrapper, val_loader, device)
        for pos_range, ppl in pos_ppl.items():
            print_flush(f"    {pos_range}: {ppl:.1f}")

        print_flush("\n  Reversal Curse:")
        tokenizer = get_tokenizer(config.tokenizer_name)
        reversal_pairs = get_reversal_pairs()
        reversal = evaluate_reversal_curse(wrapper, tokenizer, reversal_pairs, device)
        print_flush(f"    Forward PPL: {reversal['forward_ppl']:.1f}")
        print_flush(f"    Backward PPL: {reversal['backward_ppl']:.1f}")
        print_flush(f"    Gap: {reversal['reversal_gap']:+.1f}")

        results["selective"] = {
            "best_val_ppl": best_ppl,
            "best_epoch": best_epoch,
            "skip_interval": args.skip_interval,
            "position_wise_ppl": pos_ppl,
            "reversal_curse": reversal,
        }

        del model, wrapper
        clear_gpu_cache(device)

    # =========================================================================
    # Summary
    # =========================================================================
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | Best PPL | Epoch |")
    print_flush("|-------|----------|-------|")
    for model_name, r in results.items():
        print_flush(f"| {model_name} | {r['best_val_ppl']:.1f} | {r['best_epoch']} |")

    print_flush("\n| Model | Forward PPL | Backward PPL | Gap |")
    print_flush("|-------|-------------|--------------|-----|")
    for model_name, r in results.items():
        rev = r["reversal_curse"]
        print_flush(
            f"| {model_name} | {rev['forward_ppl']:.1f} | "
            f"{rev['backward_ppl']:.1f} | {rev['reversal_gap']:+.1f} |"
        )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
