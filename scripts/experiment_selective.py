#!/usr/bin/env python3
"""
Selective Output LM Experiment

仮説: LLMは即座に出力せず、隠れ状態を追加処理してから出力すべき

動作:
- 従来のContinuous: トークン入力 → 即座に次トークン予測
- Selective: トークン入力 → 隠れ状態を追加処理 → 次トークン予測

skip_interval (追加処理回数):
- 0: 追加処理なし = 従来のContinuousと同等
- 1: 1回追加処理後に出力（デフォルト）
- 2: 2回追加処理後に出力

Usage:
    # Selective only (default, skip_interval=1)
    python3 scripts/experiment_selective.py

    # With different skip interval
    python3 scripts/experiment_selective.py --skip-interval 2

    # Compare with baseline (skip_interval=0)
    python3 scripts/experiment_selective.py --models baseline selective
"""

import argparse
import sys
import time
from typing import Optional

sys.path.insert(0, ".")

import torch
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


def train_selective(
    model,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int,
    gradient_clip: float = 1.0,
    use_selective: bool = True,
) -> tuple[float, int, Optional[dict]]:
    """
    SelectiveOutputLM訓練

    use_selective=True: skip_interval回の追加処理後に出力
    use_selective=False: 通常のContinuous（即座に出力）
    """
    best_val_ppl = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        model.train()

        epoch_loss = 0.0
        epoch_tokens = 0
        epoch_steps = 0

        for batch in train_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # compute_loss uses forward_selective internally
            loss, stats = model.compute_loss(input_ids, labels, use_selective=use_selective)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            batch_tokens = (labels.size(1) - 1) * labels.size(0)
            epoch_loss += loss.item() * batch_tokens
            epoch_tokens += batch_tokens
            epoch_steps += stats.get("total_process_steps", batch_tokens)

        train_ppl = torch.exp(torch.tensor(epoch_loss / epoch_tokens)).item()
        avg_steps = epoch_steps / epoch_tokens

        # Validation
        model.eval()
        val_loss = 0.0
        val_tokens = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                loss, _ = model.compute_loss(input_ids, labels, use_selective=use_selective)

                batch_tokens = (labels.size(1) - 1) * labels.size(0)
                val_loss += loss.item() * batch_tokens
                val_tokens += batch_tokens

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
            f"  Epoch {epoch:2d}: train={train_ppl:7.1f}, val={val_ppl:7.1f}, "
            f"steps/tok={avg_steps:.1f} ({elapsed:.1f}s) {marker}"
        )

        if patience_counter >= patience:
            print_flush("  Early stopping")
            break

    return best_val_ppl, best_epoch, best_state_dict


def main():
    parser = argparse.ArgumentParser(description="Selective Output LM Experiment")
    parser.add_argument(
        "--models", nargs="+", default=["selective"],
        choices=["baseline", "selective"],
        help="Models to run (baseline=skip_interval=0, selective=skip_interval from --skip-interval)"
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--skip-interval", type=int, default=1,
                        help="Number of extra processing steps (0=Continuous, 1=1 extra step)")

    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    config = PythiaConfig()

    # Print experiment info
    print_flush("=" * 70)
    print_flush("SELECTIVE OUTPUT LM EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Models: {args.models}")
    print_flush(f"Samples: {args.samples:,}")
    print_flush(f"Sequence length: {args.seq_length}")
    print_flush(f"Epochs: {args.epochs}")
    print_flush(f"Skip interval: {args.skip_interval} ({args.skip_interval} extra processing steps)")
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
    # Baseline (skip_interval=0, equivalent to Continuous)
    # =========================================================================
    if "baseline" in args.models:
        print_flush("\n" + "=" * 70)
        print_flush("BASELINE (skip_interval=0, no extra processing)")
        print_flush("=" * 70)

        model = create_model("selective", config, skip_interval=0)
        model = model.to(device)

        param_info = model.num_parameters()
        print_flush(f"  Parameters: {param_info['total']:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        print_flush("\n[Baseline] Training...")
        best_ppl, best_epoch, best_state = train_selective(
            model, train_loader, val_loader, optimizer, device,
            args.epochs, args.patience, GRADIENT_CLIP,
            use_selective=False,  # No extra processing
        )
        print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

        if best_state:
            model.load_state_dict(best_state)

        # Evaluation
        print_flush("\n  Position-wise PPL:")
        model.eval()

        class BaselineWrapper:
            def __init__(self, model):
                self.model = model

            def __call__(self, input_ids):
                logits, _ = self.model(input_ids, use_selective=False)
                return logits

        wrapper = BaselineWrapper(model)
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

        results["baseline"] = {
            "best_val_ppl": best_ppl,
            "best_epoch": best_epoch,
            "skip_interval": 0,
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
        print_flush(f"SELECTIVE (skip_interval={args.skip_interval}, {args.skip_interval} extra steps)")
        print_flush("=" * 70)

        model = create_model("selective", config, skip_interval=args.skip_interval)
        model = model.to(device)

        param_info = model.num_parameters()
        print_flush(f"  Parameters: {param_info['total']:,}")
        print_flush(f"    - hidden_proj: {param_info['hidden_proj']:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        print_flush("\n[Selective] Training...")
        best_ppl, best_epoch, best_state = train_selective(
            model, train_loader, val_loader, optimizer, device,
            args.epochs, args.patience, GRADIENT_CLIP,
            use_selective=True,  # Use extra processing
        )
        print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

        if best_state:
            model.load_state_dict(best_state)

        # Evaluation (using selective mode)
        print_flush("\n  Position-wise PPL (selective mode):")
        model.eval()

        class SelectiveWrapper:
            def __init__(self, model):
                self.model = model

            def __call__(self, input_ids):
                logits, _ = self.model(input_ids, use_selective=True)
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

        del model
        clear_gpu_cache(device)

    # =========================================================================
    # Summary
    # =========================================================================
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | Skip Interval | Best PPL | Epoch |")
    print_flush("|-------|---------------|----------|-------|")
    for model_name, r in results.items():
        print_flush(f"| {model_name} | {r['skip_interval']} | {r['best_val_ppl']:.1f} | {r['best_epoch']} |")

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
