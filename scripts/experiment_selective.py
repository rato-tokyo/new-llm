#!/usr/bin/env python3
"""
Selective Output LM Experiment

仮説: LLMは即座に出力せず、隠れ状態を追加処理してから出力すべき

動作:
- Baseline (extra_passes=0): トークン入力 → 即座に次トークン予測（追加処理なし）
- Selective (extra_passes=1): トークン入力 → 1回追加処理 → 次トークン予測

Usage:
    # Selective only (default)
    python3 scripts/experiment_selective.py

    # Compare with baseline
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


class ModelWrapper:
    """評価用モデルラッパー（use_selectiveを固定）"""
    def __init__(self, model, use_selective: bool):
        self.model = model
        self.use_selective = use_selective

    def __call__(self, input_ids):
        logits, _ = self.model(input_ids, use_selective=self.use_selective)
        return logits

    def eval(self):
        self.model.eval()


def compute_ppl(model, data_loader, device, use_selective: bool) -> float:
    """共通PPL計算関数"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss, _ = model.compute_loss(input_ids, labels, use_selective=use_selective)

            batch_tokens = (labels.size(1) - 1) * labels.size(0)
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


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

    use_selective=True: extra_passes=1（1回追加処理してから出力）
    use_selective=False: extra_passes=0（即座に出力、通常のContinuous）
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

        for batch in train_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            loss, stats = model.compute_loss(input_ids, labels, use_selective=use_selective)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            batch_tokens = (labels.size(1) - 1) * labels.size(0)
            epoch_loss += loss.item() * batch_tokens
            epoch_tokens += batch_tokens

        train_ppl = torch.exp(torch.tensor(epoch_loss / epoch_tokens)).item()

        # Validation（共通関数を使用）
        val_ppl = compute_ppl(model, val_loader, device, use_selective)
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

        extra_passes = 1 if use_selective else 0
        print_flush(
            f"  Epoch {epoch:2d}: train={train_ppl:7.1f}, val={val_ppl:7.1f}, "
            f"extra={extra_passes} ({elapsed:.1f}s) {marker}"
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
        help="Models to run (baseline=extra_passes=0, selective=extra_passes=1)"
    )
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--nope", action="store_true", help="Use NoPE (no position encoding)")

    args = parser.parse_args()

    set_seed(42)
    device = get_device()
    config = PythiaConfig()

    # NoPE: rotary_pct=0 でPosition Encodingを無効化
    if args.nope:
        config.rotary_pct = 0.0

    # Print experiment info
    print_flush("=" * 70)
    print_flush("SELECTIVE OUTPUT LM EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Models: {args.models}")
    print_flush(f"Samples: {args.samples:,}")
    print_flush(f"Sequence length: {args.seq_length}")
    print_flush(f"Epochs: {args.epochs}")
    print_flush(f"Position Encoding: {'NoPE' if args.nope else 'RoPE'}")
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
    # Baseline (extra_passes=0, equivalent to Continuous)
    # =========================================================================
    if "baseline" in args.models:
        print_flush("\n" + "=" * 70)
        print_flush("BASELINE (extra_passes=0, no extra processing)")
        print_flush("=" * 70)

        model = create_model("selective", config)
        model = model.to(device)

        param_info = model.num_parameters()
        print_flush(f"  Parameters: {param_info['total']:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        print_flush("\n[Baseline] Training...")
        best_ppl, best_epoch, best_state = train_selective(
            model, train_loader, val_loader, optimizer, device,
            args.epochs, args.patience, GRADIENT_CLIP,
            use_selective=False,  # extra_passes=0
        )
        print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

        if best_state:
            model.load_state_dict(best_state)

        # Evaluation
        print_flush("\n  Position-wise PPL:")
        model.eval()
        wrapper = ModelWrapper(model, use_selective=False)
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
            "extra_passes": 0,
            "position_wise_ppl": pos_ppl,
            "reversal_curse": reversal,
        }

        del model
        clear_gpu_cache(device)

    # =========================================================================
    # Selective Output LM (extra_passes=1)
    # =========================================================================
    if "selective" in args.models:
        print_flush("\n" + "=" * 70)
        print_flush("SELECTIVE (extra_passes=1, 1 extra processing)")
        print_flush("=" * 70)

        model = create_model("selective", config)
        model = model.to(device)

        param_info = model.num_parameters()
        print_flush(f"  Parameters: {param_info['total']:,}")
        print_flush(f"    - hidden_proj: {param_info['hidden_proj']:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        print_flush("\n[Selective] Training...")
        best_ppl, best_epoch, best_state = train_selective(
            model, train_loader, val_loader, optimizer, device,
            args.epochs, args.patience, GRADIENT_CLIP,
            use_selective=True,  # extra_passes=1
        )
        print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

        if best_state:
            model.load_state_dict(best_state)

        # Evaluation (using selective mode)
        print_flush("\n  Position-wise PPL (extra_passes=1):")
        model.eval()
        wrapper = ModelWrapper(model, use_selective=True)
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
            "extra_passes": 1,
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

    print_flush("\n| Model | Extra Passes | Best PPL | Epoch |")
    print_flush("|-------|--------------|----------|-------|")
    for model_name, r in results.items():
        print_flush(f"| {model_name} | {r['extra_passes']} | {r['best_val_ppl']:.1f} | {r['best_epoch']} |")

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
