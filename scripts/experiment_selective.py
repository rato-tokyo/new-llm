#!/usr/bin/env python3
"""
Selective Output LM Experiment

仮説: LLMは即座に出力せず、隠れ状態を追加処理してから出力すべき

動作:
- baseline: トークン入力 → 即座に次トークン予測
- selective: トークン入力 → 1回追加処理 → 次トークン予測（h2のみ使用）
- combined: トークン入力 → 1回追加処理 → h1+h2の両方を使用して予測

Usage:
    # Combined only (default)
    python3 scripts/experiment_selective.py

    # Compare all models
    python3 scripts/experiment_selective.py --models baseline selective combined
"""

import argparse
import sys
import time
from typing import Optional

sys.path.insert(0, ".")

import torch

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
    """評価用モデルラッパー（modeを固定）"""
    def __init__(self, model, mode: str):
        self.model = model
        self.mode = mode

    def __call__(self, input_ids):
        logits, _ = self.model(input_ids, mode=self.mode)
        return logits

    def eval(self):
        self.model.eval()


def compute_ppl(model, data_loader, device, mode: str) -> float:
    """共通PPL計算関数"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss, _ = model.compute_loss(input_ids, labels, mode=mode)

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
    mode: str = "selective",
) -> tuple[float, int, Optional[dict]]:
    """
    SelectiveOutputLM訓練

    mode: "baseline", "selective", or "combined"
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

            loss, stats = model.compute_loss(input_ids, labels, mode=mode)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            batch_tokens = (labels.size(1) - 1) * labels.size(0)
            epoch_loss += loss.item() * batch_tokens
            epoch_tokens += batch_tokens

        train_ppl = torch.exp(torch.tensor(epoch_loss / epoch_tokens)).item()

        # Validation（共通関数を使用）
        val_ppl = compute_ppl(model, val_loader, device, mode)
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
            f"mode={mode} ({elapsed:.1f}s) {marker}"
        )

        if patience_counter >= patience:
            print_flush("  Early stopping")
            break

    return best_val_ppl, best_epoch, best_state_dict


def run_model(
    model_name: str,
    mode: str,
    config: PythiaConfig,
    train_loader,
    val_loader,
    device: torch.device,
    args,
) -> dict:
    """単一モデルの訓練と評価"""
    print_flush("\n" + "=" * 70)
    print_flush(f"{model_name.upper()} (mode={mode})")
    print_flush("=" * 70)

    model = create_model("selective", config)
    model = model.to(device)

    param_info = model.num_parameters()
    print_flush(f"  Parameters: {param_info['total']:,}")
    if mode in ["selective", "combined"]:
        print_flush(f"    - hidden_proj: {param_info['hidden_proj']:,}")
    if mode == "combined":
        print_flush(f"    - combine_proj: {param_info['combine_proj']:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print_flush(f"\n[{model_name}] Training...")
    best_ppl, best_epoch, best_state = train_selective(
        model, train_loader, val_loader, optimizer, device,
        args.epochs, args.patience, GRADIENT_CLIP,
        mode=mode,
    )
    print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

    if best_state:
        model.load_state_dict(best_state)

    # Evaluation
    print_flush(f"\n  Position-wise PPL (mode={mode}):")
    model.eval()
    wrapper = ModelWrapper(model, mode=mode)
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

    result = {
        "best_val_ppl": best_ppl,
        "best_epoch": best_epoch,
        "mode": mode,
        "position_wise_ppl": pos_ppl,
        "reversal_curse": reversal,
    }

    del model
    clear_gpu_cache(device)

    return result


def main():
    parser = argparse.ArgumentParser(description="Selective Output LM Experiment")
    parser.add_argument(
        "--models", nargs="+", default=["combined"],
        choices=["baseline", "selective", "combined"],
        help="Models to run"
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

    # Run each model
    for model_name in args.models:
        result = run_model(
            model_name=model_name,
            mode=model_name,  # mode名 = model名
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            args=args,
        )
        results[model_name] = result

    # =========================================================================
    # Summary
    # =========================================================================
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | Mode | Best PPL | Epoch |")
    print_flush("|-------|------|----------|-------|")
    for model_name, r in results.items():
        print_flush(f"| {model_name} | {r['mode']} | {r['best_val_ppl']:.1f} | {r['best_epoch']} |")

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
