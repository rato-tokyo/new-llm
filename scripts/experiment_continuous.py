#!/usr/bin/env python3
"""
Continuous LM Experiment

仮説: トークン化による離散化で情報が失われている

モード:
- discrete: 通常のLM（トークン埋め込みを入力）
- continuous: 前の隠れ状態を直接入力として使用
- continuous_extra: continuous + 追加処理（h2のみ）
- continuous_combined: continuous + 追加処理（h1+h2）

Usage:
    # 全モード比較
    python3 scripts/experiment_continuous.py --models discrete continuous continuous_extra continuous_combined

    # discrete vs continuous のみ
    python3 scripts/experiment_continuous.py --models discrete continuous
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


# モード設定のマッピング
MODE_CONFIGS = {
    "discrete": {"mode": "discrete", "extra_pass": False, "use_h1": False},
    "continuous": {"mode": "continuous", "extra_pass": False, "use_h1": False},
    "continuous_extra": {"mode": "continuous", "extra_pass": True, "use_h1": False},
    "continuous_combined": {"mode": "continuous", "extra_pass": True, "use_h1": True},
}


class ModelWrapper:
    """評価用モデルラッパー"""
    def __init__(self, model, mode: str, extra_pass: bool, use_h1: bool):
        self.model = model
        self.mode = mode
        self.extra_pass = extra_pass
        self.use_h1 = use_h1

    def __call__(self, input_ids):
        logits, _ = self.model(
            input_ids, mode=self.mode, extra_pass=self.extra_pass, use_h1=self.use_h1
        )
        return logits

    def eval(self):
        self.model.eval()


def compute_ppl(model, data_loader, device, mode: str, extra_pass: bool, use_h1: bool) -> float:
    """共通PPL計算関数"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            loss, _ = model.compute_loss(
                input_ids, labels, mode=mode, extra_pass=extra_pass, use_h1=use_h1
            )

            batch_tokens = (labels.size(1) - 1) * labels.size(0)
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def train_continuous(
    model,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    patience: int,
    gradient_clip: float,
    mode: str,
    extra_pass: bool,
    use_h1: bool,
) -> tuple[float, int, Optional[dict]]:
    """ContinuousLM訓練"""
    best_val_ppl = float("inf")
    best_epoch = 0
    best_state_dict = None
    patience_counter = 0

    mode_name = mode
    if extra_pass:
        mode_name += "_extra"
    if use_h1:
        mode_name += "_combined"

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

            loss, stats = model.compute_loss(
                input_ids, labels, mode=mode, extra_pass=extra_pass, use_h1=use_h1
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            batch_tokens = (labels.size(1) - 1) * labels.size(0)
            epoch_loss += loss.item() * batch_tokens
            epoch_tokens += batch_tokens

        train_ppl = torch.exp(torch.tensor(epoch_loss / epoch_tokens)).item()

        # Validation
        val_ppl = compute_ppl(model, val_loader, device, mode, extra_pass, use_h1)
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
            f"mode={mode_name} ({elapsed:.1f}s) {marker}"
        )

        if patience_counter >= patience:
            print_flush("  Early stopping")
            break

    return best_val_ppl, best_epoch, best_state_dict


def run_model(
    model_name: str,
    config: PythiaConfig,
    train_loader,
    val_loader,
    device: torch.device,
    args,
) -> dict:
    """単一モデルの訓練と評価"""
    mode_config = MODE_CONFIGS[model_name]
    mode = mode_config["mode"]
    extra_pass = mode_config["extra_pass"]
    use_h1 = mode_config["use_h1"]

    print_flush("\n" + "=" * 70)
    print_flush(f"{model_name.upper()}")
    print_flush("=" * 70)

    model = create_model("continuous", config)
    model = model.to(device)

    param_info = model.num_parameters()
    print_flush(f"  Parameters: {param_info['total']:,}")
    if mode == "continuous":
        print_flush(f"    - hidden_proj: {param_info['hidden_proj']:,}")
    if extra_pass:
        print_flush(f"    - extra_proj: {param_info['extra_proj']:,}")
    if use_h1:
        print_flush(f"    - combine_proj: {param_info['combine_proj']:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print_flush(f"\n[{model_name}] Training...")
    best_ppl, best_epoch, best_state = train_continuous(
        model, train_loader, val_loader, optimizer, device,
        args.epochs, args.patience, GRADIENT_CLIP,
        mode=mode, extra_pass=extra_pass, use_h1=use_h1,
    )
    print_flush(f"  Best: epoch {best_epoch}, ppl={best_ppl:.1f}")

    if best_state:
        model.load_state_dict(best_state)

    # Evaluation
    print_flush("\n  Position-wise PPL:")
    model.eval()
    wrapper = ModelWrapper(model, mode, extra_pass, use_h1)
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
        "mode": model_name,
        "position_wise_ppl": pos_ppl,
        "reversal_curse": reversal,
    }

    del model
    clear_gpu_cache(device)

    return result


def main():
    parser = argparse.ArgumentParser(description="Continuous LM Experiment")
    parser.add_argument(
        "--models", nargs="+",
        default=["discrete", "continuous"],
        choices=list(MODE_CONFIGS.keys()),
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
    print_flush("CONTINUOUS LM EXPERIMENT")
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
