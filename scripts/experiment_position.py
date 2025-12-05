#!/usr/bin/env python3
"""
Position Encoding Experiment

位置エンコーディングの比較実験。
RoPE (2D), RoPE3D (3D), ALiBi, NoPE, Learnable を統一モデルで比較。

Usage:
    # 全種類比較
    python3 scripts/experiment_position.py --samples 10000 --epochs 30

    # 特定の位置エンコーディングのみ
    python3 scripts/experiment_position.py --pos-types rope alibi
    python3 scripts/experiment_position.py --pos-types none

    # RoPE vs RoPE3D 比較
    python3 scripts/experiment_position.py --pos-types rope rope3d

    # RoPE3D のみ（RoPEをスキップ）
    python3 scripts/experiment_position.py --pos-types rope rope3d --skip rope

    # ALiBi slope変更
    python3 scripts/experiment_position.py --pos-types alibi --alibi-slope 0.125

    # Learnable position encoding
    python3 scripts/experiment_position.py --pos-types learnable --learnable-nonlinear gelu
    python3 scripts/experiment_position.py --pos-types rope learnable --learnable-dim 32
"""

import argparse
import sys
import time
from typing import Any, List

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, ".")

from config.pythia import PythiaConfig  # noqa: E402
from src.config.experiment_defaults import EARLY_STOPPING_PATIENCE  # noqa: E402
from src.models.unified_pythia import UnifiedPythiaModel  # noqa: E402
from src.models.position_encoding import PositionEncodingConfig  # noqa: E402
from src.data.reversal_pairs import get_reversal_pairs  # noqa: E402
from src.utils.io import print_flush  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402
from src.utils.training import (  # noqa: E402
    prepare_data_loaders,
    get_device,
    get_tokenizer,
    train_epoch,
    evaluate,
)
from src.utils.evaluation import (  # noqa: E402
    evaluate_position_wise_ppl,
    evaluate_reversal_curse,
)
from src.utils.device import clear_gpu_cache  # noqa: E402


# Position encoding display names
POS_ENCODING_NAMES = {
    "rope": "RoPE (2D)",
    "rope3d": "RoPE3D (3D)",
    "alibi": "ALiBi",
    "none": "NoPE (None)",
    "learnable": "Learnable (K only)",
}


def train_model(
    pos_type: str,
    config: PythiaConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    lr: float,
    rotary_pct: float,
    rope3d_pct: float,
    alibi_slope: float,
    learnable_dim: int | None = None,
    learnable_nonlinear: str = "gelu",
) -> dict[str, Any]:
    """Train a model with specified position encoding."""
    pos_name = POS_ENCODING_NAMES.get(pos_type, pos_type)
    print_flush(f"\n  Position encoding: {pos_name}")

    # Create position encoding config
    pos_config = PositionEncodingConfig(
        type=pos_type,
        rotary_pct=rotary_pct,
        rope3d_pct=rope3d_pct,
        alibi_slope=alibi_slope,
        learnable_dim=learnable_dim,
        learnable_nonlinear=learnable_nonlinear,
        max_position_embeddings=config.max_position_embeddings,
    )

    # Create model
    model = UnifiedPythiaModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        pos_encoding=pos_config,
    )
    model = model.to(device)

    param_info = model.num_parameters()
    print_flush(f"  Total parameters: {param_info['total']:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print_flush(f"\n  Training {pos_name}...")
    best_val_ppl = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_ppl = torch.exp(torch.tensor(train_loss)).item()
        _, val_ppl = evaluate(model, val_loader, device)
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

    print_flush("\n  Position-wise PPL:")
    pos_ppl = evaluate_position_wise_ppl(model, val_loader, device)
    for pos_range, ppl in pos_ppl.items():
        print_flush(f"    Position {pos_range}: {ppl:.1f}")

    # Reversal Curse evaluation
    print_flush("\n  Reversal Curse evaluation:")
    tokenizer = get_tokenizer(config.tokenizer_name)
    reversal_pairs = get_reversal_pairs()
    reversal = evaluate_reversal_curse(model, tokenizer, reversal_pairs, device)
    print_flush(f"    Forward PPL: {reversal['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {reversal['backward_ppl']:.1f}")
    print_flush(f"    Reversal Ratio: {reversal['reversal_ratio']:.3f}")
    print_flush(f"    Reversal Gap: {reversal['reversal_gap']:+.1f}")

    result = {
        "pos_type": pos_type,
        "pos_name": pos_name,
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
        "position_wise_ppl": pos_ppl,
        "reversal_curse": reversal,
        "params": param_info["total"],
    }

    del model
    clear_gpu_cache(device)

    return result


def run_experiment(
    num_samples: int = 10000,
    seq_length: int = 128,
    num_epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    pos_types: List[str] = ["rope", "alibi", "none"],
    rotary_pct: float = 0.25,
    rope3d_pct: float = 0.25,
    alibi_slope: float = 0.0625,
    learnable_dim: int | None = None,
    learnable_nonlinear: str = "gelu",
) -> dict[str, Any]:
    """Run position encoding comparison experiment."""
    set_seed(42)

    device = get_device()
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("POSITION ENCODING EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples:,}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Epochs: {num_epochs}")
    print_flush(f"Learning rate: {lr}")
    print_flush(f"Position types: {pos_types}")
    print_flush(f"RoPE rotary_pct: {rotary_pct}")
    if "rope3d" in pos_types:
        print_flush(f"RoPE3D rotary_pct: {rope3d_pct}")
    print_flush(f"ALiBi slope: {alibi_slope}")
    if "learnable" in pos_types:
        print_flush(f"Learnable dim: {learnable_dim or 'full head_dim'}")
        print_flush(f"Learnable nonlinear: {learnable_nonlinear}")
    print_flush("=" * 70)

    print_flush("\n[Data] Loading Pile data...")
    train_loader, val_loader = prepare_data_loaders(
        num_samples=num_samples,
        seq_length=seq_length,
        tokenizer_name=config.tokenizer_name,
        val_split=0.1,
        batch_size=batch_size,
    )

    results: dict[str, Any] = {}

    for i, pos_type in enumerate(pos_types, 1):
        pos_name = POS_ENCODING_NAMES.get(pos_type, pos_type)
        print_flush("\n" + "=" * 70)
        print_flush(f"{i}. {pos_name.upper()}")
        print_flush("=" * 70)

        result = train_model(
            pos_type=pos_type,
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=num_epochs,
            lr=lr,
            rotary_pct=rotary_pct,
            rope3d_pct=rope3d_pct,
            alibi_slope=alibi_slope,
            learnable_dim=learnable_dim,
            learnable_nonlinear=learnable_nonlinear,
        )
        results[pos_type] = result

    # ===== Summary =====
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Position Encoding | PPL | Epoch | Params |")
    print_flush("|-------------------|-----|-------|--------|")

    for pos_type in pos_types:
        r = results[pos_type]
        print_flush(
            f"| {r['pos_name']} | {r['best_val_ppl']:.1f} | "
            f"{r['best_epoch']} | {r['params']:,} |"
        )

    # Position-wise comparison
    if len(pos_types) > 1:
        print_flush("\n" + "=" * 70)
        print_flush("POSITION-WISE PPL COMPARISON")
        print_flush("=" * 70)

        # Header
        header = "| Position |"
        for pos_type in pos_types:
            header += f" {POS_ENCODING_NAMES.get(pos_type, pos_type)} |"
        print_flush(f"\n{header}")

        sep = "|----------|"
        for _ in pos_types:
            sep += "------|"
        print_flush(sep)

        # Data rows
        first_pos = results[pos_types[0]]["position_wise_ppl"]
        for pos_range in first_pos:
            row = f"| {pos_range} |"
            for pos_type in pos_types:
                val = results[pos_type]["position_wise_ppl"][pos_range]
                row += f" {val:.1f} |"
            print_flush(row)

        # Reversal Curse comparison
        print_flush("\n" + "=" * 70)
        print_flush("REVERSAL CURSE COMPARISON")
        print_flush("=" * 70)

        print_flush("\n| Model | Forward PPL | Backward PPL | Ratio | Gap |")
        print_flush("|-------|-------------|--------------|-------|-----|")

        for pos_type in pos_types:
            r = results[pos_type]
            rev = r["reversal_curse"]
            print_flush(
                f"| {r['pos_name']} | {rev['forward_ppl']:.1f} | "
                f"{rev['backward_ppl']:.1f} | "
                f"{rev['reversal_ratio']:.3f} | "
                f"{rev['reversal_gap']:+.1f} |"
            )

        print_flush("\n(Reversal Ratio closer to 1.0 = less reversal curse)")

    # PPL difference analysis
    if "rope" in results and len(pos_types) > 1:
        rope_ppl = results["rope"]["best_val_ppl"]
        print_flush("\n" + "=" * 70)
        print_flush("PPL DIFFERENCE FROM RoPE BASELINE")
        print_flush("=" * 70)

        for pos_type in pos_types:
            if pos_type == "rope":
                continue
            r = results[pos_type]
            diff = r["best_val_ppl"] - rope_ppl
            diff_pct = (diff / rope_ppl) * 100
            print_flush(f"{r['pos_name']}: {diff:+.1f} ({diff_pct:+.1f}%)")

    return results


def main() -> None:
    config = PythiaConfig()

    parser = argparse.ArgumentParser(description="Position Encoding Experiment")
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
        "--pos-types",
        nargs="+",
        default=["rope", "alibi", "none"],
        choices=["rope", "rope3d", "alibi", "none", "learnable"],
        help="Position encoding types to compare",
    )
    parser.add_argument(
        "--rotary-pct", type=float, default=0.25, help="RoPE rotary percentage"
    )
    parser.add_argument(
        "--rope3d-pct", type=float, default=0.25, help="RoPE3D rotary percentage"
    )
    parser.add_argument(
        "--alibi-slope", type=float, default=0.0625, help="ALiBi slope (uniform)"
    )
    parser.add_argument(
        "--learnable-dim",
        type=int,
        default=None,
        help="Learnable position encoding dimension (default: full head_dim)",
    )
    parser.add_argument(
        "--learnable-nonlinear",
        type=str,
        default="gelu",
        choices=["gelu", "relu", "tanh", "none"],
        help="Learnable position encoding nonlinearity",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=["rope", "rope3d", "alibi", "none", "learnable"],
        help="Position encoding types to skip (bypass)",
    )
    args = parser.parse_args()

    # Remove skipped types from pos_types
    pos_types = [pt for pt in args.pos_types if pt not in args.skip]
    if not pos_types:
        print("Error: All position types were skipped. Nothing to run.")
        return

    run_experiment(
        num_samples=args.samples,
        seq_length=args.seq_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pos_types=pos_types,
        rotary_pct=args.rotary_pct,
        rope3d_pct=args.rope3d_pct,
        alibi_slope=args.alibi_slope,
        learnable_dim=args.learnable_dim,
        learnable_nonlinear=args.learnable_nonlinear,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
