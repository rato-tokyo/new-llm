#!/usr/bin/env python3
"""
Infini-Attention Experiment

1層目: Infini-Attention (NoPE, compressive memory)
2層目以降: MLA with ALiBi

Pythia (RoPE) vs Infini-Pythia (1層目Infini + MLA ALiBi) の比較。

Usage:
    python3 scripts/experiment_infini.py --samples 5000 --epochs 30
    python3 scripts/experiment_infini.py --samples 5000 --skip-baseline
    python3 scripts/experiment_infini.py --seq-length 512  # longer context
"""

import argparse
import sys
import time
from typing import Any

import torch

# Add project root to path
sys.path.insert(0, ".")

from config.pythia import PythiaConfig  # noqa: E402
from src.config.experiment_defaults import EARLY_STOPPING_PATIENCE  # noqa: E402
from src.models.pythia import PythiaModel  # noqa: E402
from src.models.infini_pythia import InfiniPythiaModel  # noqa: E402
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


def train_infini_model(
    model: InfiniPythiaModel,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int,
    reset_memory_per_epoch: bool = True,
) -> tuple[float, int]:
    """
    Infini-Pythiaモデルを訓練

    Args:
        model: InfiniPythiaModel
        train_loader: 訓練データローダー
        val_loader: 検証データローダー
        optimizer: オプティマイザ
        device: デバイス
        num_epochs: エポック数
        reset_memory_per_epoch: エポックごとにメモリをリセットするか

    Returns:
        best_val_ppl, best_epoch
    """
    best_val_ppl = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Reset memory at start of each epoch (optional)
        if reset_memory_per_epoch:
            model.reset_memory()

        # Train
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward with memory update
            logits = model(input_ids, update_memory=True)

            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_tokens += labels.numel()

        train_loss = total_loss / total_tokens
        train_ppl = torch.exp(torch.tensor(train_loss)).item()

        # Evaluate (without memory update for consistency)
        model.eval()
        eval_loss = 0.0
        eval_tokens = 0

        # Reset memory for evaluation
        model.reset_memory()

        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)

                logits = model(input_ids, update_memory=False)

                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    reduction="sum",
                )

                eval_loss += loss.item()
                eval_tokens += labels.numel()

        val_ppl = torch.exp(torch.tensor(eval_loss / eval_tokens)).item()
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

        # Get gate values
        gate_values = model.get_infini_gate_values()
        gate_mean = gate_values.mean().item()

        print_flush(
            f"  Epoch {epoch:2d}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
            f"gate={gate_mean:.3f} [{elapsed:.1f}s] {marker}"
        )

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print_flush("  -> Early stop")
            break

    return best_val_ppl, best_epoch


def run_experiment(
    num_samples: int = 5000,
    seq_length: int = 256,
    num_epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-4,
    kv_dim: int = 128,
    alibi_slope: float = 0.0625,
    use_delta_rule: bool = True,
    skip_baseline: bool = False,
) -> dict[str, Any]:
    """Run Infini-Attention experiment."""
    set_seed(42)

    device = get_device()
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("INFINI-ATTENTION EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples:,}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Epochs: {num_epochs}")
    print_flush(f"Learning rate: {lr}")
    print_flush(f"KV dim (Infini): {kv_dim}")
    print_flush(f"ALiBi slope: {alibi_slope}")
    print_flush(f"Delta rule: {use_delta_rule}")
    print_flush(f"Skip baseline: {skip_baseline}")
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

    # ===== 1. Pythia Baseline (RoPE) =====
    if skip_baseline:
        print_flush("\n" + "=" * 70)
        print_flush("1. PYTHIA (RoPE) - SKIPPED")
        print_flush("=" * 70)
        results["pythia"] = None
    else:
        print_flush("\n" + "=" * 70)
        print_flush("1. PYTHIA (RoPE baseline)")
        print_flush("=" * 70)

        pythia_model = PythiaModel(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
            rotary_pct=config.rotary_pct,
        )
        pythia_model = pythia_model.to(device)

        param_info = pythia_model.num_parameters()
        print_flush(f"  Total parameters: {param_info['total']:,}")

        optimizer = torch.optim.AdamW(pythia_model.parameters(), lr=lr)

        print_flush("\n[Pythia] Training...")
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
                f"  Epoch {epoch:2d}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
                f"[{elapsed:.1f}s] {marker}"
            )

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print_flush("  -> Early stop")
                break

        print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

        print_flush("\n  Position-wise PPL:")
        pythia_pos_ppl = evaluate_position_wise_ppl(pythia_model, val_loader, device)
        for pos_range, ppl in pythia_pos_ppl.items():
            print_flush(f"    Position {pos_range}: {ppl:.1f}")

        # Reversal Curse evaluation
        print_flush("\n  Reversal Curse evaluation:")
        tokenizer = get_tokenizer(config.tokenizer_name)
        reversal_pairs = get_reversal_pairs()
        pythia_reversal = evaluate_reversal_curse(
            pythia_model, tokenizer, reversal_pairs, device
        )
        print_flush(f"    Forward PPL: {pythia_reversal['forward_ppl']:.1f}")
        print_flush(f"    Backward PPL: {pythia_reversal['backward_ppl']:.1f}")
        print_flush(f"    Reversal Ratio: {pythia_reversal['reversal_ratio']:.3f}")
        print_flush(f"    Reversal Gap: {pythia_reversal['reversal_gap']:+.1f}")

        results["pythia"] = {
            "best_val_ppl": best_val_ppl,
            "best_epoch": best_epoch,
            "position_wise_ppl": pythia_pos_ppl,
            "reversal_curse": pythia_reversal,
        }

        del pythia_model
        clear_gpu_cache(device)

    # ===== 2. Infini-Pythia (1層目Infini + ALiBi) =====
    print_flush("\n" + "=" * 70)
    print_flush(f"2. INFINI-PYTHIA (1層目Infini + MLA ALiBi)")
    print_flush("=" * 70)
    print_flush("  Layer 0: Infini-Attention (NoPE, compressive memory)")
    print_flush("  Layer 1-5: MLA with ALiBi")

    infini_model = InfiniPythiaModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        kv_dim=kv_dim,
        q_compressed=False,
        alibi_slope=alibi_slope,
        use_delta_rule=use_delta_rule,
    )
    infini_model = infini_model.to(device)

    param_info = infini_model.num_parameters()
    print_flush(f"\n  Total parameters: {param_info['total']:,}")
    print_flush(f"  Infini layer: {param_info['infini_layer']:,}")
    print_flush(f"  MLA layers: {param_info['mla_layers']:,}")

    cache_info = infini_model.kv_cache_info(seq_length)
    print_flush(f"  KV Cache reduction: {cache_info['reduction_percent']:.1f}%")
    print_flush(f"  Infini memory: {cache_info['infini_memory_bytes']:,} bytes (fixed)")

    optimizer = torch.optim.AdamW(infini_model.parameters(), lr=lr)

    print_flush("\n[Infini] Training...")
    best_val_ppl, best_epoch = train_infini_model(
        infini_model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        reset_memory_per_epoch=True,
    )

    print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

    # Final gate values
    gate_values = infini_model.get_infini_gate_values()
    print_flush(f"\n  Final gate values (per head):")
    for i, g in enumerate(gate_values):
        print_flush(f"    Head {i}: {g.item():.3f}")

    print_flush("\n  Position-wise PPL:")
    infini_model.reset_memory()
    infini_pos_ppl = evaluate_position_wise_ppl(
        infini_model, val_loader, device, return_recon_loss=False
    )
    for pos_range, ppl in infini_pos_ppl.items():
        print_flush(f"    Position {pos_range}: {ppl:.1f}")

    # Reversal Curse evaluation
    print_flush("\n  Reversal Curse evaluation:")
    tokenizer = get_tokenizer(config.tokenizer_name)
    reversal_pairs = get_reversal_pairs()
    infini_model.reset_memory()
    infini_reversal = evaluate_reversal_curse(
        infini_model, tokenizer, reversal_pairs, device
    )
    print_flush(f"    Forward PPL: {infini_reversal['forward_ppl']:.1f}")
    print_flush(f"    Backward PPL: {infini_reversal['backward_ppl']:.1f}")
    print_flush(f"    Reversal Ratio: {infini_reversal['reversal_ratio']:.3f}")
    print_flush(f"    Reversal Gap: {infini_reversal['reversal_gap']:+.1f}")

    results["infini"] = {
        "best_val_ppl": best_val_ppl,
        "best_epoch": best_epoch,
        "position_wise_ppl": infini_pos_ppl,
        "reversal_curse": infini_reversal,
        "final_gate_values": gate_values.tolist(),
    }

    del infini_model
    clear_gpu_cache(device)

    # ===== Summary =====
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | PPL | Epoch |")
    print_flush("|-------|-----|-------|")

    if results["pythia"] is not None:
        print_flush(
            f"| Pythia (RoPE) | {results['pythia']['best_val_ppl']:.1f} | "
            f"{results['pythia']['best_epoch']} |"
        )

    print_flush(
        f"| Infini-Pythia | {results['infini']['best_val_ppl']:.1f} | "
        f"{results['infini']['best_epoch']} |"
    )

    if results["pythia"] is not None:
        diff = results["infini"]["best_val_ppl"] - results["pythia"]["best_val_ppl"]
        print_flush(f"\nDifference: {diff:+.1f} ppl")

        print_flush("\n" + "=" * 70)
        print_flush("POSITION-WISE PPL")
        print_flush("=" * 70)

        pythia_pos = results["pythia"]["position_wise_ppl"]
        infini_pos = results["infini"]["position_wise_ppl"]

        print_flush("\n| Position | Pythia | Infini | Diff |")
        print_flush("|----------|--------|--------|------|")
        for pos_range in pythia_pos:
            pythia_val = pythia_pos[pos_range]
            infini_val = infini_pos[pos_range]
            pos_diff = infini_val - pythia_val
            print_flush(
                f"| {pos_range} | {pythia_val:.1f} | {infini_val:.1f} | {pos_diff:+.1f} |"
            )

        # Reversal Curse comparison
        print_flush("\n" + "=" * 70)
        print_flush("REVERSAL CURSE")
        print_flush("=" * 70)

        pythia_rev = results["pythia"]["reversal_curse"]
        infini_rev = results["infini"]["reversal_curse"]

        print_flush("\n| Model | Forward PPL | Backward PPL | Ratio | Gap |")
        print_flush("|-------|-------------|--------------|-------|-----|")
        print_flush(
            f"| Pythia | {pythia_rev['forward_ppl']:.1f} | "
            f"{pythia_rev['backward_ppl']:.1f} | "
            f"{pythia_rev['reversal_ratio']:.3f} | "
            f"{pythia_rev['reversal_gap']:+.1f} |"
        )
        print_flush(
            f"| Infini | {infini_rev['forward_ppl']:.1f} | "
            f"{infini_rev['backward_ppl']:.1f} | "
            f"{infini_rev['reversal_ratio']:.3f} | "
            f"{infini_rev['reversal_gap']:+.1f} |"
        )

        print_flush("\n(Reversal Ratio closer to 1.0 = less reversal curse)")

    return results


def main() -> None:
    config = PythiaConfig()

    parser = argparse.ArgumentParser(description="Infini-Attention Experiment")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=256, help="Sequence length (longer is better for Infini)")
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
        "--kv-dim", type=int, default=128, help="KV compression dimension"
    )
    parser.add_argument(
        "--alibi-slope", type=float, default=0.0625, help="ALiBi slope (uniform)"
    )
    parser.add_argument(
        "--no-delta-rule", action="store_true", help="Disable delta rule for Infini"
    )
    parser.add_argument(
        "--skip-baseline", action="store_true", help="Skip Pythia baseline"
    )
    args = parser.parse_args()

    run_experiment(
        num_samples=args.samples,
        seq_length=args.seq_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        kv_dim=args.kv_dim,
        alibi_slope=args.alibi_slope,
        use_delta_rule=not args.no_delta_rule,
        skip_baseline=args.skip_baseline,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
