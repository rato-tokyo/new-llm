#!/usr/bin/env python3
"""
Pythia vs Context-Pythia 比較実験

Pythia-70M（ベースライン）とContext-Pythia（KV圧縮50%）を比較する。

Usage:
    python3 scripts/experiment_pythia_comparison.py --samples 10000 --epochs 10
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.insert(0, ".")

from config import ContextPythiaConfig, PythiaConfig
from src.models.pythia import PythiaModel
from src.models.context_pythia import ContextPythiaModel
from src.utils.data_pythia import prepare_pythia_phase2_data
from src.utils.io import print_flush
from src.utils.device import clear_gpu_cache


def train_epoch(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[float, float]:
    """
    1エポックの学習

    Returns:
        avg_loss, accuracy
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    num_samples = len(train_inputs)
    indices = torch.randperm(num_samples)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_idx = indices[start:end]

        batch_inputs = train_inputs[batch_idx].to(device)
        batch_targets = train_targets[batch_idx].to(device)

        optimizer.zero_grad()

        # Forward
        logits = model(batch_inputs)  # [batch, seq_len, vocab_size]

        # Reshape for loss
        batch_size_actual, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = batch_targets.reshape(-1)

        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_size_actual * seq_len
        total_correct += (logits_flat.argmax(dim=-1) == targets_flat).sum().item()
        total_tokens += batch_size_actual * seq_len

        del batch_inputs, batch_targets, logits
        clear_gpu_cache(device)

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens

    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[float, float, float]:
    """
    評価

    Returns:
        avg_loss, perplexity, accuracy
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    num_samples = len(val_inputs)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)

        batch_inputs = val_inputs[start:end].to(device)
        batch_targets = val_targets[start:end].to(device)

        logits = model(batch_inputs)

        batch_size_actual, seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = batch_targets.reshape(-1)

        loss = criterion(logits_flat, targets_flat)

        total_loss += loss.item() * batch_size_actual * seq_len
        total_correct += (logits_flat.argmax(dim=-1) == targets_flat).sum().item()
        total_tokens += batch_size_actual * seq_len

        del batch_inputs, batch_targets, logits
        clear_gpu_cache(device)

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = total_correct / total_tokens

    return avg_loss, perplexity, accuracy


def train_model(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    val_inputs: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    モデルを学習

    Returns:
        results: 学習結果
    """
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()

    best_val_ppl = float("inf")
    best_epoch = 0

    print_flush(f"\n[{model_name}] Training...")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print_flush(f"  Trainable: {trainable_params:,} / {total_params:,} parameters")

    history = []

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_inputs, train_targets, optimizer, criterion, device, batch_size
        )

        # Evaluate
        val_loss, val_ppl, val_acc = evaluate(
            model, val_inputs, val_targets, criterion, device, batch_size
        )

        epoch_time = time.time() - epoch_start

        # Early stopping: stop immediately if val_ppl worsens
        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            best_epoch = epoch
            marker = " *"
        else:
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train_loss={train_loss:.4f} "
            f"val_ppl={val_ppl:.1f} val_acc={val_acc*100:.2f}% "
            f"[{epoch_time:.1f}s]{marker}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
            "val_acc": val_acc,
        })

        # Stop immediately if val_ppl didn't improve
        if not improved and epoch > 1:
            print_flush(f"  → Early stop: val_ppl worsened ({val_ppl:.1f} > {best_val_ppl:.1f})")
            break

    print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

    return {
        "best_epoch": best_epoch,
        "best_val_ppl": best_val_ppl,
        "history": history,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }


def run_experiment(
    num_samples: int = 10000,
    seq_length: int = 128,
    num_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    skip_baseline: bool = False,
) -> Dict[str, Any]:
    """
    比較実験を実行

    Args:
        skip_baseline: Trueの場合、Pythia-70M (baseline)をスキップ

    Returns:
        results: 両モデルの結果
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print_flush(f"Device: {device} ({gpu_name}, {gpu_mem:.1f}GB)")
    else:
        print_flush(f"Device: {device}")

    pythia_config = PythiaConfig()
    context_config = ContextPythiaConfig()

    print_flush("=" * 70)
    print_flush("PYTHIA vs CONTEXT-PYTHIA COMPARISON")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples:,}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Epochs: {num_epochs}")
    print_flush(f"Batch size: {batch_size}")
    print_flush(f"Learning rate: {lr}")
    print_flush("=" * 70)
    print_flush("\nArchitecture comparison:")
    print_flush(f"  Pythia:         hidden_size={pythia_config.hidden_size}, intermediate={pythia_config.intermediate_size}")
    print_flush(f"  Context-Pythia: context_dim={context_config.context_dim}, intermediate={context_config.intermediate_size}")
    print_flush("=" * 70)

    # Load data
    print_flush("\n[Data] Loading Pile data...")
    train_inputs, train_targets, val_inputs, val_targets = prepare_pythia_phase2_data(
        num_samples=num_samples,
        seq_length=seq_length,
        val_split=0.1,
        tokenizer_name=pythia_config.tokenizer_name,
        device=device,
    )
    print_flush(f"  Train: {len(train_inputs):,} samples")
    print_flush(f"  Val: {len(val_inputs):,} samples")

    results = {}

    # KV cache sizes (K + V, float32)
    kv_size_pythia = pythia_config.hidden_size * seq_length * pythia_config.num_layers * 2 * 4
    kv_size_context = context_config.context_dim * seq_length * context_config.num_layers * 2 * 4

    # ===== 1. Pythia (Baseline) =====
    if skip_baseline:
        print_flush("\n" + "=" * 70)
        print_flush("1. PYTHIA-70M (Baseline) - SKIPPED")
        print_flush("=" * 70)
        results["pythia"] = None
    else:
        print_flush("\n" + "=" * 70)
        print_flush("1. PYTHIA-70M (Baseline)")
        print_flush("=" * 70)

        pythia_model = PythiaModel(
            vocab_size=pythia_config.vocab_size,
            hidden_size=pythia_config.hidden_size,
            num_layers=pythia_config.num_layers,
            num_heads=pythia_config.num_attention_heads,
            intermediate_size=pythia_config.intermediate_size,
            max_position_embeddings=pythia_config.max_position_embeddings,
            rotary_pct=pythia_config.rotary_pct,
        ).to(device)

        pythia_results = train_model(
            pythia_model,
            train_inputs, train_targets,
            val_inputs, val_targets,
            device,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            model_name="Pythia-70M",
        )
        results["pythia"] = pythia_results

        print_flush(f"  KV Cache: {kv_size_pythia / 1024:.1f} KB per sample")

        del pythia_model
        clear_gpu_cache(device)

    # ===== 2. Context-Pythia (Ours) =====
    print_flush("\n" + "=" * 70)
    print_flush("2. CONTEXT-PYTHIA (Ours - 50% KV Compression)")
    print_flush("=" * 70)

    context_pythia_model = ContextPythiaModel(
        vocab_size=context_config.vocab_size,
        embed_dim=context_config.embed_dim,
        context_dim=context_config.context_dim,
        num_layers=context_config.num_layers,
        num_heads=context_config.num_attention_heads,
        intermediate_size=context_config.intermediate_size,
        max_position_embeddings=context_config.max_position_embeddings,
        rotary_pct=context_config.rotary_pct,
    ).to(device)

    # Load Phase 1 checkpoint
    checkpoint_path = Path(context_config.phase1_checkpoint_path)
    if checkpoint_path.exists():
        print_flush(f"  Loading Phase 1 checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        context_pythia_model.context_block.load_state_dict(
            checkpoint["context_block_state_dict"]
        )
        print_flush("  ✓ ContextBlock loaded")
    else:
        print_flush(f"  ⚠️ Phase 1 checkpoint not found: {checkpoint_path}")
        print_flush("  Using random initialization for ContextBlock")

    # Freeze ContextBlock
    context_pythia_model.freeze_context_block()

    context_pythia_results = train_model(
        context_pythia_model,
        train_inputs, train_targets,
        val_inputs, val_targets,
        device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        model_name="Context-Pythia",
    )
    results["context_pythia"] = context_pythia_results

    print_flush(f"  KV Cache: {kv_size_context / 1024:.1f} KB per sample")

    del context_pythia_model
    clear_gpu_cache(device)

    # ===== Summary =====
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | Best PPL | KV Cache | Reduction |")
    print_flush("|-------|----------|----------|-----------|")

    if results["pythia"] is not None:
        print_flush(
            f"| Pythia-70M | {results['pythia']['best_val_ppl']:.1f} | "
            f"{kv_size_pythia / 1024:.1f} KB | - |"
        )
    else:
        print_flush(f"| Pythia-70M | (skipped) | {kv_size_pythia / 1024:.1f} KB | - |")

    print_flush(
        f"| Context-Pythia | {results['context_pythia']['best_val_ppl']:.1f} | "
        f"{kv_size_context / 1024:.1f} KB | 50% |"
    )

    if results["pythia"] is not None:
        ppl_diff = results["context_pythia"]["best_val_ppl"] - results["pythia"]["best_val_ppl"]
        print_flush(f"\nPPL difference: {ppl_diff:+.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Pythia vs Context-Pythia Comparison")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip Pythia-70M baseline")
    args = parser.parse_args()

    run_experiment(
        num_samples=args.samples,
        seq_length=args.seq_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        skip_baseline=args.skip_baseline,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
