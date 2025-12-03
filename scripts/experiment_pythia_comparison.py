#!/usr/bin/env python3
"""
Pythia vs Context-Pythia 比較実験

オリジナルPythiaアーキテクチャとContext-Pythiaを比較し、
KVキャッシュ削減効果と性能を検証する。

Usage:
    # 開発モード（限定データ）
    python3 scripts/experiment_pythia_comparison.py --dev

    # フルモード
    python3 scripts/experiment_pythia_comparison.py --samples 10000
"""

import argparse
import time
import sys
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, ".")

from config.pythia import PythiaConfig, ContextPythiaConfig
from src.models.pythia import PythiaModel
from src.models.context_pythia import ContextPythiaModel
from src.utils.io import print_flush


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_memory_usage() -> str:
    """Get current memory usage."""
    import psutil
    process = psutil.Process()
    cpu_gb = process.memory_info().rss / (1024**3)
    if torch.cuda.is_available():
        gpu_gb = torch.cuda.memory_allocated() / (1024**3)
        return f"CPU: {cpu_gb:.1f}GB, GPU: {gpu_gb:.1f}GB"
    return f"CPU: {cpu_gb:.1f}GB"


def load_sample_data(
    num_samples: int,
    seq_length: int,
    vocab_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load sample data for training/evaluation.

    For development, use random data. For production, load from Pile.
    """
    print_flush(f"Generating random data: {num_samples} samples, seq_len={seq_length}")

    # Random token IDs
    input_ids = torch.randint(0, vocab_size, (num_samples, seq_length), device=device)

    # Targets: shifted input_ids
    targets = input_ids[:, 1:].contiguous()
    inputs = input_ids[:, :-1].contiguous()

    return inputs, targets


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_batches: int = None,
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if max_batches and batch_idx >= max_batches:
            break

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward
        logits = model(inputs)

        # Loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item() * targets.numel()
        predictions = logits.argmax(dim=-1)
        total_correct += (predictions == targets).sum().item()
        total_tokens += targets.numel()

        if (batch_idx + 1) % 10 == 0:
            print_flush(f"  Batch {batch_idx + 1}: loss={loss.item():.4f}")

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    return ppl, accuracy


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                reduction='sum',
            )

            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_tokens += targets.numel()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    return ppl, accuracy


def measure_memory(
    model: nn.Module,
    seq_length: int,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """Measure peak memory usage during forward pass."""
    if not torch.cuda.is_available():
        return {"peak_mb": 0, "model_mb": 0}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model.eval()

    # Measure model memory
    model_memory = torch.cuda.memory_allocated() / (1024 * 1024)

    # Forward pass
    dummy_input = torch.randint(0, 1000, (batch_size, seq_length), device=device)
    with torch.no_grad():
        _ = model(dummy_input)

    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "model_mb": model_memory,
        "peak_mb": peak_memory,
        "forward_mb": peak_memory - model_memory,
    }


def run_comparison(
    num_samples: int,
    seq_length: int,
    batch_size: int,
    num_epochs: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Run comparison experiment."""

    set_seed(seed)
    config = PythiaConfig()

    print_flush("=" * 70)
    print_flush("PYTHIA VS CONTEXT-PYTHIA COMPARISON")
    print_flush("=" * 70)
    print_flush(f"Samples: {num_samples}")
    print_flush(f"Sequence length: {seq_length}")
    print_flush(f"Batch size: {batch_size}")
    print_flush(f"Epochs: {num_epochs}")
    print_flush(f"Device: {device}")
    print_flush("=" * 70)

    # Load data
    print_flush("\n[Data] Loading...")
    inputs, targets = load_sample_data(
        num_samples=num_samples,
        seq_length=seq_length + 1,  # +1 for target shift
        vocab_size=config.vocab_size,
        device=device,
    )

    # Split train/val
    split_idx = int(num_samples * 0.9)
    train_inputs, val_inputs = inputs[:split_idx], inputs[split_idx:]
    train_targets, val_targets = targets[:split_idx], targets[split_idx:]

    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print_flush(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    results = {}

    # ========== Original Pythia ==========
    print_flush("\n" + "=" * 70)
    print_flush("[1/2] ORIGINAL PYTHIA")
    print_flush("=" * 70)

    pythia = PythiaModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        rotary_pct=config.rotary_pct,
    ).to(device)

    pythia_params = pythia.num_parameters()
    print_flush(f"Parameters: {pythia_params['total']:,}")
    print_flush(f"Memory: {get_memory_usage()}")

    # Measure memory
    pythia_memory = measure_memory(pythia, seq_length, batch_size, device)
    print_flush(f"Peak memory: {pythia_memory['peak_mb']:.1f} MB")

    # Train
    optimizer = torch.optim.AdamW(pythia.parameters(), lr=config.learning_rate)

    print_flush("\nTraining...")
    start_time = time.time()

    for epoch in range(num_epochs):
        train_ppl, train_acc = train_epoch(
            pythia, train_loader, optimizer, device, max_batches=50
        )
        val_ppl, val_acc = evaluate(pythia, val_loader, device)
        print_flush(
            f"Epoch {epoch + 1}: train_ppl={train_ppl:.1f}, "
            f"val_ppl={val_ppl:.1f}, val_acc={val_acc*100:.1f}%"
        )

    pythia_time = time.time() - start_time

    results["pythia"] = {
        "params": pythia_params["total"],
        "val_ppl": val_ppl,
        "val_acc": val_acc,
        "peak_memory_mb": pythia_memory["peak_mb"],
        "train_time": pythia_time,
    }

    # Clear memory
    del pythia, optimizer
    torch.cuda.empty_cache()

    # ========== Context-Pythia ==========
    print_flush("\n" + "=" * 70)
    print_flush("[2/2] CONTEXT-PYTHIA (50% KV Reduction)")
    print_flush("=" * 70)

    context_config = ContextPythiaConfig()

    context_pythia = ContextPythiaModel(
        vocab_size=context_config.vocab_size,
        hidden_size=context_config.hidden_size,
        context_dim=context_config.context_dim,
        num_layers=context_config.num_layers,
        num_heads=context_config.num_heads,
        intermediate_size=context_config.intermediate_size,
        max_position_embeddings=context_config.max_position_embeddings,
        rotary_pct=context_config.rotary_pct,
    ).to(device)

    context_params = context_pythia.num_parameters()
    print_flush(f"Parameters: {context_params['total']:,}")
    print_flush(f"  - ContextBlock: {context_params['context_block']:,}")
    print_flush(f"  - Layers: {context_params['layers']:,}")
    print_flush(f"Memory: {get_memory_usage()}")

    # KV cache comparison
    kv_comparison = context_pythia.kv_cache_size_comparison(seq_length)
    print_flush(f"\nKV Cache Reduction:")
    print_flush(f"  Original: {kv_comparison['original_mb']:.2f} MB")
    print_flush(f"  Context:  {kv_comparison['context_mb']:.2f} MB")
    print_flush(f"  Reduction: {kv_comparison['reduction_pct']:.1f}%")

    # Measure memory
    context_memory = measure_memory(context_pythia, seq_length, batch_size, device)
    print_flush(f"Peak memory: {context_memory['peak_mb']:.1f} MB")

    # Train
    optimizer = torch.optim.AdamW(context_pythia.parameters(), lr=config.learning_rate)

    print_flush("\nTraining...")
    start_time = time.time()

    for epoch in range(num_epochs):
        train_ppl, train_acc = train_epoch(
            context_pythia, train_loader, optimizer, device, max_batches=50
        )
        val_ppl, val_acc = evaluate(context_pythia, val_loader, device)
        print_flush(
            f"Epoch {epoch + 1}: train_ppl={train_ppl:.1f}, "
            f"val_ppl={val_ppl:.1f}, val_acc={val_acc*100:.1f}%"
        )

    context_time = time.time() - start_time

    results["context_pythia"] = {
        "params": context_params["total"],
        "val_ppl": val_ppl,
        "val_acc": val_acc,
        "peak_memory_mb": context_memory["peak_mb"],
        "train_time": context_time,
        "kv_reduction_pct": kv_comparison["reduction_pct"],
    }

    # ========== Summary ==========
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush("\n| Model | Params | Val PPL | Val Acc | Peak Mem | KV Reduction |")
    print_flush("|-------|--------|---------|---------|----------|--------------|")

    p = results["pythia"]
    print_flush(
        f"| Pythia | {p['params']:,} | {p['val_ppl']:.1f} | "
        f"{p['val_acc']*100:.1f}% | {p['peak_memory_mb']:.0f} MB | - |"
    )

    c = results["context_pythia"]
    print_flush(
        f"| Context-Pythia | {c['params']:,} | {c['val_ppl']:.1f} | "
        f"{c['val_acc']*100:.1f}% | {c['peak_memory_mb']:.0f} MB | "
        f"{c['kv_reduction_pct']:.0f}% |"
    )

    return results


def main() -> None:
    config = PythiaConfig()

    parser = argparse.ArgumentParser(
        description='Pythia vs Context-Pythia Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--dev', action='store_true',
        help='Development mode (limited data)'
    )
    parser.add_argument(
        '--samples', type=int, default=config.dev_num_samples,
        help=f'Number of samples (default: {config.dev_num_samples})'
    )
    parser.add_argument(
        '--seq-length', type=int, default=config.dev_max_seq_length,
        help=f'Sequence length (default: {config.dev_max_seq_length})'
    )
    parser.add_argument(
        '--batch-size', type=int, default=config.phase2_batch_size,
        help=f'Batch size (default: {config.phase2_batch_size})'
    )
    parser.add_argument(
        '--epochs', type=int, default=3,
        help='Number of epochs (default: 3)'
    )
    parser.add_argument(
        '--seed', type=int, default=config.random_seed,
        help=f'Random seed (default: {config.random_seed})'
    )

    args = parser.parse_args()

    # Development mode overrides
    if args.dev:
        args.samples = 500
        args.seq_length = 128
        args.epochs = 2
        print_flush("*** DEVELOPMENT MODE ***")

    device = torch.device(config.device)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print_flush(f"Device: {device} ({gpu_name}, {gpu_mem:.1f}GB)")
    else:
        print_flush(f"Device: {device}")

    results = run_comparison(
        num_samples=args.samples,
        seq_length=args.seq_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        seed=args.seed,
        device=device,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
