#!/usr/bin/env python3
"""
KA-Attention vs Pythia Comparison Experiment

KA-Attention方式とBaselineのPythiaを比較する実験。

Usage:
    python3 scripts/experiment_ka_comparison.py --samples 10000 --epochs 10
"""

import argparse
import time
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.pythia import PythiaConfig
from src.models.pythia import PythiaModel
from src.models.ka_attention import KAPythiaModel
from src.utils.io import print_flush


def prepare_data(
    num_samples: int,
    seq_length: int,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader]:
    """Prepare training and validation data from Pile"""
    print_flush(f"Preparing data: {num_samples:,} samples, seq_len={seq_length}")

    # Load from cache or download
    from src.utils.data_pythia import load_pile_tokens

    total_tokens_needed = num_samples * seq_length
    tokens = load_pile_tokens(total_tokens_needed + seq_length)

    # Create samples
    all_input_ids = []
    all_labels = []

    for i in range(num_samples):
        start = i * seq_length
        input_ids = tokens[start:start + seq_length]
        labels = tokens[start + 1:start + seq_length + 1]
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    all_input_ids = torch.stack(all_input_ids)
    all_labels = torch.stack(all_labels)

    # Split
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size

    train_dataset = TensorDataset(all_input_ids[:train_size], all_labels[:train_size])
    val_dataset = TensorDataset(all_input_ids[train_size:], all_labels[train_size:])

    print_flush(f"  Train: {train_size:,} samples")
    print_flush(f"  Val: {val_size:,} samples")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for input_ids, labels in val_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    model_name: str,
) -> Tuple[float, int]:
    """Train model and return best validation perplexity"""
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_flush(f"  Trainable: {trainable_params:,} / {total_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    patience = 3

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        train_ppl = torch.exp(torch.tensor(train_loss)).item()
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        epoch_time = time.time() - start_time

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1

        print_flush(
            f"  Epoch {epoch:2d}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
            f"[{epoch_time:.1f}s]{marker}"
        )

        if patience_counter >= patience:
            print_flush(f"  → Early stop: val_ppl worsened for {patience} epochs")
            break

    best_val_ppl = torch.exp(torch.tensor(best_val_loss)).item()
    print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

    return best_val_ppl, best_epoch


def main():
    parser = argparse.ArgumentParser(description="KA-Attention vs Pythia Comparison")
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--skip-baseline", action="store_true", help="Skip baseline training")
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_flush(f"Device: cuda ({gpu_name}, {gpu_mem:.1f}GB)")
    else:
        device = torch.device("cpu")
        print_flush("Device: cpu")

    print_flush("=" * 70)
    print_flush("KA-ATTENTION vs PYTHIA COMPARISON")
    print_flush("=" * 70)
    print_flush(f"Samples: {args.samples:,}")
    print_flush(f"Sequence length: {args.seq_length}")
    print_flush(f"Epochs: {args.epochs}")
    print_flush(f"Learning rate: {args.lr}")
    print_flush("=" * 70)

    # Config
    config = PythiaConfig()

    print_flush("\nArchitecture comparison:")
    print_flush(f"  Pythia:      hidden_size={config.hidden_size}, intermediate={config.intermediate_size}")
    print_flush(f"  KA-Pythia:   hidden_size={config.hidden_size}, intermediate={config.intermediate_size}")
    print_flush("  (Same architecture, different attention mechanism)")
    print_flush("=" * 70)

    # Load data
    print_flush("\n[Data] Loading Pile data...")
    train_loader, val_loader = prepare_data(
        num_samples=args.samples,
        seq_length=args.seq_length,
    )

    results = {}

    # 1. Baseline Pythia
    if not args.skip_baseline:
        print_flush("\n" + "=" * 70)
        print_flush("1. PYTHIA-70M (Baseline)")
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

        print_flush(f"\n[Pythia-70M] Training...")
        best_ppl, best_epoch = train_model(
            pythia_model,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.lr,
            "Pythia",
        )
        results["Pythia"] = best_ppl

        del pythia_model
        torch.cuda.empty_cache()

    # 2. KA-Pythia
    print_flush("\n" + "=" * 70)
    print_flush("2. KA-PYTHIA (KA-Attention)")
    print_flush("=" * 70)

    ka_model = KAPythiaModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        rotary_pct=config.rotary_pct,
    )

    print_flush(f"\n[KA-Pythia] Training...")
    best_ppl, best_epoch = train_model(
        ka_model,
        train_loader,
        val_loader,
        device,
        args.epochs,
        args.lr,
        "KA-Pythia",
    )
    results["KA-Pythia"] = best_ppl

    # Summary
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    for name, ppl in results.items():
        print_flush(f"  {name}: val_ppl={ppl:.1f}")

    if "Pythia" in results and "KA-Pythia" in results:
        diff = results["KA-Pythia"] - results["Pythia"]
        pct = (diff / results["Pythia"]) * 100
        print_flush(f"\n  Difference: {diff:+.1f} ppl ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
