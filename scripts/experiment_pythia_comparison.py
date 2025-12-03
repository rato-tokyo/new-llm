#!/usr/bin/env python3
"""
Pythia vs Context-Pythia 比較実験

オリジナルPythiaアーキテクチャとContext-Pythiaを比較し、
KVキャッシュ削減効果と性能を検証する。

Usage:
    python3 scripts/experiment_pythia_comparison.py --samples 10000 --seq-length 256 --epochs 10
"""

import argparse
import time
import sys
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer

# Add project root to path
sys.path.insert(0, ".")

from config.pythia import PythiaConfig, ContextPythiaConfig
from src.models.pythia import PythiaModel
from src.models.context_pythia import ContextPythiaModel
from src.losses.diversity import oacd_loss
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


def load_pile_data(
    num_samples: int,
    seq_length: int,
    config: PythiaConfig,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load real data from Pile dataset using Pythia tokenizer.

    Args:
        num_samples: Number of samples to load
        seq_length: Sequence length (will load seq_length + 1 for target shift)
        config: PythiaConfig with tokenizer settings
        device: Device to load data to

    Returns:
        inputs: [num_samples, seq_length]
        targets: [num_samples, seq_length]
    """
    print_flush(f"Loading Pile dataset: {num_samples} samples, seq_len={seq_length}")

    # Load tokenizer
    print_flush(f"  Loading tokenizer: {config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Pile dataset (streaming to avoid downloading full dataset)
    print_flush("  Loading dataset (streaming)...")
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    # Collect samples
    all_input_ids = []
    total_tokens_needed = num_samples * (seq_length + 1)

    print_flush(f"  Tokenizing (need {total_tokens_needed:,} tokens)...")

    for example in dataset:
        text = example["text"]
        if not text or len(text) < 100:  # Skip very short texts
            continue

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) >= seq_length + 1:
            # Take chunks of seq_length + 1
            for i in range(0, len(tokens) - seq_length, seq_length + 1):
                chunk = tokens[i:i + seq_length + 1]
                all_input_ids.append(chunk)

                if len(all_input_ids) >= num_samples:
                    break

        if len(all_input_ids) >= num_samples:
            break

        if len(all_input_ids) % 1000 == 0 and len(all_input_ids) > 0:
            print_flush(f"    Collected {len(all_input_ids):,} samples...")

    if len(all_input_ids) < num_samples:
        print_flush(f"  Warning: Only collected {len(all_input_ids)} samples (requested {num_samples})")
        num_samples = len(all_input_ids)

    # Convert to tensor
    all_input_ids = all_input_ids[:num_samples]
    input_ids = torch.tensor(all_input_ids, dtype=torch.long, device=device)

    # Split into inputs and targets
    inputs = input_ids[:, :-1].contiguous()
    targets = input_ids[:, 1:].contiguous()

    print_flush(f"  Loaded {num_samples:,} samples, {inputs.numel():,} tokens")

    return inputs, targets


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Train for one epoch (full data)."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
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

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == num_batches:
            print_flush(f"  Batch {batch_idx + 1}/{num_batches}: loss={loss.item():.4f}")

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


def train_phase1_oacd(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config: PythiaConfig,
) -> float:
    """
    Phase 1: Train ContextBlock with OACD diversity loss.

    Only trains the context_block parameters.
    """
    model.train()

    # Only optimize context_block
    optimizer = torch.optim.AdamW(
        model.context_block.parameters(),
        lr=config.phase1_learning_rate,
    )

    total_loss = 0.0
    num_batches = 0

    for iteration in range(config.phase1_max_iterations):
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (inputs, _) in enumerate(train_loader):
            if batch_idx >= config.phase1_batches_per_iteration:
                break

            inputs = inputs.to(device)

            optimizer.zero_grad()

            # Forward to get context
            _, context = model.forward_with_context_output(inputs)

            # Flatten context for OACD loss: [batch*seq, context_dim]
            context_flat = context.view(-1, context.size(-1))

            # OACD diversity loss
            loss = oacd_loss(context_flat, centroid_weight=0.1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.context_block.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / max(batch_count, 1)
        total_loss += avg_loss
        num_batches += 1

        if (iteration + 1) % 10 == 0:
            print_flush(f"  Phase 1 Iter {iteration + 1}: OACD loss={avg_loss:.4f}")

        # Early stopping check
        if avg_loss < -config.phase1_convergence_threshold:
            print_flush(f"  Phase 1 converged at iteration {iteration + 1}")
            break

    return total_loss / max(num_batches, 1)


def measure_memory(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Measure peak memory usage during forward pass using real data."""
    if not torch.cuda.is_available():
        return {"peak_mb": 0, "model_mb": 0}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model.eval()

    # Measure model memory
    model_memory = torch.cuda.memory_allocated() / (1024 * 1024)

    # Forward pass with real data
    with torch.no_grad():
        _ = model(sample_input.to(device))

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

    # Load data from Pile
    print_flush("\n[Data] Loading from Pile...")
    inputs, targets = load_pile_data(
        num_samples=num_samples,
        seq_length=seq_length,
        config=config,
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

    # Measure memory using real data sample
    sample_input = train_inputs[:batch_size]
    pythia_memory = measure_memory(pythia, sample_input, device)
    print_flush(f"Peak memory: {pythia_memory['peak_mb']:.1f} MB")

    # Train
    optimizer = torch.optim.AdamW(pythia.parameters(), lr=config.learning_rate)

    print_flush("\nTraining...")
    start_time = time.time()

    for epoch in range(num_epochs):
        train_ppl, train_acc = train_epoch(pythia, train_loader, optimizer, device)
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

    # Measure memory using real data sample
    context_memory = measure_memory(context_pythia, sample_input, device)
    print_flush(f"Peak memory: {context_memory['peak_mb']:.1f} MB")

    # ========== Phase 1: OACD Training ==========
    print_flush("\n--- Phase 1: OACD (ContextBlock diversity) ---")
    start_time = time.time()

    phase1_loss = train_phase1_oacd(
        context_pythia, train_loader, device, context_config
    )
    phase1_time = time.time() - start_time
    print_flush(f"Phase 1 completed: avg_loss={phase1_loss:.4f}, time={phase1_time:.1f}s")

    # ========== Phase 2: Full Training (ContextBlock frozen) ==========
    print_flush("\n--- Phase 2: Full Training (ContextBlock frozen) ---")

    # Freeze ContextBlock
    context_pythia.freeze_context_block()

    # Optimizer for non-frozen parameters only
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, context_pythia.parameters()),
        lr=context_config.phase2_learning_rate,
    )

    start_time = time.time()

    for epoch in range(num_epochs):
        train_ppl, train_acc = train_epoch(context_pythia, train_loader, optimizer, device)
        val_ppl, val_acc = evaluate(context_pythia, val_loader, device)
        print_flush(
            f"Epoch {epoch + 1}: train_ppl={train_ppl:.1f}, "
            f"val_ppl={val_ppl:.1f}, val_acc={val_acc*100:.1f}%"
        )

    phase2_time = time.time() - start_time
    context_time = phase1_time + phase2_time

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
        '--samples', type=int, required=True,
        help='Number of samples (REQUIRED)'
    )
    parser.add_argument(
        '--seq-length', type=int, required=True,
        help='Sequence length (REQUIRED)'
    )
    parser.add_argument(
        '--epochs', type=int, required=True,
        help='Number of epochs (REQUIRED)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=config.phase2_batch_size,
        help=f'Batch size (default: {config.phase2_batch_size})'
    )
    parser.add_argument(
        '--seed', type=int, default=config.random_seed,
        help=f'Random seed (default: {config.random_seed})'
    )

    args = parser.parse_args()

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
