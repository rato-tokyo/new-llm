#!/usr/bin/env python3
"""
Phase 1: ContextBlock OACD Training

ContextBlockの多様性学習を行い、チェックポイントを保存する。
このスクリプトで学習したパラメータは、experiment_pythia_comparison.pyで使用される。

Note: Phase 1ではシーケンス長は本質的に関係ない（contextをflattenするため）。
内部で固定長(128)のシーケンスを使用する。

Usage:
    python3 scripts/train_phase1.py --samples 1000
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub.utils import HfHubHTTPError

# Add project root to path
sys.path.insert(0, ".")

from config.pythia import ContextPythiaConfig
from src.models.context_pythia import ContextPythiaModel
from src.losses.diversity import oacd_loss
from src.utils.io import print_flush


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Phase 1用の固定シーケンス長（任意の値でよい）
PHASE1_SEQ_LENGTH = 128


def load_pile_samples(
    num_samples: int,
    config: ContextPythiaConfig,
    device: torch.device,
    max_retries: int = 5,
    retry_delay: float = 30.0,
) -> torch.Tensor:
    """Load samples from Pile dataset for Phase 1 training.

    Phase 1ではcontextをflattenするため、シーケンス長は本質的に関係ない。
    内部で固定長(128)のシーケンスを使用する。

    Args:
        num_samples: シーケンス数（= サンプル数）
        max_retries: 429エラー時の最大リトライ回数
        retry_delay: リトライ間の待機時間（秒）

    Returns:
        input_ids: [num_samples, PHASE1_SEQ_LENGTH]
    """
    total_tokens_needed = num_samples * PHASE1_SEQ_LENGTH
    print_flush(f"Loading Pile dataset: {num_samples:,} samples ({total_tokens_needed:,} tokens)")

    # Load tokenizer
    print_flush(f"  Loading tokenizer: {config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Pile dataset (streaming)
    print_flush("  Loading dataset (streaming)...")
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
    )

    # Collect tokens with retry logic
    all_tokens = []
    retry_count = 0

    print_flush(f"  Tokenizing...")

    dataset_iter = iter(dataset)
    while len(all_tokens) < total_tokens_needed:
        try:
            example = next(dataset_iter)
            text = example["text"]
            if not text or len(text) < 100:
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

            if len(all_tokens) % 100000 == 0 and len(all_tokens) > 0:
                print_flush(f"    Collected {len(all_tokens):,} tokens...")

            retry_count = 0  # Reset on success

        except StopIteration:
            break
        except HfHubHTTPError as e:
            if "429" in str(e) and retry_count < max_retries:
                retry_count += 1
                print_flush(f"    Rate limited (429). Retry {retry_count}/{max_retries} after {retry_delay}s...")
                time.sleep(retry_delay)
                # Recreate iterator
                dataset = load_dataset(
                    "monology/pile-uncopyrighted",
                    split="train",
                    streaming=True,
                )
                dataset_iter = iter(dataset)
                # Skip already collected samples (approximate)
                skip_count = len(all_tokens) // 500  # Rough estimate
                for _ in range(skip_count):
                    try:
                        next(dataset_iter)
                    except StopIteration:
                        break
            else:
                raise

    # Split into fixed-length sequences
    all_tokens = all_tokens[:num_samples * PHASE1_SEQ_LENGTH]

    input_ids = torch.tensor(all_tokens, dtype=torch.long, device=device)
    input_ids = input_ids.view(num_samples, PHASE1_SEQ_LENGTH)

    print_flush(f"  Loaded {num_samples:,} samples ({input_ids.numel():,} tokens)")

    return input_ids


def train_phase1(
    model: ContextPythiaModel,
    train_loader: DataLoader,
    config: ContextPythiaConfig,
    device: torch.device,
) -> float:
    """
    Phase 1: Train ContextBlock with OACD diversity loss.

    Returns:
        Final average loss
    """
    model.train()

    # Only optimize context_block
    optimizer = torch.optim.AdamW(
        model.context_block.parameters(),
        lr=config.phase1_learning_rate,
    )

    total_loss = 0.0
    num_iterations = 0

    print_flush(f"\nPhase 1 Training: {config.phase1_max_iterations} iterations")
    print_flush(f"  Learning rate: {config.phase1_learning_rate}")
    print_flush(f"  Batches per iteration: {config.phase1_batches_per_iteration}")
    print_flush(f"  Convergence threshold: {config.phase1_convergence_threshold}")

    start_time = time.time()

    for iteration in range(config.phase1_max_iterations):
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (inputs,) in enumerate(train_loader):
            if batch_idx >= config.phase1_batches_per_iteration:
                break

            inputs = inputs.to(device)

            optimizer.zero_grad()

            # Forward to get context
            _, context = model.forward_with_context_output(inputs)

            # Flatten context: [batch*seq, context_dim]
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
        num_iterations += 1

        if (iteration + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print_flush(f"  Iter {iteration + 1}/{config.phase1_max_iterations}: "
                       f"loss={avg_loss:.4f}, time={elapsed:.1f}s")

        # Early stopping
        if avg_loss < -config.phase1_convergence_threshold:
            print_flush(f"  Converged at iteration {iteration + 1}")
            break

    final_loss = total_loss / max(num_iterations, 1)
    total_time = time.time() - start_time

    print_flush(f"\nPhase 1 completed:")
    print_flush(f"  Iterations: {num_iterations}")
    print_flush(f"  Final loss: {final_loss:.4f}")
    print_flush(f"  Total time: {total_time:.1f}s")

    return final_loss


def save_checkpoint(
    model: ContextPythiaModel,
    config: ContextPythiaConfig,
    final_loss: float,
) -> None:
    """Save ContextBlock checkpoint."""
    checkpoint_path = Path(config.phase1_checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "context_block_state_dict": model.context_block.state_dict(),
        "config": {
            "context_dim": config.context_dim,
            "hidden_size": config.hidden_size,
        },
        "final_loss": final_loss,
    }

    torch.save(checkpoint, checkpoint_path)
    print_flush(f"\nCheckpoint saved: {checkpoint_path}")


def main() -> None:
    config = ContextPythiaConfig()

    parser = argparse.ArgumentParser(
        description='Phase 1: ContextBlock OACD Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--samples', type=int, required=True,
        help='Number of samples (REQUIRED)'
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

    # Device setup
    device = torch.device(config.device)
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print_flush(f"Device: {device} ({gpu_name}, {gpu_mem:.1f}GB)")
    else:
        print_flush(f"Device: {device}")

    set_seed(args.seed)

    print_flush("=" * 70)
    print_flush("PHASE 1: CONTEXTBLOCK OACD TRAINING")
    print_flush("=" * 70)
    print_flush(f"Samples: {args.samples:,}")
    print_flush(f"Sequence length: {PHASE1_SEQ_LENGTH} (fixed)")
    print_flush(f"Batch size: {args.batch_size}")
    print_flush(f"Checkpoint: {config.phase1_checkpoint_path}")
    print_flush("=" * 70)

    # Load data
    input_ids = load_pile_samples(
        num_samples=args.samples,
        config=config,
        device=device,
    )

    # Create dataloader
    dataset = TensorDataset(input_ids)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create model
    print_flush("\n[Model] Creating Context-Pythia...")
    model = ContextPythiaModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        context_dim=config.context_dim,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        rotary_pct=config.rotary_pct,
    ).to(device)

    context_params = sum(p.numel() for p in model.context_block.parameters())
    print_flush(f"ContextBlock parameters: {context_params:,}")

    # Train Phase 1
    final_loss = train_phase1(model, train_loader, config, device)

    # Save checkpoint
    save_checkpoint(model, config, final_loss)

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
