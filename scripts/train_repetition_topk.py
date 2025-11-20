#!/usr/bin/env python3
"""CVFPT Performance Evaluation on Top-K Frequent Tokens

This script evaluates Context Vector Fixed Point Training (CVFPT) performance
on the most frequent tokens in a dataset.

Purpose:
    - Measure convergence speed for different tokens
    - Identify which tokens converge faster/slower
    - Establish baseline CVFPT performance metrics

Strategy:
    - Extract top-K most frequent tokens from WikiText dataset
    - Train CVFPT on each token individually
    - Record convergence loss and training time
    - Generate performance report
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.new_llm import NewLLM
from src.utils.config import NewLLMConfig
from src.data.repetition_dataset import RepetitionDataset, collate_fn
from src.training.convergence_loss import (
    ContextConvergenceLoss,
    compute_context_change_rate,
    compute_convergence_metric
)
from transformers import GPT2Tokenizer


def extract_top_k_tokens(
    k: int = 100,
    tokenizer=None,
    start_id: int = 0
) -> list:
    """Extract top-K tokens from tokenizer vocabulary

    Args:
        k: Number of tokens to extract
        tokenizer: Tokenizer to use
        start_id: Starting token ID (default: 0)

    Returns:
        List of (token_id, token_text) tuples
    """
    print(f"\n📊 Extracting top-{k} tokens from tokenizer vocabulary...")
    print(f"Token ID range: {start_id} to {start_id + k - 1}")

    # Simply get tokens by ID
    top_k_tokens = []
    for token_id in range(start_id, start_id + k):
        if token_id < len(tokenizer):
            token_text = tokenizer.decode([token_id])
            top_k_tokens.append((token_id, token_text))
        else:
            break

    print(f"\n✓ {len(top_k_tokens)} tokens extracted")
    print(f"Sample tokens: {top_k_tokens[:10]}")

    return top_k_tokens


def train_single_token_cvfpt(
    token_text: str,
    token_id: int,
    model: nn.Module,
    tokenizer,
    device: torch.device,
    epochs: int = 5,
    repetitions: int = 50,
    batch_size: int = 4,
    lr: float = 0.001,
    max_length: int = 512
) -> dict:
    """Train CVFPT on a single token and measure performance

    Args:
        token_text: Token text (for display)
        token_id: Token ID
        model: New-LLM model
        tokenizer: Tokenizer
        device: Device
        epochs: Number of epochs
        repetitions: Number of repetitions
        batch_size: Batch size
        lr: Learning rate
        max_length: Maximum sequence length

    Returns:
        Performance metrics dictionary
    """
    # Create dataset for this single token
    dataset = RepetitionDataset(
        phrases=[token_text],
        repetitions=repetitions,
        tokenizer=tokenizer,
        max_length=max_length
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Loss function
    loss_fn = ContextConvergenceLoss(
        cycle_length=1,  # Single token repetition
        convergence_weight=1.0,
        token_weight=0.0
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    model.train()
    epoch_losses = []
    epoch_change_rates = []
    epoch_times = []

    start_time = time.time()

    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_change = 0.0

        for input_ids in train_loader:
            input_ids = input_ids.to(device)

            optimizer.zero_grad()
            logits, context_trajectory = model(input_ids)

            loss, metrics = loss_fn(
                context_vectors=context_trajectory,
                logits=logits,
                targets=input_ids
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += metrics['loss']
            change_rate = compute_context_change_rate(context_trajectory)
            epoch_change += change_rate

        # Average metrics
        avg_loss = epoch_loss / len(train_loader)
        avg_change = epoch_change / len(train_loader)
        epoch_time = time.time() - epoch_start

        epoch_losses.append(avg_loss)
        epoch_change_rates.append(avg_change)
        epoch_times.append(epoch_time)

    total_time = time.time() - start_time

    # Performance metrics
    metrics = {
        'token_text': token_text,
        'token_id': token_id,
        'initial_loss': epoch_losses[0],
        'final_loss': epoch_losses[-1],
        'loss_reduction': epoch_losses[0] - epoch_losses[-1],
        'improvement_rate': (epoch_losses[0] - epoch_losses[-1]) / epoch_losses[0] if epoch_losses[0] > 0 else 0,
        'initial_change_rate': epoch_change_rates[0],
        'final_change_rate': epoch_change_rates[-1],
        'total_time': total_time,
        'avg_epoch_time': total_time / epochs,
        'converged': epoch_losses[-1] < 0.01,  # Convergence threshold
        'epoch_losses': epoch_losses,
        'epoch_change_rates': epoch_change_rates
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="CVFPT performance evaluation on top-K tokens")

    # Token selection
    parser.add_argument('--top-k', type=int, default=20,
                       help='Number of top tokens to evaluate (by token ID)')
    parser.add_argument('--start-id', type=int, default=0,
                       help='Starting token ID (default: 0)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs per token')
    parser.add_argument('--repetitions', type=int, default=50,
                       help='Number of repetitions per token')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')

    # Model parameters
    parser.add_argument('--context-dim', type=int, default=256,
                       help='Context vector dimension')
    parser.add_argument('--embed-dim', type=int, default=256,
                       help='Token embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='Hidden dimension')
    parser.add_argument('--layers', type=int, default=2,
                       help='Number of FNN layers')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='checkpoints/cvfpt_topk',
                       help='Output directory')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')

    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ========== Initialize Tokenizer ==========
    print("\n📚 Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print(f"✓ Tokenizer loaded: {len(tokenizer):,} tokens")

    # ========== Extract Top-K Tokens ==========
    top_k_tokens = extract_top_k_tokens(
        k=args.top_k,
        tokenizer=tokenizer,
        start_id=args.start_id
    )

    # ========== Initialize Model ==========
    print("\n🏗️  Building New-LLM model...")
    config = NewLLMConfig()
    config.vocab_size = len(tokenizer)
    config.context_vector_dim = args.context_dim
    config.embed_dim = args.embed_dim
    config.hidden_dim = args.hidden_dim
    config.num_layers = args.layers
    config.max_seq_length = args.max_length

    model = NewLLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params:,} parameters")

    # ========== Evaluate Each Token ==========
    print(f"\n{'='*80}")
    print("🚀 Starting CVFPT Performance Evaluation")
    print(f"{'='*80}")
    print(f"Top-K Tokens: {args.top_k}")
    print(f"Epochs per Token: {args.epochs}")
    print(f"Repetitions: {args.repetitions}")
    print(f"{'='*80}\n")

    all_results = []

    for idx, (token_id, token_text) in enumerate(top_k_tokens):
        print(f"\n{'='*80}")
        print(f"Token {idx+1}/{args.top_k}: '{token_text}' (ID: {token_id})")
        print(f"{'='*80}")

        # Re-initialize model for each token (fresh start)
        model = NewLLM(config).to(device)

        # Train CVFPT on this token
        try:
            metrics = train_single_token_cvfpt(
                token_text=token_text,
                token_id=token_id,
                model=model,
                tokenizer=tokenizer,
                device=device,
                epochs=args.epochs,
                repetitions=args.repetitions,
                batch_size=args.batch_size,
                lr=args.lr,
                max_length=args.max_length
            )

            all_results.append(metrics)

            # Print summary
            print(f"\n📊 Results for '{token_text}':")
            print(f"  Initial Loss: {metrics['initial_loss']:.6f}")
            print(f"  Final Loss: {metrics['final_loss']:.6f}")
            print(f"  Improvement: {metrics['improvement_rate']*100:.1f}%")
            print(f"  Converged: {'✓ Yes' if metrics['converged'] else '✗ No'}")
            print(f"  Time: {metrics['total_time']:.2f}s")

        except Exception as e:
            print(f"✗ Error training token '{token_text}': {e}")
            continue

    # ========== Generate Report ==========
    print(f"\n{'='*80}")
    print("📊 CVFPT Performance Report")
    print(f"{'='*80}\n")

    # Sort by improvement rate
    all_results.sort(key=lambda x: x['improvement_rate'], reverse=True)

    print("Top 10 Best Converging Tokens:")
    print(f"{'Rank':<6} {'Token':<20} {'Initial':<10} {'Final':<10} {'Improve%':<10} {'Time(s)':<10}")
    print("-" * 80)
    for i, result in enumerate(all_results[:10]):
        print(f"{i+1:<6} {result['token_text']:<20} "
              f"{result['initial_loss']:<10.6f} {result['final_loss']:<10.6f} "
              f"{result['improvement_rate']*100:<10.1f} {result['total_time']:<10.2f}")

    print("\n" + "="*80)
    print("Bottom 10 Worst Converging Tokens:")
    print(f"{'Rank':<6} {'Token':<20} {'Initial':<10} {'Final':<10} {'Improve%':<10} {'Time(s)':<10}")
    print("-" * 80)
    for i, result in enumerate(all_results[-10:]):
        print(f"{i+1:<6} {result['token_text']:<20} "
              f"{result['initial_loss']:<10.6f} {result['final_loss']:<10.6f} "
              f"{result['improvement_rate']*100:<10.1f} {result['total_time']:<10.2f}")

    # Statistics
    avg_improvement = sum(r['improvement_rate'] for r in all_results) / len(all_results)
    convergence_rate = sum(1 for r in all_results if r['converged']) / len(all_results)
    avg_time = sum(r['total_time'] for r in all_results) / len(all_results)

    print("\n" + "="*80)
    print("Overall Statistics:")
    print(f"  Average Improvement: {avg_improvement*100:.1f}%")
    print(f"  Convergence Rate: {convergence_rate*100:.1f}% ({sum(1 for r in all_results if r['converged'])}/{len(all_results)})")
    print(f"  Average Time per Token: {avg_time:.2f}s")
    print("="*80)

    # Save results
    results_file = os.path.join(args.output_dir, 'cvfpt_performance_topk.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
