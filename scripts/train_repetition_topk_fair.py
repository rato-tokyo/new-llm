#!/usr/bin/env python3
"""CVFPT Performance Evaluation on Top-K Tokens (Fair Round-Robin Version)

This script evaluates CVFPT performance with fair training:
- All tokens share the same model initialization
- Training proceeds in round-robin fashion (token1 → token2 → ... → tokenN)
- Each epoch, all tokens are trained once before moving to next epoch
- Eliminates bias from sequential training

Comparison:
    Sequential (biased):
        Token 1: fully train epochs 1-3 with init_1
        Token 2: fully train epochs 1-3 with init_2
        ...

    Round-Robin (fair):
        Epoch 1: Token 1, Token 2, ..., Token N (all with same model state)
        Epoch 2: Token 1, Token 2, ..., Token N (all with same model state)
        Epoch 3: Token 1, Token 2, ..., Token N (all with same model state)
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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


def extract_top_k_tokens(k: int, tokenizer=None, start_id: int = 0) -> list:
    """Extract top-K tokens from tokenizer vocabulary"""
    print(f"\n📊 Extracting top-{k} tokens from tokenizer vocabulary...")
    print(f"Token ID range: {start_id} to {start_id + k - 1}")

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


def create_dataloaders_for_all_tokens(
    tokens: list,
    tokenizer,
    repetitions: int,
    batch_size: int,
    max_length: int
) -> dict:
    """Create DataLoaders for all tokens"""
    print(f"\n🔧 Creating DataLoaders for {len(tokens)} tokens...")

    dataloaders = {}
    for token_id, token_text in tqdm(tokens, desc="Building DataLoaders"):
        dataset = RepetitionDataset(
            phrases=[token_text],
            repetitions=repetitions,
            tokenizer=tokenizer,
            max_length=max_length
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

        dataloaders[token_id] = {
            'text': token_text,
            'dataloader': dataloader,
            'epoch_losses': [],
            'epoch_change_rates': []
        }

    print(f"✓ DataLoaders ready for {len(dataloaders)} tokens")
    return dataloaders


def train_all_tokens_fair(
    model: nn.Module,
    dataloaders: dict,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int
) -> dict:
    """Train all tokens in fair round-robin fashion

    Each epoch:
        - Train token 1 for 1 batch
        - Train token 2 for 1 batch
        - ...
        - Train token N for 1 batch

    This ensures all tokens are trained with the same model state.
    """
    model.train()

    print(f"\n{'='*80}")
    print(f"🎯 Fair Round-Robin Training")
    print(f"{'='*80}")
    print(f"Tokens: {len(dataloaders)}")
    print(f"Epochs: {epochs}")
    print(f"Strategy: Train all tokens equally each epoch")
    print(f"{'='*80}\n")

    # Track initial losses (epoch 0 - before training)
    print("📊 Recording initial losses...")
    with torch.no_grad():
        for token_id, data in tqdm(dataloaders.items(), desc="Initial evaluation"):
            dataloader = data['dataloader']
            batch = next(iter(dataloader))
            input_ids = batch.to(device)

            logits, context_trajectory = model(input_ids)
            loss, metrics = loss_fn(
                context_vectors=context_trajectory,
                logits=logits,
                targets=input_ids
            )

            data['epoch_losses'].append(metrics['loss'])
            change_rate = compute_context_change_rate(context_trajectory)
            data['epoch_change_rates'].append(change_rate)

    # Training loop (fair round-robin)
    for epoch in range(epochs):
        epoch_start = time.time()

        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*80}")

        # Train each token once per epoch
        progress_bar = tqdm(
            dataloaders.items(),
            desc=f"Epoch {epoch+1}/{epochs}",
            leave=True
        )

        for token_id, data in progress_bar:
            token_text = data['text']
            dataloader = data['dataloader']

            # Train on one batch
            batch = next(iter(dataloader))
            input_ids = batch.to(device)

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

            # Record metrics
            data['epoch_losses'].append(metrics['loss'])
            change_rate = compute_context_change_rate(context_trajectory)
            data['epoch_change_rates'].append(change_rate)

            # Update progress bar
            progress_bar.set_postfix({
                'token': token_text[:10],
                'loss': f"{metrics['loss']:.4f}",
                'change': f"{change_rate:.4f}"
            })

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

    # Compile results
    results = []
    for token_id, data in dataloaders.items():
        initial_loss = data['epoch_losses'][0]
        final_loss = data['epoch_losses'][-1]

        result = {
            'token_id': token_id,
            'token_text': data['text'],
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'loss_reduction': initial_loss - final_loss,
            'improvement_rate': (initial_loss - final_loss) / initial_loss if initial_loss > 0 else 0,
            'initial_change_rate': data['epoch_change_rates'][0],
            'final_change_rate': data['epoch_change_rates'][-1],
            'converged': final_loss < 0.01,
            'epoch_losses': data['epoch_losses'],
            'epoch_change_rates': data['epoch_change_rates']
        }
        results.append(result)

    return results


def main():
    parser = argparse.ArgumentParser(description="Fair CVFPT performance evaluation")

    # Token selection
    parser.add_argument('--top-k', type=int, default=20,
                       help='Number of tokens to evaluate')
    parser.add_argument('--start-id', type=int, default=0,
                       help='Starting token ID')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of epochs')
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
    parser.add_argument('--output-dir', type=str, default='checkpoints/cvfpt_topk_fair',
                       help='Output directory')
    parser.add_argument('--max-length', type=int, default=512,
                       help='Maximum sequence length')

    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize Tokenizer
    print("\n📚 Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print(f"✓ Tokenizer loaded: {len(tokenizer):,} tokens")

    # Extract Top-K Tokens
    top_k_tokens = extract_top_k_tokens(
        k=args.top_k,
        tokenizer=tokenizer,
        start_id=args.start_id
    )

    # Initialize Model (ONCE - shared by all tokens)
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
    print(f"✓ Shared by all {args.top_k} tokens (fair comparison)")

    # Create DataLoaders for all tokens
    dataloaders = create_dataloaders_for_all_tokens(
        tokens=top_k_tokens,
        tokenizer=tokenizer,
        repetitions=args.repetitions,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # Loss function
    # CRITICAL: token_weight=0.01 prevents degenerate solution where model stops updating context
    # Without reconstruction loss, model can achieve low convergence loss by simply not updating context
    loss_fn = ContextConvergenceLoss(
        cycle_length=1,
        convergence_weight=1.0,
        token_weight=0.01  # Reconstruction loss (prevents degenerate solution)
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Fair Round-Robin Training
    print(f"\n{'='*80}")
    print("🚀 Starting Fair CVFPT Performance Evaluation")
    print(f"{'='*80}")
    print(f"Top-K Tokens: {args.top_k}")
    print(f"Epochs: {args.epochs}")
    print(f"Repetitions: {args.repetitions}")
    print(f"Training Strategy: Round-Robin (fair)")
    print(f"{'='*80}\n")

    start_time = time.time()

    all_results = train_all_tokens_fair(
        model=model,
        dataloaders=dataloaders,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs
    )

    total_time = time.time() - start_time

    # Generate Report
    print(f"\n{'='*80}")
    print("📊 Fair CVFPT Performance Report")
    print(f"{'='*80}\n")

    # Sort by improvement rate
    all_results.sort(key=lambda x: x['improvement_rate'], reverse=True)

    print("Top 10 Best Converging Tokens:")
    print(f"{'Rank':<6} {'Token':<20} {'Initial':<10} {'Final':<10} {'Improve%':<10}")
    print("-" * 80)
    for i, result in enumerate(all_results[:10]):
        print(f"{i+1:<6} {result['token_text']:<20} "
              f"{result['initial_loss']:<10.6f} {result['final_loss']:<10.6f} "
              f"{result['improvement_rate']*100:<10.1f}")

    print("\n" + "="*80)
    print("Bottom 10 Worst Converging Tokens:")
    print(f"{'Rank':<6} {'Token':<20} {'Initial':<10} {'Final':<10} {'Improve%':<10}")
    print("-" * 80)
    for i, result in enumerate(all_results[-10:]):
        print(f"{i+1:<6} {result['token_text']:<20} "
              f"{result['initial_loss']:<10.6f} {result['final_loss']:<10.6f} "
              f"{result['improvement_rate']*100:<10.1f}")

    # Statistics
    avg_improvement = sum(r['improvement_rate'] for r in all_results) / len(all_results)
    convergence_rate = sum(1 for r in all_results if r['converged']) / len(all_results)

    print("\n" + "="*80)
    print("Overall Statistics:")
    print(f"  Average Improvement: {avg_improvement*100:.1f}%")
    print(f"  Convergence Rate: {convergence_rate*100:.1f}% ({sum(1 for r in all_results if r['converged'])}/{len(all_results)})")
    print(f"  Total Time: {total_time:.2f}s")
    print(f"  Time per Token: {total_time/len(all_results):.2f}s")
    print("="*80)

    # Save results
    results_file = os.path.join(args.output_dir, 'cvfpt_performance_topk_fair.json')
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {results_file}")


if __name__ == "__main__":
    main()
