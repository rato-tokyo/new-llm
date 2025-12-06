#!/usr/bin/env python3
"""
Infini-Pythia WikiText-2 Training Script

Layer 0をInfini-Attentionに置き換えたPythiaモデルをWikiText-2でスクラッチ訓練。

Usage:
    python3 scripts/train_infini_pythia_wikitext.py --epochs 30
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

import torch
import torch.nn as nn

from config.pythia import PythiaConfig
from src.config.experiment_defaults import EARLY_STOPPING_PATIENCE
from src.models.infini_pythia import InfiniPythiaModel
from src.utils.data_loading import load_wikitext2
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.tokenizer_utils import get_tokenizer
from src.utils.training import get_device


def evaluate_ppl(
    model: nn.Module,
    tokens: torch.Tensor,
    device: torch.device,
    segment_length: int = 256,
) -> float:
    """セグメント分割でPPL評価"""
    model.eval()
    model.reset_memory()

    total_loss = 0.0
    total_tokens = 0

    tokens = tokens.to(device)
    seq_len = len(tokens)

    with torch.no_grad():
        for start in range(0, seq_len - 1, segment_length):
            end = min(start + segment_length, seq_len)
            segment = tokens[start:end]

            if len(segment) < 2:
                continue

            input_ids = segment[:-1].unsqueeze(0)
            labels = segment[1:]

            logits = model(input_ids)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += labels.numel()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


def train_epoch(
    model: nn.Module,
    tokens: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    segment_length: int = 256,
) -> float:
    """1エポック訓練"""
    model.train()
    model.reset_memory()

    total_loss = 0.0
    total_tokens = 0

    tokens = tokens.to(device)
    seq_len = len(tokens)

    for start in range(0, seq_len - 1, segment_length):
        end = min(start + segment_length, seq_len)
        segment = tokens[start:end]

        if len(segment) < 2:
            continue

        input_ids = segment[:-1].unsqueeze(0)
        labels = segment[1:]

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.numel()
        total_tokens += labels.numel()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


def main():
    parser = argparse.ArgumentParser(description="Infini-Pythia WikiText-2 Training")

    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--segment-length", type=int, default=256, help="Segment length")
    parser.add_argument(
        "--patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Early stopping patience"
    )
    parser.add_argument("--output", default="infini_pythia_wikitext.pt", help="Output path")

    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    print_flush("=" * 70)
    print_flush("INFINI-PYTHIA WIKITEXT-2 TRAINING")
    print_flush("=" * 70)
    print_flush(f"Device: {device}")
    print_flush(f"Epochs: {args.epochs}")
    print_flush(f"Learning rate: {args.lr}")
    print_flush(f"Segment length: {args.segment_length}")

    # Load tokenizer and data
    tokenizer = get_tokenizer("EleutherAI/pythia-70m")

    print_flush("\nLoading WikiText-2...")
    train_tokens = load_wikitext2(tokenizer, split="train")
    val_tokens = load_wikitext2(tokenizer, split="validation")

    print_flush(f"Train tokens: {len(train_tokens):,}")
    print_flush(f"Val tokens: {len(val_tokens):,}")

    # Create model
    print_flush("\nCreating Infini-Pythia model...")
    config = PythiaConfig()
    model = InfiniPythiaModel(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        num_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        rotary_pct=config.rotary_pct,
        use_delta_rule=True,
    )
    model = model.to(device)

    # Count parameters
    param_info = model.num_parameters()
    print_flush(f"Total parameters: {param_info['total']:,}")
    print_flush(f"  Infini Layer: {param_info['infini_layer']:,}")
    print_flush(f"  Pythia Layers: {param_info['pythia_layers']:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Evaluate before training
    print_flush("\nEvaluating before training...")
    pre_ppl = evaluate_ppl(model, val_tokens, device, args.segment_length)
    print_flush(f"  Val PPL (before): {pre_ppl:.1f}")

    # Training
    print_flush("\n" + "=" * 70)
    print_flush("Training...")
    print_flush("=" * 70)

    best_val_ppl = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_ppl = train_epoch(
            model, train_tokens, optimizer, device, args.segment_length
        )

        # Evaluate
        val_ppl = evaluate_ppl(model, val_tokens, device, args.segment_length)

        elapsed = time.time() - start_time

        # Check for improvement
        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
            f"({elapsed:.1f}s) {marker}"
        )

        # Early stopping
        if patience_counter >= args.patience:
            print_flush(f"  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # Load best weights
    model.load_state_dict(best_state)

    # Evaluate after training
    print_flush("\nEvaluating after training...")
    post_ppl = evaluate_ppl(model, val_tokens, device, args.segment_length)
    print_flush(f"  Val PPL (after): {post_ppl:.1f}")

    # Save
    print_flush(f"\nSaving to {args.output}...")
    save_dict = {
        "model_state_dict": best_state,
        "config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_layers,
            "num_heads": config.num_attention_heads,
            "intermediate_size": config.intermediate_size,
        },
        "pre_training_ppl": pre_ppl,
        "post_training_ppl": post_ppl,
        "best_val_ppl": best_val_ppl,
    }
    torch.save(save_dict, args.output)

    # Summary
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)
    print_flush("| Metric | Value |")
    print_flush("|--------|-------|")
    print_flush(f"| PPL (before training) | {pre_ppl:.1f} |")
    print_flush(f"| PPL (after training) | {post_ppl:.1f} |")
    print_flush(f"| Best Val PPL | {best_val_ppl:.1f} |")

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
