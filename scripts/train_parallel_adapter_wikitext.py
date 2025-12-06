#!/usr/bin/env python3
"""
WikiText-2 Parallel Adapter Training Script

WikiText-2で訓練し、同じデータセットで評価することで公平な比較を実現。

2つの訓練方法:
1. Sliding window方式: 評価と同じ方法で訓練
2. Segment-based方式: 従来の方法（比較用）

Usage:
    # Sliding window方式（推奨）
    python3 scripts/train_parallel_adapter_wikitext.py --method sliding

    # Segment-based方式
    python3 scripts/train_parallel_adapter_wikitext.py --method segment

    # 両方試す
    python3 scripts/train_parallel_adapter_wikitext.py --method both
"""

import argparse
import sys
import time

sys.path.insert(0, ".")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from src.config.experiment_defaults import EARLY_STOPPING_PATIENCE
from src.models.infini_adapter import create_pythia_with_parallel_infini
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.training import get_device


def load_wikitext2(tokenizer, split: str = "train"):
    """WikiText-2データをロード"""
    print_flush(f"Loading WikiText-2 ({split})...")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # 全テキストを連結
    text = "\n\n".join(dataset["text"])

    # トークン化
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print_flush(f"  Total tokens: {len(tokens):,}")

    return torch.tensor(tokens, dtype=torch.long)


def evaluate_sliding_window(
    model,
    tokens: torch.Tensor,
    device: torch.device,
    context_length: int = 2048,
    stride: int = 512,
):
    """Sliding window方式でPPL評価"""
    model.eval()
    model.reset_memory()

    total_loss = 0.0
    total_tokens = 0

    tokens = tokens.to(device)
    seq_len = len(tokens)

    with torch.no_grad():
        for start in range(0, seq_len - 1, stride):
            end = min(start + context_length, seq_len)
            input_ids = tokens[start:end].unsqueeze(0)

            target_start = min(stride, end - start - 1)
            if target_start <= 0:
                continue

            labels = input_ids.clone()
            labels[0, :target_start] = -100

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            num_target_tokens = (labels != -100).sum().item()
            if num_target_tokens > 0:
                total_loss += loss.item() * num_target_tokens
                total_tokens += num_target_tokens

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


def evaluate_segment(
    model,
    tokens: torch.Tensor,
    device: torch.device,
    segment_length: int = 256,
):
    """Segment方式でPPL評価"""
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
            labels = segment[1:].unsqueeze(0)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


def train_sliding_window(
    model,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    device: torch.device,
    num_epochs: int,
    context_length: int = 2048,
    stride: int = 512,
    lr: float = 1e-4,
    patience: int = EARLY_STOPPING_PATIENCE,
):
    """Sliding window方式で訓練"""
    trainable_params = list(model.infini_adapter.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in trainable_params)
    print_flush(f"  Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")

    train_tokens = train_tokens.to(device)
    seq_len = len(train_tokens)

    best_val_ppl = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Training
        model.train()
        model.reset_memory()

        total_loss = 0.0
        total_tokens_count = 0

        for start in range(0, seq_len - 1, stride):
            end = min(start + context_length, seq_len)
            input_ids = train_tokens[start:end].unsqueeze(0)

            target_start = min(stride, end - start - 1)
            if target_start <= 0:
                continue

            labels = input_ids.clone()
            labels[0, :target_start] = -100

            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            num_target_tokens = (labels != -100).sum().item()
            if num_target_tokens > 0:
                total_loss += loss.item() * num_target_tokens
                total_tokens_count += num_target_tokens

        train_ppl = torch.exp(torch.tensor(total_loss / total_tokens_count)).item()

        # Validation
        val_ppl = evaluate_sliding_window(model, val_tokens, device, context_length, stride)

        elapsed = time.time() - start_time
        alpha = model.get_alpha()

        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            best_state = {
                k: v.cpu().clone() for k, v in model.infini_adapter.state_dict().items()
            }
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
            f"alpha={alpha:.4f} ({elapsed:.1f}s) {marker}"
        )

        if patience_counter >= patience:
            print_flush(f"  Early stopping at epoch {epoch}")
            break

    return best_val_ppl, best_state


def train_segment(
    model,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    device: torch.device,
    num_epochs: int,
    segment_length: int = 256,
    lr: float = 1e-4,
    patience: int = EARLY_STOPPING_PATIENCE,
):
    """Segment方式で訓練"""
    trainable_params = list(model.infini_adapter.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in trainable_params)
    print_flush(f"  Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")

    train_tokens = train_tokens.to(device)
    seq_len = len(train_tokens)

    best_val_ppl = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        # Training
        model.train()
        model.reset_memory()

        total_loss = 0.0
        total_tokens_count = 0

        for start in range(0, seq_len - 1, segment_length):
            end = min(start + segment_length, seq_len)
            segment = train_tokens[start:end]

            if len(segment) < 2:
                continue

            input_ids = segment[:-1].unsqueeze(0)
            labels = segment[1:].unsqueeze(0)

            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            total_loss += loss.item() * labels.numel()
            total_tokens_count += labels.numel()

        train_ppl = torch.exp(torch.tensor(total_loss / total_tokens_count)).item()

        # Validation (using same method as training)
        val_ppl = evaluate_segment(model, val_tokens, device, segment_length)

        elapsed = time.time() - start_time
        alpha = model.get_alpha()

        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            best_state = {
                k: v.cpu().clone() for k, v in model.infini_adapter.state_dict().items()
            }
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
            f"alpha={alpha:.4f} ({elapsed:.1f}s) {marker}"
        )

        if patience_counter >= patience:
            print_flush(f"  Early stopping at epoch {epoch}")
            break

    return best_val_ppl, best_state


def main():
    parser = argparse.ArgumentParser(description="WikiText-2 Parallel Adapter Training")

    parser.add_argument("--model", default="EleutherAI/pythia-70m", help="Base model")
    parser.add_argument("--method", choices=["sliding", "segment", "both"], default="both",
                        help="Training method")
    parser.add_argument("--context-length", type=int, default=2048, help="Context length for sliding")
    parser.add_argument("--stride", type=int, default=512, help="Stride for sliding window")
    parser.add_argument("--segment-length", type=int, default=256, help="Segment length")
    parser.add_argument("--epochs", type=int, default=30, help="Max epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Early stopping patience")
    parser.add_argument("--initial-alpha", type=float, default=0.0, help="Initial alpha")

    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    print_flush("=" * 70)
    print_flush("WIKITEXT-2 PARALLEL ADAPTER TRAINING")
    print_flush("=" * 70)
    print_flush(f"Model: {args.model}")
    print_flush(f"Device: {device}")
    print_flush(f"Method: {args.method}")
    print_flush(f"Epochs: {args.epochs}")
    print_flush(f"Learning rate: {args.lr}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load WikiText-2
    train_tokens = load_wikitext2(tokenizer, split="train")
    val_tokens = load_wikitext2(tokenizer, split="validation")
    test_tokens = load_wikitext2(tokenizer, split="test")

    results = {}

    # =========================================================================
    # Method 1: Sliding Window Training
    # =========================================================================
    if args.method in ["sliding", "both"]:
        print_flush("\n" + "=" * 70)
        print_flush("METHOD 1: SLIDING WINDOW TRAINING")
        print_flush("=" * 70)
        print_flush(f"Context length: {args.context_length}")
        print_flush(f"Stride: {args.stride}")

        model_sliding = create_pythia_with_parallel_infini(
            model_name=args.model,
            use_delta_rule=True,
            use_alibi=False,
            initial_alpha=args.initial_alpha,
            freeze_base_model=True,
        )
        model_sliding = model_sliding.to(device)

        # Evaluate before training
        print_flush("\nBefore training:")
        pre_ppl = evaluate_sliding_window(
            model_sliding, val_tokens, device, args.context_length, args.stride
        )
        print_flush(f"  Val PPL: {pre_ppl:.2f}")

        # Train
        print_flush("\nTraining...")
        best_ppl, best_state = train_sliding_window(
            model_sliding, train_tokens, val_tokens, device,
            num_epochs=args.epochs,
            context_length=args.context_length,
            stride=args.stride,
            lr=args.lr,
            patience=args.patience,
        )

        # Load best state
        model_sliding.infini_adapter.load_state_dict(best_state)

        # Evaluate on test set
        print_flush("\nEvaluating on test set:")
        test_ppl = evaluate_sliding_window(
            model_sliding, test_tokens, device, args.context_length, args.stride
        )
        print_flush(f"  Test PPL: {test_ppl:.2f}")
        print_flush(f"  Final alpha: {model_sliding.get_alpha():.4f}")

        # Save
        output_path = "parallel_adapter_wikitext_sliding.pt"
        torch.save({
            "infini_adapter_state_dict": best_state,
            "config": {
                "hidden_size": model_sliding.config.hidden_size,
                "num_heads": model_sliding.config.num_attention_heads,
                "use_alibi": False,
                "initial_alpha": args.initial_alpha,
                "method": "sliding",
                "context_length": args.context_length,
                "stride": args.stride,
            },
            "pre_training_ppl": pre_ppl,
            "post_training_ppl": test_ppl,
            "final_alpha": model_sliding.get_alpha(),
        }, output_path)
        print_flush(f"Saved to {output_path}")

        results["sliding"] = {
            "pre_ppl": pre_ppl,
            "test_ppl": test_ppl,
            "alpha": model_sliding.get_alpha(),
        }

        del model_sliding
        torch.cuda.empty_cache()

    # =========================================================================
    # Method 2: Segment-based Training
    # =========================================================================
    if args.method in ["segment", "both"]:
        print_flush("\n" + "=" * 70)
        print_flush("METHOD 2: SEGMENT-BASED TRAINING")
        print_flush("=" * 70)
        print_flush(f"Segment length: {args.segment_length}")

        model_segment = create_pythia_with_parallel_infini(
            model_name=args.model,
            use_delta_rule=True,
            use_alibi=False,
            initial_alpha=args.initial_alpha,
            freeze_base_model=True,
        )
        model_segment = model_segment.to(device)

        # Evaluate before training (use sliding window for fair comparison)
        print_flush("\nBefore training:")
        pre_ppl_sliding = evaluate_sliding_window(
            model_segment, val_tokens, device, args.context_length, args.stride
        )
        pre_ppl_segment = evaluate_segment(
            model_segment, val_tokens, device, args.segment_length
        )
        print_flush(f"  Val PPL (sliding): {pre_ppl_sliding:.2f}")
        print_flush(f"  Val PPL (segment): {pre_ppl_segment:.2f}")

        # Train
        print_flush("\nTraining...")
        best_ppl, best_state = train_segment(
            model_segment, train_tokens, val_tokens, device,
            num_epochs=args.epochs,
            segment_length=args.segment_length,
            lr=args.lr,
            patience=args.patience,
        )

        # Load best state
        model_segment.infini_adapter.load_state_dict(best_state)

        # Evaluate on test set (both methods)
        print_flush("\nEvaluating on test set:")
        test_ppl_sliding = evaluate_sliding_window(
            model_segment, test_tokens, device, args.context_length, args.stride
        )
        test_ppl_segment = evaluate_segment(
            model_segment, test_tokens, device, args.segment_length
        )
        print_flush(f"  Test PPL (sliding): {test_ppl_sliding:.2f}")
        print_flush(f"  Test PPL (segment): {test_ppl_segment:.2f}")
        print_flush(f"  Final alpha: {model_segment.get_alpha():.4f}")

        # Save
        output_path = "parallel_adapter_wikitext_segment.pt"
        torch.save({
            "infini_adapter_state_dict": best_state,
            "config": {
                "hidden_size": model_segment.config.hidden_size,
                "num_heads": model_segment.config.num_attention_heads,
                "use_alibi": False,
                "initial_alpha": args.initial_alpha,
                "method": "segment",
                "segment_length": args.segment_length,
            },
            "pre_training_ppl": pre_ppl_sliding,
            "post_training_ppl": test_ppl_sliding,
            "final_alpha": model_segment.get_alpha(),
        }, output_path)
        print_flush(f"Saved to {output_path}")

        results["segment"] = {
            "pre_ppl": pre_ppl_sliding,
            "test_ppl_sliding": test_ppl_sliding,
            "test_ppl_segment": test_ppl_segment,
            "alpha": model_segment.get_alpha(),
        }

        del model_segment
        torch.cuda.empty_cache()

    # =========================================================================
    # Summary
    # =========================================================================
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)
    print_flush("| Method | Pre-PPL | Test PPL (sliding) | Alpha |")
    print_flush("|--------|---------|-------------------|-------|")
    print_flush(f"| Original Pythia | 40.96 | 40.96 | - |")

    if "sliding" in results:
        r = results["sliding"]
        print_flush(f"| Sliding window train | {r['pre_ppl']:.2f} | {r['test_ppl']:.2f} | {r['alpha']:.4f} |")

    if "segment" in results:
        r = results["segment"]
        print_flush(f"| Segment-based train | {r['pre_ppl']:.2f} | {r['test_ppl_sliding']:.2f} | {r['alpha']:.4f} |")

    print_flush("\n" + "=" * 70)
    print_flush("ANALYSIS")
    print_flush("=" * 70)

    baseline_ppl = 40.96

    if "sliding" in results:
        r = results["sliding"]
        if r["test_ppl"] < baseline_ppl:
            print_flush(f"✓ Sliding window training IMPROVED by {baseline_ppl - r['test_ppl']:.2f}")
        else:
            print_flush(f"⚠️ Sliding window training DEGRADED by {r['test_ppl'] - baseline_ppl:.2f}")

    if "segment" in results:
        r = results["segment"]
        if r["test_ppl_sliding"] < baseline_ppl:
            print_flush(f"✓ Segment training IMPROVED by {baseline_ppl - r['test_ppl_sliding']:.2f}")
        else:
            print_flush(f"⚠️ Segment training DEGRADED by {r['test_ppl_sliding'] - baseline_ppl:.2f}")

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
