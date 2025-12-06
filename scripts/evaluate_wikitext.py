#!/usr/bin/env python3
"""
WikiText-2 PPL Evaluation Script

標準的なベンチマークであるWikiText-2でPPLを評価。
Pythia-70mの公式PPLは約32-35程度。

Usage:
    python3 scripts/evaluate_wikitext.py
"""

import argparse
import sys

sys.path.insert(0, ".")

import torch
from transformers import GPTNeoXForCausalLM

from src.utils.data_loading import load_wikitext2
from src.utils.evaluation import evaluate_ppl_segment, evaluate_ppl_sliding_window
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.tokenizer_utils import get_tokenizer
from src.utils.training import get_device


def main():
    parser = argparse.ArgumentParser(description="WikiText-2 PPL Evaluation")

    parser.add_argument("--model", default="EleutherAI/pythia-70m", help="Model name")
    parser.add_argument("--context-length", type=int, default=2048, help="Context length")
    parser.add_argument("--stride", type=int, default=512, help="Stride for sliding window")

    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    print_flush("=" * 70)
    print_flush("WIKITEXT-2 PPL EVALUATION")
    print_flush("=" * 70)
    print_flush(f"Model: {args.model}")
    print_flush(f"Device: {device}")
    print_flush(f"Context length: {args.context_length}")
    print_flush(f"Stride: {args.stride}")

    # Load tokenizer and data
    tokenizer = get_tokenizer(args.model)
    tokens = load_wikitext2(tokenizer, split="test")

    # =====================================================================
    # Evaluate Original Pythia-70m
    # =====================================================================
    print_flush("\n" + "=" * 70)
    print_flush("ORIGINAL PYTHIA-70M")
    print_flush("=" * 70)

    print_flush("\nLoading original Pythia-70m...")
    original_model = GPTNeoXForCausalLM.from_pretrained(args.model)
    original_model = original_model.to(device)
    original_model.eval()

    # Method 1: Sliding window (standard)
    print_flush("\n1. Sliding window evaluation (standard method):")
    ppl_sliding = evaluate_ppl_sliding_window(
        original_model, tokens, device,
        context_length=args.context_length,
        stride=args.stride,
    )
    print_flush(f"   PPL: {ppl_sliding:.2f}")

    # Method 2: Simple non-overlapping
    print_flush("\n2. Simple non-overlapping evaluation:")
    ppl_simple = evaluate_ppl_segment(
        original_model, tokens, device,
        segment_length=args.context_length,
    )
    print_flush(f"   PPL: {ppl_simple:.2f}")

    # Method 3: Segment-based (like training)
    print_flush("\n3. Segment-based evaluation (256 tokens):")
    ppl_segment = evaluate_ppl_segment(
        original_model, tokens, device,
        segment_length=256,
    )
    print_flush(f"   PPL: {ppl_segment:.2f}")

    # Clean up
    del original_model
    torch.cuda.empty_cache()

    # =====================================================================
    # Summary
    # =====================================================================
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)
    print_flush("| Method | PPL |")
    print_flush("|--------|-----|")
    print_flush(f"| Pythia-70m (sliding window, stride={args.stride}) | {ppl_sliding:.2f} |")
    print_flush(f"| Pythia-70m (simple, {args.context_length} tokens) | {ppl_simple:.2f} |")
    print_flush(f"| Pythia-70m (segment, 256 tokens) | {ppl_segment:.2f} |")

    print_flush("\n" + "=" * 70)
    print_flush("REFERENCE")
    print_flush("=" * 70)
    print_flush("Expected Pythia-70m WikiText-2 PPL: ~32-35")
    print_flush("(See: https://huggingface.co/EleutherAI/pythia-70m)")

    if ppl_sliding > 50:
        print_flush("\nWARNING: PPL is higher than expected. Check evaluation method.")
    else:
        print_flush(f"\nPPL {ppl_sliding:.2f} is in expected range")

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
