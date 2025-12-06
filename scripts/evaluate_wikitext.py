#!/usr/bin/env python3
"""
WikiText-2 PPL Evaluation Script

標準的なベンチマークであるWikiText-2でPPLを評価。
Pythia-70mの公式PPLは約32-35程度。

Usage:
    python3 scripts/evaluate_wikitext.py
    python3 scripts/evaluate_wikitext.py --parallel-adapter parallel_adapter.pt
"""

import argparse
import sys

sys.path.insert(0, ".")

import torch
from transformers import GPTNeoXForCausalLM

from src.models.infini_adapter import create_pythia_with_parallel_infini
from src.utils.data_loading import load_wikitext2
from src.utils.evaluation import evaluate_ppl_segment, evaluate_ppl_sliding_window
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.tokenizer_utils import get_tokenizer
from src.utils.training import get_device


def main():
    parser = argparse.ArgumentParser(description="WikiText-2 PPL Evaluation")

    parser.add_argument("--model", default="EleutherAI/pythia-70m", help="Model name")
    parser.add_argument("--parallel-adapter", type=str, help="Path to parallel adapter checkpoint")
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

    # Method 3: Segment-based (like parallel adapter training)
    print_flush("\n3. Segment-based evaluation (256 tokens, like training):")
    ppl_segment = evaluate_ppl_segment(
        original_model, tokens, device,
        segment_length=256,
    )
    print_flush(f"   PPL: {ppl_segment:.2f}")

    # Clean up
    del original_model
    torch.cuda.empty_cache()

    # =====================================================================
    # Evaluate Parallel Adapter (if provided)
    # =====================================================================
    if args.parallel_adapter:
        print_flush("\n" + "=" * 70)
        print_flush("PARALLEL ADAPTER")
        print_flush("=" * 70)

        print_flush(f"\nLoading parallel adapter from {args.parallel_adapter}...")
        checkpoint = torch.load(args.parallel_adapter, map_location=device)

        # Create model
        adapter_model = create_pythia_with_parallel_infini(
            model_name=args.model,
            use_delta_rule=True,
            initial_alpha=checkpoint["config"].get("initial_alpha", 0.0),
            freeze_base_model=True,
        )
        adapter_model = adapter_model.to(device)

        # Load weights
        adapter_model.infini_adapter.load_state_dict(checkpoint["infini_adapter_state_dict"])
        print_flush(f"  Loaded alpha: {adapter_model.get_alpha():.4f}")

        # Evaluate with Sliding window (same as baseline)
        print_flush("\n4. Parallel Adapter - Sliding window evaluation:")
        adapter_model.reset_memory()
        ppl_adapter_sliding = evaluate_ppl_sliding_window(
            adapter_model, tokens, device,
            context_length=args.context_length,
            stride=args.stride,
        )
        print_flush(f"   PPL: {ppl_adapter_sliding:.2f}")

        # Evaluate with memory (segment-based, like training)
        print_flush("\n5. Parallel Adapter - Segment-based (256 tokens, like training):")
        ppl_adapter_segment = evaluate_ppl_segment(
            adapter_model, tokens, device,
            segment_length=256,
            reset_memory_fn=adapter_model.reset_memory,
        )
        print_flush(f"   PPL: {ppl_adapter_segment:.2f}")

        # Also evaluate with alpha=0 (baseline check)
        print_flush("\n6. Parallel Adapter with alpha=0 - Sliding window (baseline check):")
        original_alpha = adapter_model.get_alpha()
        adapter_model.set_alpha(0.0)
        adapter_model.reset_memory()
        ppl_alpha0 = evaluate_ppl_sliding_window(
            adapter_model, tokens, device,
            context_length=args.context_length,
            stride=args.stride,
        )
        print_flush(f"   PPL: {ppl_alpha0:.2f}")
        adapter_model.set_alpha(original_alpha)

        del adapter_model
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
    if args.parallel_adapter:
        print_flush(f"| Parallel Adapter (sliding window) | {ppl_adapter_sliding:.2f} |")
        print_flush(f"| Parallel Adapter (segment, 256) | {ppl_adapter_segment:.2f} |")
        print_flush(f"| Parallel Adapter (alpha=0, sliding) | {ppl_alpha0:.2f} |")

    print_flush("\n" + "=" * 70)
    print_flush("REFERENCE")
    print_flush("=" * 70)
    print_flush("Expected Pythia-70m WikiText-2 PPL: ~32-35")
    print_flush("(See: https://huggingface.co/EleutherAI/pythia-70m)")

    if ppl_sliding > 50:
        print_flush("\n⚠️ WARNING: PPL is higher than expected. Check evaluation method.")
    else:
        print_flush(f"\n✓ PPL {ppl_sliding:.2f} is in expected range")

    # =====================================================================
    # Analysis
    # =====================================================================
    if args.parallel_adapter:
        print_flush("\n" + "=" * 70)
        print_flush("ANALYSIS")
        print_flush("=" * 70)

        # Compare sliding window results
        print_flush("\nSliding window comparison:")
        print_flush(f"  Original Pythia:    {ppl_sliding:.2f}")
        print_flush(f"  Parallel Adapter:   {ppl_adapter_sliding:.2f}")
        print_flush(f"  Adapter (alpha=0):  {ppl_alpha0:.2f}")

        if abs(ppl_alpha0 - ppl_sliding) < 1.0:
            print_flush("  ✓ alpha=0 matches baseline (sanity check passed)")
        else:
            print_flush(f"  ⚠️ alpha=0 differs from baseline by {abs(ppl_alpha0 - ppl_sliding):.2f}")

        if ppl_adapter_sliding < ppl_sliding:
            improvement = ppl_sliding - ppl_adapter_sliding
            print_flush(f"\n  ✓ Parallel Adapter IMPROVED PPL by {improvement:.2f}")
        else:
            degradation = ppl_adapter_sliding - ppl_sliding
            print_flush(f"\n  ⚠️ Parallel Adapter DEGRADED PPL by {degradation:.2f}")

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
