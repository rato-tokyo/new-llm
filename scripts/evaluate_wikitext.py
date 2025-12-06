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
from datasets import load_dataset
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from src.models.infini_adapter import create_pythia_with_parallel_infini
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.training import get_device


def load_wikitext2(tokenizer, split: str = "test"):
    """WikiText-2データをロード"""
    print_flush(f"Loading WikiText-2 ({split})...")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # 全テキストを連結
    text = "\n\n".join(dataset["text"])

    # トークン化
    tokens = tokenizer.encode(text, add_special_tokens=False)
    print_flush(f"  Total tokens: {len(tokens):,}")

    return torch.tensor(tokens, dtype=torch.long)


def evaluate_ppl_sliding_window(
    model,
    tokens: torch.Tensor,
    device: torch.device,
    context_length: int = 2048,
    stride: int = 512,
):
    """
    Sliding window方式でPPL評価

    HuggingFace推奨の評価方法：
    - context_lengthのウィンドウをstrideずつスライド
    - 重複部分はコンテキストとして使用し、stride部分のみlossを計算
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_windows = 0

    tokens = tokens.to(device)
    seq_len = len(tokens)

    with torch.no_grad():
        for start in range(0, seq_len - 1, stride):
            end = min(start + context_length, seq_len)
            input_ids = tokens[start:end].unsqueeze(0)

            # ターゲットの開始位置（strideより前はコンテキスト）
            target_start = min(stride, end - start - 1)

            if target_start <= 0:
                continue

            # labels: 最初のtarget_start個は-100（無視）、残りは予測対象
            labels = input_ids.clone()
            labels[0, :target_start] = -100

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            # 実際に計算されたトークン数
            num_target_tokens = (labels != -100).sum().item()
            if num_target_tokens > 0:
                total_loss += loss.item() * num_target_tokens
                total_tokens += num_target_tokens
                num_windows += 1

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    print_flush(f"  Windows evaluated: {num_windows}")
    print_flush(f"  Tokens evaluated: {total_tokens:,}")

    return ppl


def evaluate_ppl_simple(
    model,
    tokens: torch.Tensor,
    device: torch.device,
    context_length: int = 2048,
):
    """
    シンプルなPPL評価（非重複セグメント）
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    tokens = tokens.to(device)
    seq_len = len(tokens)

    with torch.no_grad():
        for start in range(0, seq_len - 1, context_length):
            end = min(start + context_length, seq_len)
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


def evaluate_parallel_adapter(
    model,
    tokens: torch.Tensor,
    device: torch.device,
    segment_length: int = 256,
):
    """
    並列Adapter用の評価（メモリリセット付き、セグメント分割）

    訓練時と同じ評価方法を使用。
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    tokens = tokens.to(device)
    seq_len = len(tokens)

    # メモリリセット
    model.reset_memory()

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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load WikiText-2
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
    ppl_simple = evaluate_ppl_simple(
        original_model, tokens, device,
        context_length=args.context_length,
    )
    print_flush(f"   PPL: {ppl_simple:.2f}")

    # Method 3: Segment-based (like parallel adapter training)
    print_flush("\n3. Segment-based evaluation (256 tokens, like training):")
    ppl_segment = evaluate_ppl_simple(
        original_model, tokens, device,
        context_length=256,
    )
    print_flush(f"   PPL: {ppl_segment:.2f}")

    # Clean up
    del original_model
    torch.cuda.empty_cache()

    # =====================================================================
    # Evaluate Parallel Adapter (if provided)
    # =====================================================================
    ppl_adapter = None
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
            use_alibi=checkpoint["config"].get("use_alibi", False),
            initial_alpha=checkpoint["config"].get("initial_alpha", 0.0),
            freeze_base_model=True,
        )
        adapter_model = adapter_model.to(device)

        # Load weights
        adapter_model.infini_adapter.load_state_dict(checkpoint["infini_adapter_state_dict"])
        print_flush(f"  Loaded alpha: {adapter_model.get_alpha():.4f}")

        # Evaluate with memory
        print_flush("\n4. Parallel Adapter evaluation (with memory, 256 segments):")
        ppl_adapter = evaluate_parallel_adapter(
            adapter_model, tokens, device,
            segment_length=256,
        )
        print_flush(f"   PPL: {ppl_adapter:.2f}")

        # Also evaluate without memory reset (alpha=0 baseline)
        print_flush("\n5. Parallel Adapter with alpha=0 (baseline check):")
        original_alpha = adapter_model.get_alpha()
        adapter_model.set_alpha(0.0)
        adapter_model.reset_memory()
        ppl_alpha0 = evaluate_parallel_adapter(
            adapter_model, tokens, device,
            segment_length=256,
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
    if ppl_adapter is not None:
        print_flush(f"| Parallel Adapter (with memory) | {ppl_adapter:.2f} |")

    print_flush("\n" + "=" * 70)
    print_flush("REFERENCE")
    print_flush("=" * 70)
    print_flush("Expected Pythia-70m WikiText-2 PPL: ~32-35")
    print_flush("(See: https://huggingface.co/EleutherAI/pythia-70m)")

    if ppl_sliding > 50:
        print_flush("\n⚠️ WARNING: PPL is higher than expected. Check evaluation method.")
    else:
        print_flush(f"\n✓ PPL {ppl_sliding:.2f} is in expected range")

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
