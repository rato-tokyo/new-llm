#!/usr/bin/env python3
"""
Baseline PPL Evaluation Script

オリジナルPythia-70m（Infiniなし）の長文PPLを測定し、
ベースラインを確立する。

Usage:
    python3 scripts/evaluate_baseline.py --num-docs 10 --tokens-per-doc 4096
"""

import argparse
import sys

sys.path.insert(0, ".")

import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.training import get_device


def load_long_documents(tokenizer, num_docs: int, tokens_per_doc: int):
    """長文データをロード"""
    from datasets import load_dataset

    print_flush(f"Loading {num_docs} long documents ({tokens_per_doc} tokens each)...")

    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
    )

    documents = []
    current_tokens = []

    for example in dataset:
        text = example["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        current_tokens.extend(tokens)

        while len(current_tokens) >= tokens_per_doc:
            doc = current_tokens[:tokens_per_doc]
            documents.append(torch.tensor(doc, dtype=torch.long))
            current_tokens = current_tokens[tokens_per_doc:]

            if len(documents) >= num_docs:
                break

        if len(documents) >= num_docs:
            break

    print_flush(f"Loaded {len(documents)} documents")
    return documents


def evaluate_ppl_standard(model, documents: list, device: torch.device, segment_length: int):
    """
    標準的なPPL評価（メモリリセットなし、セグメント分割）

    各ドキュメントをsegment_lengthで分割し、各セグメントを独立に評価。
    これはオリジナルPythiaの標準的な評価方法。
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for doc in documents:
            doc = doc.to(device)

            for start in range(0, len(doc) - 1, segment_length):
                end = min(start + segment_length, len(doc))
                segment = doc[start:end]

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


def evaluate_ppl_full_context(model, documents: list, device: torch.device, max_length: int = 2048):
    """
    フルコンテキストPPL評価（Pythiaの最大長まで）

    Pythia-70mの最大コンテキスト長は2048。
    それを超える部分は切り捨てて評価。
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for doc in documents:
            # 最大長に切り詰め
            doc = doc[:max_length].to(device)

            if len(doc) < 2:
                continue

            input_ids = doc[:-1].unsqueeze(0)
            labels = doc[1:].unsqueeze(0)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl


def main():
    parser = argparse.ArgumentParser(description="Baseline PPL Evaluation")

    parser.add_argument("--model", default="EleutherAI/pythia-70m", help="Model name")
    parser.add_argument("--num-docs", type=int, default=10, help="Number of documents")
    parser.add_argument("--tokens-per-doc", type=int, default=4096, help="Tokens per document")
    parser.add_argument("--seq-length", type=int, default=256, help="Segment length for evaluation")

    args = parser.parse_args()

    set_seed(42)
    device = get_device()

    print_flush("=" * 70)
    print_flush("BASELINE PPL EVALUATION")
    print_flush("=" * 70)
    print_flush(f"Model: {args.model}")
    print_flush(f"Device: {device}")
    print_flush(f"Documents: {args.num_docs}")
    print_flush(f"Tokens per document: {args.tokens_per_doc}")
    print_flush(f"Segment length: {args.seq_length}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load documents
    documents = load_long_documents(tokenizer, args.num_docs, args.tokens_per_doc)

    # Load original model
    print_flush("\nLoading original Pythia-70m...")
    model = GPTNeoXForCausalLM.from_pretrained(args.model)
    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print_flush(f"Total parameters: {total_params:,}")

    # Evaluate with different methods
    print_flush("\n" + "=" * 70)
    print_flush("EVALUATION RESULTS")
    print_flush("=" * 70)

    # Method 1: Segment-based (same as parallel adapter training)
    print_flush("\n1. Segment-based evaluation (segment_length=256):")
    ppl_segment = evaluate_ppl_standard(model, documents, device, args.seq_length)
    print_flush(f"   PPL: {ppl_segment:.1f}")

    # Method 2: Full context (up to 2048)
    print_flush("\n2. Full context evaluation (max_length=2048):")
    ppl_full = evaluate_ppl_full_context(model, documents, device, max_length=2048)
    print_flush(f"   PPL: {ppl_full:.1f}")

    # Method 3: Full document (4096 tokens, beyond context limit)
    print_flush("\n3. Full document evaluation (4096 tokens, truncated to 2048):")
    # This is essentially the same as method 2 since we truncate
    # But let's show what happens if we try to process 4096 tokens
    print_flush("   Note: Pythia-70m has max context of 2048, so 4096 tokens would be truncated")

    # Summary
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)
    print_flush("| Method | PPL |")
    print_flush("|--------|-----|")
    print_flush(f"| Segment-based (256 tokens) | {ppl_segment:.1f} |")
    print_flush(f"| Full context (2048 tokens) | {ppl_full:.1f} |")

    print_flush("\n" + "=" * 70)
    print_flush("COMPARISON WITH PARALLEL ADAPTER")
    print_flush("=" * 70)
    print_flush("| Model | PPL |")
    print_flush("|-------|-----|")
    print_flush(f"| Original Pythia-70m (segment=256) | {ppl_segment:.1f} |")
    print_flush(f"| Parallel Adapter (after training) | 514.1 |")

    if ppl_segment < 514.1:
        print_flush(f"\n⚠️ WARNING: Original Pythia is BETTER than Parallel Adapter!")
        print_flush(f"   Difference: {514.1 - ppl_segment:.1f} PPL worse")
    else:
        print_flush(f"\n✓ Parallel Adapter improved PPL by {ppl_segment - 514.1:.1f}")

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
