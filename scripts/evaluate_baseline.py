#!/usr/bin/env python3
"""
Baseline PPL Evaluation Script

Pretrained OpenCALMモデルの長文PPLを測定し、ベースラインを確立する。

Usage:
    python3 scripts/evaluate_baseline.py --num-docs 10 --tokens-per-doc 4096
"""

import argparse
import sys

sys.path.insert(0, ".")

import torch
from transformers import AutoModelForCausalLM

from src.config import SenriConfig, ExperimentConfig
from src.utils.data_pythia import load_long_documents_from_pile
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.tokenizer_utils import get_tokenizer
from src.utils.training import get_device


def evaluate_ppl_standard(model, documents: list, device: torch.device, segment_length: int):
    """
    標準的なPPL評価（メモリリセットなし、セグメント分割）

    各ドキュメントをsegment_lengthで分割し、各セグメントを独立に評価。
    これはオリジナルPythiaの標準的な評価方法。

    ⚠️ 問題: position_idsが各セグメントで0から始まるため、高PPLになる
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


def evaluate_ppl_with_position_ids(model, documents: list, device: torch.device, segment_length: int):
    """
    正しいposition_idsを使用したPPL評価

    各セグメントで正しいposition_idsを渡すことで、
    文書内の位置情報を保持する。
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

                # 正しいposition_idsを設定（文書内の実際の位置）
                position_ids = torch.arange(
                    start, start + len(segment) - 1,
                    device=device
                ).unsqueeze(0)

                outputs = model(input_ids, labels=labels, position_ids=position_ids)
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

    # モデル名オプション（SenriConfigから取得）
    model_config = SenriConfig()
    parser.add_argument(
        "--model", default=model_config.tokenizer_name,
        help=f"Model name (default: {model_config.tokenizer_name})"
    )

    # ExperimentConfigから評価関連の設定を追加
    exp_config = ExperimentConfig()
    parser.add_argument(
        "--num-docs", type=int, default=exp_config.num_docs,
        help=f"Number of documents (default: {exp_config.num_docs})"
    )
    parser.add_argument(
        "--tokens-per-doc", type=int, default=exp_config.tokens_per_doc,
        help=f"Tokens per document (default: {exp_config.tokens_per_doc})"
    )
    parser.add_argument(
        "--seq-length", type=int, default=exp_config.seq_length,
        help=f"Segment length for evaluation (default: {exp_config.seq_length})"
    )

    args = parser.parse_args()

    # argsから設定を更新
    exp_config = ExperimentConfig(
        num_docs=args.num_docs,
        tokens_per_doc=args.tokens_per_doc,
        seq_length=args.seq_length,
    )

    set_seed(42)
    device = get_device()

    print_flush("=" * 70)
    print_flush("BASELINE PPL EVALUATION")
    print_flush("=" * 70)
    print_flush(f"Model: {args.model}")
    print_flush(f"Device: {device}")
    print_flush(f"Documents: {exp_config.num_docs}")
    print_flush(f"Tokens per document: {exp_config.tokens_per_doc}")
    print_flush(f"Segment length: {exp_config.seq_length}")

    # Load tokenizer
    tokenizer = get_tokenizer(args.model)

    # Load documents
    documents = load_long_documents_from_pile(tokenizer, exp_config.num_docs, exp_config.tokens_per_doc)

    # Load original model
    print_flush(f"\nLoading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print_flush(f"Total parameters: {total_params:,}")

    # Evaluate with different methods
    print_flush("\n" + "=" * 70)
    print_flush("EVALUATION RESULTS")
    print_flush("=" * 70)

    # Method 1: Segment-based WITHOUT position_ids (problematic)
    print_flush("\n1. Segment-based WITHOUT position_ids (position resets each segment):")
    ppl_no_pos = evaluate_ppl_standard(model, documents, device, exp_config.seq_length)
    print_flush(f"   PPL: {ppl_no_pos:.1f}")
    print_flush("   ⚠️ This is problematic - position_ids reset to 0 for each segment")

    # Method 2: Segment-based WITH correct position_ids
    print_flush("\n2. Segment-based WITH correct position_ids:")
    ppl_with_pos = evaluate_ppl_with_position_ids(model, documents, device, exp_config.seq_length)
    print_flush(f"   PPL: {ppl_with_pos:.1f}")
    print_flush("   ✓ This preserves document position information")

    # Method 3: Full context (up to 2048)
    print_flush("\n3. Full context evaluation (max_length=2048):")
    ppl_full = evaluate_ppl_full_context(model, documents, device, max_length=2048)
    print_flush(f"   PPL: {ppl_full:.1f}")

    # Summary
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)
    print_flush("| Method | PPL |")
    print_flush("|--------|-----|")
    print_flush(f"| Segment (no position_ids) | {ppl_no_pos:.1f} |")
    print_flush(f"| Segment (with position_ids) | {ppl_with_pos:.1f} |")
    print_flush(f"| Full context (2048 tokens) | {ppl_full:.1f} |")

    print_flush("\n" + "=" * 70)
    print_flush("ANALYSIS")
    print_flush("=" * 70)
    print_flush(f"Position ID impact: {ppl_no_pos:.1f} -> {ppl_with_pos:.1f} "
                f"({ppl_no_pos/ppl_with_pos:.1f}x improvement)")

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
