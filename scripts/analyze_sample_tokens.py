"""
Sample Data Token Analysis Script

This script analyzes the tokens used in training data and generates statistics
to help create validation data with the same vocabulary.

Usage:
    python3 scripts/analyze_sample_tokens.py
"""

import os
import sys
import torch
from collections import Counter
from tokenizers import Tokenizer

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import ResidualConfig


def analyze_tokens(token_ids, tokenizer, label="Data"):
    """
    Analyze token distribution in the dataset

    Args:
        token_ids: Tensor of token IDs
        tokenizer: Tokenizer object
        label: Label for output

    Returns:
        dict with token statistics
    """
    # Convert to list
    token_list = token_ids.tolist()

    # Basic statistics
    total_tokens = len(token_list)
    unique_tokens = len(set(token_list))

    # Token frequency
    token_counts = Counter(token_list)
    most_common = token_counts.most_common(20)

    # Decode tokens
    unique_token_ids = sorted(set(token_list))

    print(f"\n{'='*70}")
    print(f"{label} Token Analysis")
    print(f"{'='*70}\n")

    print(f"Total tokens: {total_tokens:,}")
    print(f"Unique tokens: {unique_tokens:,}")
    print(f"Vocabulary coverage: {unique_tokens / 50257 * 100:.2f}%")

    print(f"\nTop 20 most frequent tokens:")
    for token_id, count in most_common:
        token_str = tokenizer.decode([token_id])
        # Escape special characters for display
        token_display = repr(token_str)[1:-1]  # Remove quotes from repr
        print(f"  {token_id:6d} ({count:4d}x): {token_display}")

    print(f"\nAll unique token IDs (first 50):")
    token_ids_str = ', '.join(str(tid) for tid in unique_token_ids[:50])
    print(f"  {token_ids_str}")
    if len(unique_token_ids) > 50:
        print(f"  ... and {len(unique_token_ids) - 50} more")

    # Save token vocabulary to file
    vocab_file = os.path.join(project_root, "cache", f"{label.lower()}_vocab.txt")
    os.makedirs(os.path.dirname(vocab_file), exist_ok=True)

    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.write(f"# Token Vocabulary for {label}\n")
        f.write(f"# Total tokens: {total_tokens}\n")
        f.write(f"# Unique tokens: {unique_tokens}\n\n")

        for token_id in unique_token_ids:
            token_str = tokenizer.decode([token_id])
            f.write(f"{token_id}\t{token_str}\n")

    print(f"\n✅ Saved vocabulary to: {vocab_file}")

    return {
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'unique_token_ids': unique_token_ids,
        'token_counts': token_counts,
        'vocab_file': vocab_file
    }


def main():
    """Main analysis function"""

    print(f"\n{'='*70}")
    print("Sample Data Token Analysis")
    print(f"{'='*70}\n")

    # Load configuration
    config = ResidualConfig()

    # Load tokenizer
    tokenizer_path = os.path.join(config.cache_dir, "tokenizer", "tokenizer.json")

    if not os.path.exists(tokenizer_path):
        print("❌ Tokenizer not found. Please run train.py first to download it.")
        return

    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Load training data
    train_data_path = os.path.join(config.cache_dir, "ultrachat_5samples_128len.pt")

    if not os.path.exists(train_data_path):
        print("❌ Training data not found. Please run train.py first to generate it.")
        return

    train_token_ids = torch.load(train_data_path)

    # Analyze
    stats = analyze_tokens(train_token_ids, tokenizer, label="Train")

    # Summary for validation data generation
    print(f"\n{'='*70}")
    print("Validation Data Generation Guide")
    print(f"{'='*70}\n")

    print(f"To generate appropriate validation data:")
    print(f"1. Use only the {stats['unique_tokens']} unique tokens from training data")
    print(f"2. Create sentences with similar token frequency distribution")
    print(f"3. Aim for ~20-30% of training data size ({int(stats['total_tokens'] * 0.25)} tokens)")
    print(f"4. Ensure different sentences from training data (naturally happens)")

    print(f"\nVocabulary file location:")
    print(f"  {stats['vocab_file']}")

    print(f"\nNext steps:")
    print(f"1. Review the vocabulary file")
    print(f"2. Use a text generation agent to create validation sentences")
    print(f"3. Save to data/example_val.txt")
    print(f"4. Configure config.py to use it")


if __name__ == "__main__":
    main()
