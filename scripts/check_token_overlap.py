#!/usr/bin/env python3
"""
訓練データと検証データのトークン重複率を調べるスクリプト

検証データの各トークンが訓練データに存在するかをチェックし、
重複率を計算します。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from config import ResidualConfig


def main():
    config = ResidualConfig()

    print("=" * 60)
    print("Token Overlap Analysis: Train vs Validation")
    print("=" * 60)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        cache_dir=os.path.join(config.cache_dir, "tokenizer")
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load cached data or download
    cache_file = os.path.join(
        config.cache_dir,
        f"ultrachat_{config.num_samples}samples_full.pt"
    )

    if os.path.exists(cache_file):
        print(f"\n📂 Loading from cache: {cache_file}")
        cached = torch.load(cache_file)
        all_token_ids = cached['token_ids'] if isinstance(cached, dict) else cached
    else:
        print(f"\n📥 Downloading UltraChat dataset...")
        dataset = load_dataset(
            config.dataset_name,
            split=config.dataset_split,
            cache_dir=os.path.join(config.cache_dir, "datasets")
        )

        all_tokens = []
        for idx in range(min(config.num_samples, len(dataset))):
            messages = dataset[idx]["messages"]
            text = "\n".join([msg["content"] for msg in messages])

            tokens = tokenizer(
                text,
                truncation=False,
                return_tensors="pt"
            )
            all_tokens.append(tokens["input_ids"].squeeze(0))

        all_token_ids = torch.cat(all_tokens)

        os.makedirs(config.cache_dir, exist_ok=True)
        torch.save({'token_ids': all_token_ids}, cache_file)

    # Split into train/val (same as colab.py)
    val_ratio = 0.2
    val_size = int(len(all_token_ids) * val_ratio)
    train_size = len(all_token_ids) - val_size

    train_token_ids = all_token_ids[:train_size]
    val_token_ids = all_token_ids[train_size:]

    print(f"\n📊 Data Split:")
    print(f"   Total tokens: {len(all_token_ids):,}")
    print(f"   Train tokens: {len(train_token_ids):,} (80%)")
    print(f"   Val tokens:   {len(val_token_ids):,} (20%)")

    # Get unique tokens in each set
    train_unique = set(train_token_ids.tolist())
    val_unique = set(val_token_ids.tolist())

    print(f"\n📈 Unique Token Counts:")
    print(f"   Train unique tokens: {len(train_unique):,}")
    print(f"   Val unique tokens:   {len(val_unique):,}")

    # Check overlap
    val_tokens_in_train = val_unique.intersection(train_unique)
    val_tokens_not_in_train = val_unique - train_unique

    print(f"\n🔍 Token Overlap Analysis:")
    print(f"   Val tokens also in train:     {len(val_tokens_in_train):,} ({len(val_tokens_in_train)/len(val_unique)*100:.1f}%)")
    print(f"   Val tokens NOT in train:      {len(val_tokens_not_in_train):,} ({len(val_tokens_not_in_train)/len(val_unique)*100:.1f}%)")

    # Count occurrences of each token
    val_token_list = val_token_ids.tolist()

    # Count how many val tokens (with repetition) are in train
    val_in_train_count = sum(1 for t in val_token_list if t in train_unique)
    val_not_in_train_count = len(val_token_list) - val_in_train_count

    print(f"\n📌 Token Coverage (counting repetitions):")
    print(f"   Val tokens covered by train:  {val_in_train_count:,} / {len(val_token_list):,} ({val_in_train_count/len(val_token_list)*100:.2f}%)")
    print(f"   Val tokens NOT covered:       {val_not_in_train_count:,} / {len(val_token_list):,} ({val_not_in_train_count/len(val_token_list)*100:.2f}%)")

    # Show examples of tokens not in train
    if val_tokens_not_in_train:
        print(f"\n📝 Examples of val tokens NOT in train (first 10):")
        for i, token_id in enumerate(list(val_tokens_not_in_train)[:10]):
            token_text = tokenizer.decode([token_id])
            # Count occurrences in val
            count_in_val = val_token_list.count(token_id)
            print(f"   {i+1}. Token ID {token_id}: '{token_text}' (appears {count_in_val}x in val)")
    else:
        print(f"\n✅ All val tokens exist in train!")

    # Summary
    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    coverage_pct = val_in_train_count / len(val_token_list) * 100

    if coverage_pct == 100.0:
        print(f"✅ PERFECT: 100% of val tokens exist in train")
    elif coverage_pct >= 99.0:
        print(f"✅ EXCELLENT: {coverage_pct:.2f}% of val tokens exist in train")
    elif coverage_pct >= 95.0:
        print(f"⚠️  GOOD: {coverage_pct:.2f}% of val tokens exist in train")
    else:
        print(f"❌ WARNING: Only {coverage_pct:.2f}% of val tokens exist in train")
        print(f"   This may affect validation accuracy!")

    print("=" * 60)


if __name__ == "__main__":
    main()
