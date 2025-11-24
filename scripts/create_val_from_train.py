"""
Create validation data from training data tokens

This ensures validation data contains ONLY tokens from training data.
Strategy: Take 20% of training tokens as validation data.
"""

import torch
import os
from transformers import AutoTokenizer

# Paths
cache_dir = "./cache"
train_cache = os.path.join(cache_dir, "ultrachat_50samples_128len.pt")
val_text_output = "./data/example_val.txt"

print("\n" + "="*70)
print("Creating Validation Data from Training Tokens")
print("="*70 + "\n")

# Load training data
if not os.path.exists(train_cache):
    print(f"❌ Training cache not found: {train_cache}")
    print("Run train.py or test_5000_tokens.py first to generate cache")
    exit(1)

train_token_ids = torch.load(train_cache)
print(f"Loaded training data: {len(train_token_ids)} tokens")

# Take last 20% as validation (ensures all tokens are in training set)
split_idx = int(len(train_token_ids) * 0.8)
val_token_ids = train_token_ids[split_idx:]

print(f"Validation tokens: {len(val_token_ids)} (last 20% of training)")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "gpt2",
    cache_dir=os.path.join(cache_dir, "tokenizer")
)

# Decode to text
val_text = tokenizer.decode(val_token_ids.tolist())

print(f"\nValidation text length: {len(val_text)} characters")
print(f"\nFirst 200 characters:\n{val_text[:200]}...\n")

# Save text file
os.makedirs(os.path.dirname(val_text_output), exist_ok=True)
with open(val_text_output, 'w', encoding='utf-8') as f:
    f.write(val_text)

print(f"✅ Saved validation text to: {val_text_output}")

# Verify compatibility
train_set = set(train_token_ids.tolist())
val_set = set(val_token_ids.tolist())
unseen = val_set - train_set

if unseen:
    print(f"\n❌ ERROR: Found {len(unseen)} unseen tokens!")
else:
    print(f"\n✅ VERIFIED: All {len(val_set)} unique validation tokens exist in training data")

print("="*70 + "\n")
