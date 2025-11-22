"""Analyze tokens used in training data and create validation data"""
import sys
sys.path.insert(0, '/Users/sakajiritomoyoshi/Desktop/git/new-llm')

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load 10 samples
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
samples = dataset.select(range(10))

# Tokenize all samples
all_token_ids = []
for sample in samples:
    text = sample['messages'][0]['content'] + " " + sample['messages'][1]['content']
    tokens = tokenizer.encode(text, max_length=512, truncation=True)
    all_token_ids.extend(tokens)

# Get unique tokens
unique_tokens = sorted(set(all_token_ids))
print(f"Total tokens: {len(all_token_ids)}")
print(f"Unique tokens: {len(unique_tokens)}")
print(f"\nFirst 50 unique token IDs: {unique_tokens[:50]}")

# Decode some common tokens
print(f"\nSample tokens:")
for tid in unique_tokens[:20]:
    print(f"  {tid}: '{tokenizer.decode([tid])}'")

# Create validation sentences using only these tokens
# Simple approach: create short sequences from frequent tokens
from collections import Counter
token_freq = Counter(all_token_ids)
most_common = [t for t, _ in token_freq.most_common(100)]

print(f"\nMost common 20 tokens:")
for tid in most_common[:20]:
    print(f"  {tid}: '{tokenizer.decode([tid])}' (count: {token_freq[tid]})")

# Save token vocabulary
print(f"\nSaving token vocabulary to /tmp/train_token_vocab.txt")
with open('/tmp/train_token_vocab.txt', 'w') as f:
    f.write(f"Total tokens: {len(all_token_ids)}\n")
    f.write(f"Unique tokens: {len(unique_tokens)}\n\n")
    f.write("Unique token IDs:\n")
    f.write(str(unique_tokens) + "\n\n")
    f.write("Most common 50 tokens:\n")
    for tid in most_common[:50]:
        f.write(f"{tid}: '{tokenizer.decode([tid])}' (count: {token_freq[tid]})\n")

print("Done!")
