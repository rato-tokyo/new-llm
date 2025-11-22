"""Check if validation tokens are in training vocabulary"""
import sys
sys.path.insert(0, '/Users/sakajiritomoyoshi/Desktop/git/new-llm')

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load 10 training samples
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
samples = dataset.select(range(10))

# Get all training tokens
train_token_ids = []
for sample in samples:
    text = sample['messages'][0]['content'] + " " + sample['messages'][1]['content']
    tokens = tokenizer.encode(text, max_length=512, truncation=True)
    train_token_ids.extend(tokens)

train_vocab = set(train_token_ids)

# Load validation tokens
val_tokens = torch.load('/Users/sakajiritomoyoshi/Desktop/git/new-llm/cache/manual_val_tokens.pt')
val_vocab = set(val_tokens.tolist())

print("="*70)
print("VALIDATION TOKEN CHECK")
print("="*70)
print(f"Training vocabulary size: {len(train_vocab)} unique tokens")
print(f"Validation vocabulary size: {len(val_vocab)} unique tokens")

# Check if all val tokens are in train vocab
missing_tokens = val_vocab - train_vocab
if missing_tokens:
    print(f"\n⚠️  {len(missing_tokens)} validation tokens NOT in training data:")
    for tid in sorted(missing_tokens):
        print(f"  {tid}: '{tokenizer.decode([tid])}'")
else:
    print(f"\n✅ All validation tokens are in training vocabulary!")

# Show coverage
coverage = len(val_vocab - missing_tokens) / len(val_vocab) * 100
print(f"\nCoverage: {coverage:.1f}% of validation tokens are in training data")
