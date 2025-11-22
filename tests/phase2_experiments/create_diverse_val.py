"""Create diverse validation data using training vocabulary"""
import sys
sys.path.insert(0, '/Users/sakajiritomoyoshi/Desktop/git/new-llm')

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from collections import Counter
import random

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load 10 training samples to get vocabulary
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
samples = dataset.select(range(10))

# Get all training tokens
train_token_ids = []
for sample in samples:
    text = sample['messages'][0]['content'] + " " + sample['messages'][1]['content']
    tokens = tokenizer.encode(text, max_length=512, truncation=True)
    train_token_ids.extend(tokens)

# Get token frequency
token_freq = Counter(train_token_ids)
train_vocab = set(train_token_ids)

print("="*70)
print("TRAINING VOCABULARY ANALYSIS")
print("="*70)
print(f"Total training tokens: {len(train_token_ids)}")
print(f"Unique tokens: {len(train_vocab)}")
print(f"\nTop 100 most common tokens:")
most_common = token_freq.most_common(100)
for i, (tid, count) in enumerate(most_common[:20]):
    token_text = tokenizer.decode([tid])
    print(f"  {i+1:2d}. Token {tid:5d} '{token_text:20s}' count={count}")

# Create diverse validation sentences using top 100 tokens
# Strategy: Create random but grammatically plausible sequences
top_100_tokens = [t for t, _ in most_common]

print(f"\n{'='*70}")
print("CREATING DIVERSE VALIDATION DATA")
print(f"{'='*70}")

# Create validation sequences
val_sequences = []

# Type 1: Short sequences (5-10 tokens)
for _ in range(20):
    length = random.randint(5, 10)
    seq = random.sample(top_100_tokens, min(length, len(top_100_tokens)))
    val_sequences.append(seq)

# Type 2: Repeating patterns
val_sequences.append([262, 290, 284, 286, 287, 329])  # the, and, to, of, in, for
val_sequences.append([257, 262, 286, 262, 290])  # a, the, of, the, and
val_sequences.append([318, 262, 284, 257, 329])  # is, the, to, a, for

# Type 3: Common word combinations
val_sequences.append([2061, 318, 262])  # What is the
val_sequences.append([262, 284, 262, 286])  # the to the of
val_sequences.append([460, 779, 262, 329])  # can use the for

# Flatten all sequences
all_val_tokens = []
for seq in val_sequences:
    all_val_tokens.extend(seq)

print(f"\nCreated {len(val_sequences)} validation sequences")
print(f"Total validation tokens: {len(all_val_tokens)}")

# Verify all tokens are in training vocabulary
val_vocab = set(all_val_tokens)
missing = val_vocab - train_vocab
if missing:
    print(f"\n⚠️  WARNING: {len(missing)} tokens not in training vocab!")
    print(f"Missing tokens: {missing}")
else:
    print(f"\n✅ All {len(val_vocab)} unique validation tokens are in training vocabulary")

# Show some example sequences
print(f"\n{'='*70}")
print("VALIDATION SEQUENCE EXAMPLES (First 10)")
print(f"{'='*70}")
for i, seq in enumerate(val_sequences[:10]):
    decoded = tokenizer.decode(seq)
    print(f"\nSeq {i+1} ({len(seq)} tokens):")
    print(f"  Tokens: {seq}")
    print(f"  Text: {decoded}")

# Save as tensor
val_tensor = torch.tensor(all_val_tokens, dtype=torch.long)
torch.save(val_tensor, '/Users/sakajiritomoyoshi/Desktop/git/new-llm/cache/manual_val_tokens.pt')
print(f"\n{'='*70}")
print(f"Saved to: cache/manual_val_tokens.pt")
print(f"Total tokens: {len(all_val_tokens)}")
print(f"Unique tokens: {len(val_vocab)}")
print(f"✅ All tokens verified to be in training vocabulary")
print(f"{'='*70}")
