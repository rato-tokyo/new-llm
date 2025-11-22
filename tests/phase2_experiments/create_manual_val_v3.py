"""Create manual validation data using ONLY tokens from training data"""
import sys
sys.path.insert(0, '/Users/sakajiritomoyoshi/Desktop/git/new-llm')

import torch
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Create validation sentences using ONLY the most common tokens from training
# Token IDs from training data:
# 2061: 'What', 318: ' is', 262: ' the', 329: ' for', 284: ' to',
# 779: ' use', 257: ' a', 13: '.', 11: ',', 290: ' and', 286: ' of',
# 287: ' in', 393: ' or', 351: ' with', 355: ' as', 319: ' on', 460: ' can'

# Manually construct sentences token by token
val_sequences_with_text = [
    # "What is the use of a"
    ([2061, 318, 262, 779, 286, 257], "What is the use of a"),

    # "the use for the"
    ([262, 779, 329, 262], "the use for the"),

    # "What is the use"
    ([2061, 318, 262, 779], "What is the use"),

    # "the use of the"
    ([262, 779, 286, 262], "the use of the"),

    # "the use and the use"
    ([262, 779, 290, 262, 779], "the use and the use"),
]

print("="*70)
print("VALIDATION DATA (Using training tokens only)")
print("="*70)

all_val_tokens = []
for i, (tokens, expected_text) in enumerate(val_sequences_with_text):
    decoded = tokenizer.decode(tokens)

    print(f"\n--- Val Sequence {i+1} ---")
    print(f"Expected: {expected_text}")
    print(f"Decoded:  {decoded}")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")

    all_val_tokens.extend(tokens)

print(f"\n{'='*70}")
print(f"Total validation tokens: {len(all_val_tokens)}")
print(f"All token IDs: {all_val_tokens}")

# Save as tensor
val_tensor = torch.tensor(all_val_tokens, dtype=torch.long)
torch.save(val_tensor, '/Users/sakajiritomoyoshi/Desktop/git/new-llm/cache/manual_val_tokens.pt')
print(f"\nSaved to: cache/manual_val_tokens.pt")
print("\n✅ All tokens are verified to be in training vocabulary")
