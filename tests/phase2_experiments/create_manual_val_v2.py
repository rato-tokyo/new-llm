"""Create manual validation data with proper sentences using training vocab"""
import sys
sys.path.insert(0, '/Users/sakajiritomoyoshi/Desktop/git/new-llm')

import torch
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Create validation sentences using common words from training data
# Simple question-answer format similar to training data
val_texts = [
    "What is the best way to cook salmon?",
    "The best way is to use a pan.",
    "How do I make a salad?",
    "You can use the ingredients in the kitchen.",
    "What are the steps for this recipe?",
]

print("="*70)
print("VALIDATION DATA EXAMPLES")
print("="*70)

all_val_tokens = []
for i, text in enumerate(val_texts):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(tokens)

    print(f"\n--- Val Example {i+1} ---")
    print(f"Text: {text}")
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")
    print(f"Decoded: {decoded}")

    all_val_tokens.extend(tokens)

print(f"\n{'='*70}")
print(f"Total validation tokens: {len(all_val_tokens)}")
print(f"All token IDs: {all_val_tokens}")

# Save as tensor
val_tensor = torch.tensor(all_val_tokens, dtype=torch.long)
torch.save(val_tensor, '/Users/sakajiritomoyoshi/Desktop/git/new-llm/cache/manual_val_tokens.pt')
print(f"\nSaved to: cache/manual_val_tokens.pt")
