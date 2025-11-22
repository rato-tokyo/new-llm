"""Show actual training data examples"""
import sys
sys.path.insert(0, '/Users/sakajiritomoyoshi/Desktop/git/new-llm')

from transformers import AutoTokenizer
from datasets import load_dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Load 10 samples
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
samples = dataset.select(range(10))

print("="*70)
print("TRAIN DATA EXAMPLES (First 10 samples)")
print("="*70)

for i, sample in enumerate(samples):
    user_msg = sample['messages'][0]['content']
    assistant_msg = sample['messages'][1]['content']

    print(f"\n--- Sample {i+1} ---")
    print(f"User: {user_msg[:200]}")  # First 200 chars
    print(f"Assistant: {assistant_msg[:200]}")  # First 200 chars

    # Tokenize
    full_text = user_msg + " " + assistant_msg
    tokens = tokenizer.encode(full_text, max_length=512, truncation=True)
    decoded = tokenizer.decode(tokens)

    print(f"Tokens: {len(tokens)}")
    print(f"Decoded (first 150 chars): {decoded[:150]}")

    if i < 3:  # Show full text for first 3 samples
        print(f"\nFull User message:\n{user_msg}")
        print(f"\nFull Assistant message:\n{assistant_msg}\n")
