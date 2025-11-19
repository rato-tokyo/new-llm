#!/usr/bin/env python3
"""
Build and save UltraChat tokenizer for use with chat interface

This script builds the tokenizer from a subset of UltraChat data
and saves it so chat.py can load it quickly without re-downloading.

Usage:
    python scripts/save_ultrachat_tokenizer.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pickle
from src.utils.config import NewLLMConfig

print("=" * 80)
print("Building UltraChat Tokenizer")
print("=" * 80)

# Create config
config = NewLLMConfig()
config.vocab_size = 1000  # Match the checkpoint

print(f"\n📋 Config:")
print(f"   Vocab size: {config.vocab_size}")

# Load dataset and build tokenizer
print(f"\n📥 Loading UltraChat data (this will download ~1GB)...")
print(f"   Using 10,000 samples to build tokenizer")

from src.training.ultrachat_dataset import load_ultrachat_data

_, _, tokenizer = load_ultrachat_data(config, max_samples=10000)

print(f"\n✓ Tokenizer built:")
print(f"   Vocabulary: {len(tokenizer.word2idx)} words")

# Save tokenizer
output_path = "checkpoints/ultrachat_tokenizer.pkl"
os.makedirs("checkpoints", exist_ok=True)

print(f"\n💾 Saving tokenizer to: {output_path}")
with open(output_path, 'wb') as f:
    pickle.dump(tokenizer, f)

# Verify it can be loaded
print(f"\n🔍 Verifying saved tokenizer...")
with open(output_path, 'rb') as f:
    loaded_tokenizer = pickle.load(f)

assert len(loaded_tokenizer.word2idx) == len(tokenizer.word2idx), "Tokenizer verification failed"

print(f"✓ Tokenizer verified")

print(f"\n" + "=" * 80)
print(f"✅ Tokenizer saved successfully!")
print(f"=" * 80)
print(f"\nYou can now run chat with:")
print(f"  python scripts/chat.py --checkpoint checkpoints/best_new_llm_ultrachat_layers1.pt")
print()
