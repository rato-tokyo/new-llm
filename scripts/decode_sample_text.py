"""
Decode sample training data to understand the content
"""

import os
import sys
import torch
from tokenizers import Tokenizer

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import ResidualConfig

# Load configuration
config = ResidualConfig()

# Load tokenizer
tokenizer_path = os.path.join(config.cache_dir, "tokenizer", "tokenizer.json")
tokenizer = Tokenizer.from_file(tokenizer_path)

# Load training data
train_data_path = os.path.join(config.cache_dir, "ultrachat_5samples_128len.pt")
train_token_ids = torch.load(train_data_path)

# Decode
text = tokenizer.decode(train_token_ids.tolist())

print("="*70)
print("Sample Training Data (First 2000 characters)")
print("="*70)
print(text[:2000])
print("\n...")
print("\n="*70)
print(f"Total text length: {len(text)} characters")
print(f"Total tokens: {len(train_token_ids)}")
print("="*70)
