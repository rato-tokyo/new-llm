"""Quick test of training pipeline"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.new_llm_flexible import NewLLMFlexible
from src.utils.dialogue_config import SmallDialogueConfig
from src.data.ultrachat_loader import UltraChatLoader

# Config
config = SmallDialogueConfig()
config.phase1_max_samples = 1
config.phase1_max_iterations = 10  # Quick test

print("=" * 60)
print("Testing New-LLM Training Pipeline")
print("=" * 60)

# Load data
print("\n1. Loading data...")
loader = UltraChatLoader(config)
dataset = loader.load_dataset(max_samples=1)
input_ids, mask, messages = loader.get_sample(dataset, 0)
print(f"   Sample: {len(input_ids)} tokens, {len(messages)} messages")

# Create model
print("\n2. Creating model...")
model = NewLLMFlexible(config)
print(f"   Parameters: {model.count_parameters():,}")
print(f"   Architecture: {config.num_layers} layers, context_dim={config.context_dim}")

# Test forward pass
print("\n3. Testing forward pass...")
input_ids_batch = input_ids.unsqueeze(0)  # [1, seq_len]
logits = model(input_ids_batch)
print(f"   Logits shape: {logits.shape}")

# Test fixed-point context
print("\n4. Testing fixed-point context...")
fixed_contexts, converged, num_iters = model.get_fixed_point_context(
    input_ids_batch,
    max_iterations=10,
    tolerance=1e-4
)
print(f"   Fixed contexts shape: {fixed_contexts.shape}")
print(f"   Converged: {converged.sum().item()}/{converged.numel()} tokens")
print(f"   Avg iterations: {num_iters.float().mean().item():.1f}")

print("\n" + "=" * 60)
print("✓ All tests passed!")
print("=" * 60)
