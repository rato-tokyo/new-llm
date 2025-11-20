#!/usr/bin/env python3
"""
Check if model has a global attractor point (all tokens converge to same context)
"""

import os
import sys
import torch
import numpy as np
from transformers import GPT2Tokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.new_llm import NewLLM


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    state_dict = checkpoint['model_state_dict']
    has_gates = any('forget_gate' in k or 'input_gate' in k for k in state_dict.keys())
    config.context_update_strategy = 'gated' if has_gates else 'simple'

    model = NewLLM(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model


def get_converged_context(model, token_id, num_reps=10, device='cpu'):
    """Get converged context after repetition"""
    input_ids = torch.tensor([[token_id] * num_reps], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, context_trajectory = model(input_ids)

    # Return context at last step (should be converged)
    return context_trajectory[0, -1, :]


def main():
    device = torch.device('cpu')

    print("="*80)
    print("Global Attractor Detection")
    print("="*80)

    # Load model
    model = load_model('checkpoints/new_llm_repetition_final.pt', device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Test 50 random tokens
    num_test_tokens = 50
    test_token_ids = np.random.randint(100, 5000, size=num_test_tokens)

    print(f"\n📊 Testing {num_test_tokens} tokens...")

    converged_contexts = []

    for token_id in test_token_ids:
        ctx = get_converged_context(model, token_id, num_reps=10, device=device)
        converged_contexts.append(ctx.cpu().numpy())

    converged_contexts = np.array(converged_contexts)  # [num_tokens, context_dim]

    print(f"\n{'='*80}")
    print("Analysis: Are all tokens converging to the same point?")
    print("="*80)

    # Compute pairwise distances
    print("\nComputing pairwise L2 distances between converged contexts...")

    distances = []
    for i in range(num_test_tokens):
        for j in range(i+1, num_test_tokens):
            dist = np.linalg.norm(converged_contexts[i] - converged_contexts[j])
            distances.append(dist)

    distances = np.array(distances)

    print(f"\nPairwise L2 Distance Statistics:")
    print(f"  Mean: {np.mean(distances):.6f}")
    print(f"  Std:  {np.std(distances):.6f}")
    print(f"  Min:  {np.min(distances):.6f}")
    print(f"  Max:  {np.max(distances):.6f}")

    # Check if all contexts are essentially identical
    if np.mean(distances) < 0.01:
        print("\n🚨 GLOBAL ATTRACTOR DETECTED!")
        print("   All tokens converge to the SAME fixed point")
        print("   This is a degenerate solution - model ignores token identity")
    elif np.mean(distances) < 0.1:
        print("\n⚠️  SUSPICIOUS: Very small distances between contexts")
        print("   Model may be learning token-independent representations")
    else:
        print("\n✅ Different tokens have different fixed points")
        print("   Model is learning token-specific representations")

    # Compute variance across all dimensions
    print(f"\n{'='*80}")
    print("Per-Dimension Variance Analysis")
    print("="*80)

    dim_variance = np.var(converged_contexts, axis=0)  # Variance for each dimension

    print(f"\nVariance across {num_test_tokens} tokens for each dimension:")
    print(f"  Mean variance: {np.mean(dim_variance):.6f}")
    print(f"  Max variance:  {np.max(dim_variance):.6f}")
    print(f"  Min variance:  {np.min(dim_variance):.6f}")

    # Count dimensions with very low variance (< 0.001)
    low_var_dims = np.sum(dim_variance < 0.001)
    print(f"\nDimensions with variance < 0.001: {low_var_dims} / {len(dim_variance)}")

    if low_var_dims > len(dim_variance) * 0.9:
        print("\n🚨 CRITICAL: >90% of dimensions have near-zero variance")
        print("   Model has collapsed to a single global attractor point")
    elif low_var_dims > len(dim_variance) * 0.5:
        print("\n⚠️  WARNING: >50% of dimensions have near-zero variance")
        print("   Model representations lack diversity")
    else:
        print("\n✅ Dimensions show reasonable variance")

    # Print a few example tokens and their contexts
    print(f"\n{'='*80}")
    print("Sample Converged Contexts (first 10 dimensions)")
    print("="*80)

    for i in range(min(5, num_test_tokens)):
        token_id = test_token_ids[i]
        token_str = tokenizer.decode([token_id])
        ctx = converged_contexts[i, :10]
        print(f"Token {token_id:5d} ('{token_str:10s}'): {ctx}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
