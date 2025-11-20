#!/usr/bin/env python3
"""
Debug script to check if CVFPT experiment has degenerate solution
"""

import os
import sys
import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.new_llm import NewLLM
from transformers import GPT2Tokenizer


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    # Detect actual updater strategy
    state_dict = checkpoint['model_state_dict']
    has_gates = any('forget_gate' in k or 'input_gate' in k for k in state_dict.keys())

    if has_gates:
        actual_strategy = 'gated'
    else:
        actual_strategy = 'simple'

    config.context_update_strategy = actual_strategy

    model = NewLLM(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, config


def debug_context_trajectory(model, token_id, num_reps=10, device='cpu'):
    """Debug: Print detailed context trajectory"""

    print(f"\n{'='*80}")
    print(f"Debug: Context Trajectory for Token ID {token_id}")
    print("=" * 80)

    # Repeat token N times
    input_ids = torch.tensor([[token_id] * num_reps], dtype=torch.long, device=device)

    with torch.no_grad():
        logits, context_trajectory = model(input_ids)

    # Print context at each step
    print(f"\nContext trajectory shape: {context_trajectory.shape}")
    print(f"Expected: [1, {num_reps}, 256]")

    # Compute L2 norms at each step
    norms = []
    for t in range(num_reps):
        ctx = context_trajectory[0, t, :]
        norm = torch.norm(ctx).item()
        norms.append(norm)
        print(f"Step {t}: L2 norm = {norm:.6f}")

    # Check if context is changing
    print(f"\n{'='*80}")
    print("Context Change Analysis")
    print("=" * 80)

    for t in range(1, num_reps):
        prev_ctx = context_trajectory[0, t-1, :]
        curr_ctx = context_trajectory[0, t, :]

        # L2 distance
        l2_diff = torch.norm(curr_ctx - prev_ctx).item()

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            prev_ctx.unsqueeze(0), curr_ctx.unsqueeze(0)
        ).item()

        print(f"Step {t-1} → {t}: L2 change = {l2_diff:.6f}, Cosine sim = {cos_sim:.6f}")

    # Check for degenerate solution
    print(f"\n{'='*80}")
    print("Degenerate Solution Check")
    print("=" * 80)

    # Are all contexts identical?
    first_ctx = context_trajectory[0, 0, :]
    all_identical = True

    for t in range(1, num_reps):
        curr_ctx = context_trajectory[0, t, :]
        if not torch.allclose(first_ctx, curr_ctx, atol=1e-4):
            all_identical = False
            break

    if all_identical:
        print("🚨 DEGENERATE SOLUTION DETECTED!")
        print("   All contexts are identical - model is not updating context!")
    else:
        print("✅ Context is being updated")

    # Check if norm is constant
    norm_std = np.std(norms)
    print(f"\nL2 Norm std dev: {norm_std:.6f}")
    if norm_std < 1e-4:
        print("🚨 WARNING: L2 norms are constant - possible degenerate solution")
    else:
        print("✅ L2 norms vary across steps")

    return context_trajectory


def compare_single_vs_repeated(model, token_id, device='cpu'):
    """Compare single-pass vs repeated context"""

    print(f"\n{'='*80}")
    print(f"Single-Pass vs Repeated Comparison for Token ID {token_id}")
    print("=" * 80)

    # Single pass
    input_single = torch.tensor([[token_id]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits_single, ctx_traj_single = model(input_single)

    single_ctx = ctx_traj_single[0, 0, :]

    # Repeated (10 times)
    input_repeated = torch.tensor([[token_id] * 10], dtype=torch.long, device=device)
    with torch.no_grad():
        logits_repeated, ctx_traj_repeated = model(input_repeated)

    repeated_ctx = ctx_traj_repeated[0, -1, :]

    # Compare
    l2_dist = torch.norm(single_ctx - repeated_ctx).item()
    cos_sim = torch.nn.functional.cosine_similarity(
        single_ctx.unsqueeze(0), repeated_ctx.unsqueeze(0)
    ).item()

    print(f"\nSingle-pass context (t=0): norm = {torch.norm(single_ctx).item():.6f}")
    print(f"Repeated context (t=9):    norm = {torch.norm(repeated_ctx).item():.6f}")
    print(f"\nL2 Distance: {l2_dist:.6f}")
    print(f"Cosine Similarity: {cos_sim:.6f}")

    # Check if they're the same
    if torch.allclose(single_ctx, repeated_ctx, atol=1e-3):
        print("\n🚨 SUSPICIOUS: Single-pass and repeated contexts are nearly identical!")
        print("   This suggests context is NOT changing with repetition")
    else:
        print("\n✅ Contexts differ appropriately")

    # Print first 10 dimensions for visual inspection
    print(f"\nFirst 10 dimensions comparison:")
    print(f"Single:   {single_ctx[:10].cpu().numpy()}")
    print(f"Repeated: {repeated_ctx[:10].cpu().numpy()}")


def main():
    device = torch.device('cpu')

    print("="*80)
    print("CVFPT Degenerate Solution Debug Script")
    print("="*80)

    # Load model
    print("\n📥 Loading model...")
    model, config = load_model('checkpoints/new_llm_repetition_final.pt', device)
    print(f"✓ Model loaded")
    print(f"  Context update strategy: {config.context_update_strategy}")
    print(f"  Context dim: {config.context_vector_dim}")

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Test with a few tokens
    test_tokens = [1000, 2000, 3000]  # Random token IDs

    for token_id in test_tokens:
        token_str = tokenizer.decode([token_id])
        print(f"\n{'='*80}")
        print(f"Testing Token: '{token_str}' (ID: {token_id})")
        print("="*80)

        # Debug trajectory
        debug_context_trajectory(model, token_id, num_reps=10, device=device)

        # Compare single vs repeated
        compare_single_vs_repeated(model, token_id, device=device)

    print(f"\n{'='*80}")
    print("Debug Complete")
    print("="*80)


if __name__ == "__main__":
    main()
