"""Test fixed-point convergence with single token

This script tests the convergence behavior with just 1 token,
comparing Sequential vs Layer-wise architectures.
"""

import torch
from src.models.new_llm_sequential import NewLLMSequential
from src.models.new_llm_layerwise import NewLLMLayerwise
from src.utils.dialogue_config import Small2LayerSequentialConfig, Small2LayerLayerwiseConfig


def test_single_token(model, config, token_id=100):
    """Test fixed-point convergence with single token"""
    print(f"\n{'='*60}")
    print(f"Architecture: {config.architecture.upper()}")
    print(f"Token ID: {token_id}")
    print(f"{'='*60}")

    # Single token
    input_ids = torch.tensor([[token_id]])  # [1, 1]

    # Get fixed-point context
    fixed_contexts, converged, num_iters = model.get_fixed_point_context(
        input_ids,
        max_iterations=config.phase1_max_iterations,
        tolerance=config.phase1_convergence_threshold,
        warmup_iterations=config.phase1_warmup_iterations
    )

    print(f"\nResults:")
    print(f"  Converged: {converged[0, 0].item()}")
    print(f"  Iterations: {num_iters[0, 0].item()}")
    print(f"  Context vector norm: {torch.norm(fixed_contexts[0, 0]).item():.4f}")
    print(f"  Context vector (first 10 dims): {fixed_contexts[0, 0, :10].tolist()}")

    return converged[0, 0].item(), num_iters[0, 0].item()


def main():
    """Run single token test for both architectures"""
    print("\n" + "="*60)
    print("Single Token Convergence Test")
    print("Sequential vs Layer-wise Architecture")
    print("="*60)

    # Test Sequential
    config_seq = Small2LayerSequentialConfig()
    config_seq.vocab_size = 50259  # Set vocab_size manually
    model_seq = NewLLMSequential(config_seq)
    model_seq.eval()

    converged_seq, iters_seq = test_single_token(model_seq, config_seq)

    # Test Layer-wise
    config_layer = Small2LayerLayerwiseConfig()
    config_layer.vocab_size = 50259  # Set vocab_size manually
    model_layer = NewLLMLayerwise(config_layer)
    model_layer.eval()

    converged_layer, iters_layer = test_single_token(model_layer, config_layer)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Sequential:")
    print(f"  Converged: {converged_seq}")
    print(f"  Iterations: {iters_seq}")
    print(f"\nLayer-wise:")
    print(f"  Converged: {converged_layer}")
    print(f"  Iterations: {iters_layer}")
    print("="*60)


if __name__ == "__main__":
    main()
