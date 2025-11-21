"""Test fixed-point convergence with 10 tokens

This script tests the convergence behavior with 10 tokens,
comparing Sequential vs Layer-wise architectures.
"""

import torch
from src.models.new_llm_sequential import NewLLMSequential
from src.models.new_llm_layerwise import NewLLMLayerwise
from src.utils.dialogue_config import Small2LayerSequentialConfig, Small2LayerLayerwiseConfig


def test_10tokens(model, config):
    """Test fixed-point convergence with 10 tokens"""
    print(f"\n{'='*60}")
    print(f"Architecture: {config.architecture.upper()}")
    print(f"{'='*60}")

    # 10 tokens (arbitrary token IDs)
    token_ids = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    input_ids = torch.tensor([token_ids])  # [1, 10]

    print(f"Testing with {len(token_ids)} tokens: {token_ids}")

    # Get fixed-point context
    fixed_contexts, converged, num_iters = model.get_fixed_point_context(
        input_ids,
        max_iterations=config.phase1_max_iterations,
        tolerance=config.phase1_convergence_threshold,
        warmup_iterations=config.phase1_warmup_iterations
    )

    # Statistics
    converged_count = converged[0].sum().item()
    total_count = converged[0].numel()
    avg_iters = num_iters[0].float().mean().item()

    print(f"\nResults:")
    print(f"  Converged: {converged_count}/{total_count} ({converged_count/total_count*100:.1f}%)")
    print(f"  Average iterations: {avg_iters:.1f}")
    print(f"  Iterations per token: {num_iters[0].tolist()}")

    return converged_count, total_count, avg_iters


def main():
    """Run 10-token test for both architectures"""
    print("\n" + "="*60)
    print("10 Tokens Convergence Test")
    print("Sequential vs Layer-wise Architecture")
    print("="*60)

    # Test Sequential
    config_seq = Small2LayerSequentialConfig()
    config_seq.vocab_size = 50259  # Set vocab_size manually
    model_seq = NewLLMSequential(config_seq)
    model_seq.eval()

    converged_seq, total_seq, avg_iters_seq = test_10tokens(model_seq, config_seq)

    # Test Layer-wise
    config_layer = Small2LayerLayerwiseConfig()
    config_layer.vocab_size = 50259  # Set vocab_size manually
    model_layer = NewLLMLayerwise(config_layer)
    model_layer.eval()

    converged_layer, total_layer, avg_iters_layer = test_10tokens(model_layer, config_layer)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Sequential:")
    print(f"  Converged: {converged_seq}/{total_seq} ({converged_seq/total_seq*100:.1f}%)")
    print(f"  Average iterations: {avg_iters_seq:.1f}")
    print(f"\nLayer-wise:")
    print(f"  Converged: {converged_layer}/{total_layer} ({converged_layer/total_layer*100:.1f}%)")
    print(f"  Average iterations: {avg_iters_layer:.1f}")
    print("="*60)


if __name__ == "__main__":
    main()
