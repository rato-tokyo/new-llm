#!/usr/bin/env python3
"""
Test script for context vector expansion functionality

Tests the weight expansion logic without requiring actual checkpoints.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.utils.config import NewLLML4Config
from src.models.context_vector_llm import ContextVectorLLM


def create_dummy_checkpoint(num_layers, context_dim):
    """Create a dummy checkpoint for testing

    Args:
        num_layers: Number of layers
        context_dim: Context vector dimension

    Returns:
        Dict with model_state_dict
    """
    class DummyConfig(NewLLML4Config):
        max_seq_length = 64
        vocab_size = 1000
        embed_dim = 256
        hidden_dim = 512
        dropout = 0.1

        def __init__(self, num_layers_val, context_dim_val):
            super().__init__()
            self.num_layers = num_layers_val
            self.context_vector_dim = context_dim_val

    config = DummyConfig(num_layers, context_dim)
    model = ContextVectorLLM(config)

    return {
        'model_state_dict': model.state_dict(),
        'config': config
    }


def expand_context_vector_weights(old_state_dict, old_context_dim, new_context_dim, num_layers):
    """Expand context vector dimensions (copied from main script for testing)"""
    new_state_dict = {}

    print(f"\n🧠 Expanding context vector: {old_context_dim} → {new_context_dim}")
    print(f"   Strategy: Zero-padding new {new_context_dim - old_context_dim} dimensions\n")

    for key, value in old_state_dict.items():
        if 'context_proj' in key:
            if 'weight' in key:
                old_weight = value
                embed_dim = old_weight.size(1) - old_context_dim

                new_weight = torch.zeros(new_context_dim, embed_dim + new_context_dim)
                new_weight[:old_context_dim, :embed_dim] = old_weight[:, :embed_dim]
                new_weight[:old_context_dim, embed_dim:embed_dim+old_context_dim] = old_weight[:, embed_dim:]

                new_state_dict[key] = new_weight
                print(f"   ✓ {key}: {old_weight.shape} → {new_weight.shape}")

            elif 'bias' in key:
                old_bias = value
                new_bias = torch.zeros(new_context_dim)
                new_bias[:old_context_dim] = old_bias
                new_state_dict[key] = new_bias
                print(f"   ✓ {key}: {old_bias.shape} → {new_bias.shape}")

        elif 'layers' in key and 'fnn' in key:
            if 'weight' in key and '.0.weight' in key:
                old_weight = value
                embed_dim = old_weight.size(1) - old_context_dim
                hidden_dim = old_weight.size(0)

                new_weight = torch.zeros(hidden_dim, embed_dim + new_context_dim)
                new_weight[:, :embed_dim] = old_weight[:, :embed_dim]
                new_weight[:, embed_dim:embed_dim+old_context_dim] = old_weight[:, embed_dim:]

                new_state_dict[key] = new_weight
                print(f"   ✓ {key}: {old_weight.shape} → {new_weight.shape}")
            else:
                new_state_dict[key] = value

        elif 'context_update' in key:
            if 'weight' in key:
                old_weight = value
                hidden_dim = old_weight.size(1)

                new_weight = torch.zeros(new_context_dim, hidden_dim)
                new_weight[:old_context_dim, :] = old_weight

                new_state_dict[key] = new_weight
                print(f"   ✓ {key}: {old_weight.shape} → {new_weight.shape}")

            elif 'bias' in key:
                old_bias = value
                new_bias = torch.zeros(new_context_dim)
                new_bias[:old_context_dim] = old_bias
                new_state_dict[key] = new_bias
                print(f"   ✓ {key}: {old_bias.shape} → {new_bias.shape}")

        elif 'context_norm' in key:
            if 'weight' in key or 'bias' in key:
                old_param = value
                new_param = torch.zeros(new_context_dim) if 'bias' in key else torch.ones(new_context_dim)
                new_param[:old_context_dim] = old_param
                new_state_dict[key] = new_param
                print(f"   ✓ {key}: {old_param.shape} → {new_param.shape}")

        elif 'forget_gate' in key or 'input_gate' in key:
            if 'weight' in key:
                old_weight = value
                hidden_dim = old_weight.size(1)
                new_weight = torch.zeros(new_context_dim, hidden_dim)
                new_weight[:old_context_dim, :] = old_weight
                new_state_dict[key] = new_weight
                print(f"   ✓ {key}: {old_weight.shape} → {new_weight.shape}")
            elif 'bias' in key:
                old_bias = value
                new_bias = torch.zeros(new_context_dim)
                new_bias[:old_context_dim] = old_bias
                new_state_dict[key] = new_bias
                print(f"   ✓ {key}: {old_bias.shape} → {new_bias.shape}")

        else:
            new_state_dict[key] = value

    print(f"\n✓ Context vector expansion complete!")
    return new_state_dict


def test_weight_preservation():
    """Test that old weights are preserved correctly"""
    print("\n" + "="*80)
    print("TEST 1: Weight Preservation")
    print("="*80)

    old_context_dim = 256
    new_context_dim = 512
    num_layers = 1

    # Create dummy checkpoint
    print(f"\n📦 Creating dummy checkpoint (ctx={old_context_dim}, layers={num_layers})")
    checkpoint = create_dummy_checkpoint(num_layers, old_context_dim)
    old_state_dict = checkpoint['model_state_dict']

    # Expand
    expanded_state_dict = expand_context_vector_weights(
        old_state_dict, old_context_dim, new_context_dim, num_layers
    )

    # Verify preservation
    print(f"\n🔍 Verifying weight preservation...")
    success = True

    for key in old_state_dict.keys():
        if 'context_proj.weight' in key:
            old_weight = old_state_dict[key]
            new_weight = expanded_state_dict[key]

            # Check that old weights are preserved in the top-left block
            embed_dim = old_weight.size(1) - old_context_dim
            preserved_embed = new_weight[:old_context_dim, :embed_dim]
            original_embed = old_weight[:, :embed_dim]

            if not torch.allclose(preserved_embed, original_embed):
                print(f"   ✗ {key}: Embed weights NOT preserved")
                success = False
            else:
                print(f"   ✓ {key}: Embed weights preserved")

            preserved_context = new_weight[:old_context_dim, embed_dim:embed_dim+old_context_dim]
            original_context = old_weight[:, embed_dim:]

            if not torch.allclose(preserved_context, original_context):
                print(f"   ✗ {key}: Context weights NOT preserved")
                success = False
            else:
                print(f"   ✓ {key}: Context weights preserved")

    if success:
        print(f"\n✅ TEST 1 PASSED: All weights preserved correctly")
    else:
        print(f"\n❌ TEST 1 FAILED: Some weights not preserved")

    return success


def test_zero_initialization():
    """Test that new dimensions are zero-initialized"""
    print("\n" + "="*80)
    print("TEST 2: Zero Initialization of New Dimensions")
    print("="*80)

    old_context_dim = 256
    new_context_dim = 512
    num_layers = 1

    checkpoint = create_dummy_checkpoint(num_layers, old_context_dim)
    old_state_dict = checkpoint['model_state_dict']

    expanded_state_dict = expand_context_vector_weights(
        old_state_dict, old_context_dim, new_context_dim, num_layers
    )

    print(f"\n🔍 Verifying zero initialization...")
    success = True

    for key in expanded_state_dict.keys():
        if 'context_proj.weight' in key:
            new_weight = expanded_state_dict[key]
            embed_dim = 256

            # Check new context dimensions (columns)
            new_context_cols = new_weight[:old_context_dim, embed_dim+old_context_dim:]
            if not torch.allclose(new_context_cols, torch.zeros_like(new_context_cols)):
                print(f"   ✗ {key}: New context columns NOT zero")
                success = False
            else:
                print(f"   ✓ {key}: New context columns are zero ({new_context_cols.shape})")

            # Check new rows
            new_rows = new_weight[old_context_dim:, :]
            if not torch.allclose(new_rows, torch.zeros_like(new_rows)):
                print(f"   ✗ {key}: New rows NOT zero")
                success = False
            else:
                print(f"   ✓ {key}: New rows are zero ({new_rows.shape})")

        elif 'context_update.weight' in key:
            new_weight = expanded_state_dict[key]
            new_rows = new_weight[old_context_dim:, :]
            if not torch.allclose(new_rows, torch.zeros_like(new_rows)):
                print(f"   ✗ {key}: New rows NOT zero")
                success = False
            else:
                print(f"   ✓ {key}: New rows are zero ({new_rows.shape})")

    if success:
        print(f"\n✅ TEST 2 PASSED: All new dimensions zero-initialized")
    else:
        print(f"\n❌ TEST 2 FAILED: Some new dimensions not zero")

    return success


def test_model_loading():
    """Test that expanded weights can be loaded into a new model"""
    print("\n" + "="*80)
    print("TEST 3: Model Loading with Expanded Weights")
    print("="*80)

    old_context_dim = 256
    new_context_dim = 512
    num_layers = 1

    # Create and expand
    checkpoint = create_dummy_checkpoint(num_layers, old_context_dim)
    old_state_dict = checkpoint['model_state_dict']

    expanded_state_dict = expand_context_vector_weights(
        old_state_dict, old_context_dim, new_context_dim, num_layers
    )

    # Create new model with expanded dimensions
    class ExpandedConfig(NewLLML4Config):
        max_seq_length = 64
        vocab_size = 1000
        embed_dim = 256
        hidden_dim = 512
        dropout = 0.1

        def __init__(self, num_layers_val, context_dim_val):
            super().__init__()
            self.num_layers = num_layers_val
            self.context_vector_dim = context_dim_val

    print(f"\n📦 Creating expanded model (ctx={new_context_dim})")
    expanded_model = ContextVectorLLM(ExpandedConfig(num_layers, new_context_dim))

    # Try to load expanded weights
    try:
        expanded_model.load_state_dict(expanded_state_dict)
        print(f"✓ Expanded weights loaded successfully")

        # Verify model can do forward pass
        batch_size = 4
        seq_len = 10
        dummy_input = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            output = expanded_model(dummy_input)

        # Handle both tuple and tensor outputs
        if isinstance(output, tuple):
            logits, context = output
            print(f"✓ Forward pass successful: logits shape {logits.shape}, context shape {context.shape}")
        else:
            print(f"✓ Forward pass successful: output shape {output.shape}")
        print(f"\n✅ TEST 3 PASSED: Model loads and runs with expanded weights")
        return True

    except Exception as e:
        print(f"✗ Failed to load or run model: {e}")
        print(f"\n❌ TEST 3 FAILED")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("Context Vector Expansion Test Suite")
    print("="*80)

    results = []

    # Run tests
    results.append(("Weight Preservation", test_weight_preservation()))
    results.append(("Zero Initialization", test_zero_initialization()))
    results.append(("Model Loading", test_model_loading()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
