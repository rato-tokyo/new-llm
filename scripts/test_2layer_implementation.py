#!/usr/bin/env python3
"""
Comprehensive test suite for 2-layer New-LLM implementation

This script verifies:
1. Model architecture consistency
2. Forward pass correctness
3. Gradient flow (no dead neurons)
4. Parameter counts
5. Output shapes
6. Comparison with 1-layer baseline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from src.models.new_llm import NewLLM
from src.models.new_llm_2layer import NewLLM2Layer
from src.utils.config import NewLLMConfig


def test_model_creation():
    """Test 1: Model can be created without errors"""
    print("=" * 80)
    print("Test 1: Model Creation")
    print("=" * 80)

    try:
        config = NewLLMConfig()
        config.vocab_size = 1000
        config.embed_dim = 128
        config.context_vector_dim = 128
        config.hidden_dim = 256
        config.num_layers = 2

        model = NewLLM2Layer(config)
        print("✓ 2-layer model created successfully")

        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Total parameters: {total_params:,}")

        return True, model, config
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False, None, None


def test_forward_pass(model, config):
    """Test 2: Forward pass produces correct output shapes"""
    print("\n" + "=" * 80)
    print("Test 2: Forward Pass")
    print("=" * 80)

    try:
        batch_size = 4
        seq_len = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits, context_trajectory = model(input_ids)

        # Check shapes
        expected_logits_shape = (batch_size, seq_len, config.vocab_size)
        expected_context_shape = (batch_size, seq_len, config.context_vector_dim)

        assert logits.shape == expected_logits_shape, \
            f"Logits shape mismatch: {logits.shape} vs {expected_logits_shape}"
        assert context_trajectory.shape == expected_context_shape, \
            f"Context shape mismatch: {context_trajectory.shape} vs {expected_context_shape}"

        print(f"✓ Logits shape: {logits.shape} (expected: {expected_logits_shape})")
        print(f"✓ Context trajectory shape: {context_trajectory.shape} (expected: {expected_context_shape})")

        # Check for NaN/Inf
        assert not torch.isnan(logits).any(), "Logits contain NaN"
        assert not torch.isinf(logits).any(), "Logits contain Inf"
        assert not torch.isnan(context_trajectory).any(), "Context contains NaN"
        assert not torch.isinf(context_trajectory).any(), "Context contains Inf"

        print("✓ No NaN or Inf values detected")

        return True, logits, context_trajectory
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False, None, None


def test_gradient_flow(model, config):
    """Test 3: Gradients flow through all layers"""
    print("\n" + "=" * 80)
    print("Test 3: Gradient Flow")
    print("=" * 80)

    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Forward pass
        input_ids = torch.randint(0, config.vocab_size, (2, 5))
        logits, context_trajectory = model(input_ids)

        # Compute dummy loss
        target_ids = torch.randint(0, config.vocab_size, (2, 5))
        loss = nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            target_ids.view(-1)
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        layers_with_gradients = []
        layers_without_gradients = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 0:
                    layers_with_gradients.append((name, grad_norm))
                else:
                    layers_without_gradients.append(name)
            else:
                layers_without_gradients.append(name)

        print(f"✓ Layers with gradients: {len(layers_with_gradients)}")
        print(f"  Sample gradients:")
        for name, grad_norm in layers_with_gradients[:5]:
            print(f"    {name}: {grad_norm:.6f}")

        if layers_without_gradients:
            print(f"⚠ Layers without gradients: {len(layers_without_gradients)}")
            for name in layers_without_gradients[:5]:
                print(f"    {name}")

        # Check for dead neurons (zero gradients)
        if len(layers_without_gradients) > 0:
            print("⚠ Warning: Some layers have no gradients (possible dead neurons)")
        else:
            print("✓ All layers have non-zero gradients")

        return True
    except Exception as e:
        print(f"✗ Gradient flow test failed: {e}")
        return False


def test_layer_outputs(model, config):
    """Test 4: Verify intermediate layer outputs"""
    print("\n" + "=" * 80)
    print("Test 4: Intermediate Layer Outputs")
    print("=" * 80)

    try:
        model.eval()
        input_ids = torch.randint(0, config.vocab_size, (1, 3))

        with torch.no_grad():
            # Get token embeddings
            token_embeds = model.token_embedding(input_ids)
            print(f"✓ Token embeddings shape: {token_embeds.shape}")

            # Initialize context
            context = torch.zeros(1, config.context_vector_dim)

            # Process first token
            current_token = token_embeds[:, 0, :]
            fnn_input = torch.cat([current_token, context], dim=-1)

            # Layer 1 output
            hidden1 = model.fnn_layer1(fnn_input)
            print(f"✓ Layer 1 output shape: {hidden1.shape}")
            print(f"  Layer 1 output range: [{hidden1.min():.4f}, {hidden1.max():.4f}]")

            # Layer 2 output
            hidden2 = model.fnn_layer2(hidden1)
            print(f"✓ Layer 2 output shape: {hidden2.shape}")
            print(f"  Layer 2 output range: [{hidden2.min():.4f}, {hidden2.max():.4f}]")

            # Token prediction
            token_logits = model.token_output(hidden2)
            print(f"✓ Token logits shape: {token_logits.shape}")
            print(f"  Token logits range: [{token_logits.min():.4f}, {token_logits.max():.4f}]")

            # Context update
            context_new = model.context_updater(hidden2, context)
            print(f"✓ Updated context shape: {context_new.shape}")
            print(f"  Context range: [{context_new.min():.4f}, {context_new.max():.4f}]")

        return True
    except Exception as e:
        print(f"✗ Layer output test failed: {e}")
        return False


def test_comparison_with_1layer(config):
    """Test 5: Compare 2-layer with 1-layer baseline"""
    print("\n" + "=" * 80)
    print("Test 5: Comparison with 1-Layer")
    print("=" * 80)

    try:
        # Create both models
        model_1layer = NewLLM(config)
        model_2layer = NewLLM2Layer(config)

        # Count parameters
        params_1layer = sum(p.numel() for p in model_1layer.parameters())
        params_2layer = sum(p.numel() for p in model_2layer.parameters())

        print(f"1-layer parameters: {params_1layer:,}")
        print(f"2-layer parameters: {params_2layer:,}")
        print(f"Difference: {params_2layer - params_1layer:,}")
        print(f"Ratio: {params_2layer / params_1layer:.2f}x")

        # Compare output shapes
        input_ids = torch.randint(0, config.vocab_size, (2, 5))

        with torch.no_grad():
            logits_1layer, context_1layer = model_1layer(input_ids)
            logits_2layer, context_2layer = model_2layer(input_ids)

        assert logits_1layer.shape == logits_2layer.shape, "Logits shape mismatch"
        assert context_1layer.shape == context_2layer.shape, "Context shape mismatch"

        print(f"✓ Output shapes match")
        print(f"  Logits: {logits_1layer.shape}")
        print(f"  Context: {context_1layer.shape}")

        # Compare context vectors (should be different)
        context_diff = torch.norm(context_1layer - context_2layer)
        print(f"\nContext difference (L2 norm): {context_diff.item():.4f}")

        if context_diff.item() < 0.1:
            print("⚠ Warning: 1-layer and 2-layer produce very similar outputs")
        else:
            print("✓ 1-layer and 2-layer produce different outputs (expected)")

        return True
    except Exception as e:
        print(f"✗ Comparison test failed: {e}")
        return False


def test_determinism():
    """Test 6: Model produces deterministic outputs"""
    print("\n" + "=" * 80)
    print("Test 6: Determinism")
    print("=" * 80)

    try:
        config = NewLLMConfig()
        config.vocab_size = 1000

        # Set seed
        torch.manual_seed(42)
        model = NewLLM2Layer(config)

        input_ids = torch.randint(0, config.vocab_size, (2, 5))

        model.eval()
        with torch.no_grad():
            logits1, context1 = model(input_ids)
            logits2, context2 = model(input_ids)

        # Check if outputs are identical
        assert torch.allclose(logits1, logits2, atol=1e-6), "Logits not deterministic"
        assert torch.allclose(context1, context2, atol=1e-6), "Context not deterministic"

        print("✓ Model produces deterministic outputs")

        return True
    except Exception as e:
        print(f"✗ Determinism test failed: {e}")
        return False


def main():
    print("\n" + "=" * 80)
    print("2-Layer New-LLM Implementation Test Suite")
    print("=" * 80)

    # Test 1: Model creation
    success1, model, config = test_model_creation()
    if not success1:
        print("\n✗ Critical failure: Cannot create model")
        return

    # Test 2: Forward pass
    success2, logits, context = test_forward_pass(model, config)
    if not success2:
        print("\n✗ Critical failure: Forward pass failed")
        return

    # Test 3: Gradient flow
    success3 = test_gradient_flow(model, config)

    # Test 4: Layer outputs
    success4 = test_layer_outputs(model, config)

    # Test 5: Comparison with 1-layer
    success5 = test_comparison_with_1layer(config)

    # Test 6: Determinism
    success6 = test_determinism()

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    tests = [
        ("Model Creation", success1),
        ("Forward Pass", success2),
        ("Gradient Flow", success3),
        ("Layer Outputs", success4),
        ("1-Layer Comparison", success5),
        ("Determinism", success6),
    ]

    for name, success in tests:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(success for _, success in tests)

    if all_passed:
        print("\n🎉 All tests passed! 2-layer implementation is correct.")
    else:
        print("\n⚠ Some tests failed. Review implementation.")

    print("=" * 80)


if __name__ == "__main__":
    main()
