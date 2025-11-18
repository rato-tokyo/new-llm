#!/usr/bin/env python3
"""
Test gradient freezing functionality in ContextExpansionTrainer

Verifies that freeze_base_dims correctly freezes base dimension gradients.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.utils.config import NewLLML4Config
from src.models.context_vector_llm import ContextVectorLLM
from src.training.context_expansion_trainer import ContextExpansionTrainer
from torch.utils.data import TensorDataset, DataLoader


def create_test_model_and_trainer(context_dim, freeze_base_dims, base_context_dim=256):
    """Create a test model and trainer for gradient freezing tests"""

    class TestConfig(NewLLML4Config):
        max_seq_length = 64
        vocab_size = 100
        embed_dim = 128
        hidden_dim = 256
        dropout = 0.1

        def __init__(self, num_layers_val, context_dim_val):
            super().__init__()
            self.num_layers = num_layers_val
            self.context_vector_dim = context_dim_val

    config = TestConfig(num_layers_val=1, context_dim_val=context_dim)
    model = ContextVectorLLM(config)

    # Create dummy dataset
    batch_size = 4
    seq_len = 10
    num_samples = 20

    inputs = torch.randint(0, 100, (num_samples, seq_len))
    targets = torch.randint(0, 100, (num_samples, seq_len))

    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Create trainer (CPU mode for testing)
    config.device = "cpu"
    model = model.to("cpu")

    trainer = ContextExpansionTrainer(
        model=model,
        train_dataloader=dataloader,
        val_dataloader=dataloader,
        config=config,
        model_name="test_model",
        use_amp=False,  # Disable AMP for CPU testing
        base_context_dim=base_context_dim,
        freeze_base_dims=freeze_base_dims
    )

    # Override scaler for CPU testing
    trainer.scaler = None

    return model, trainer, config


def test_freeze_mode_gradients():
    """Test that freeze mode correctly zeros base dimension gradients"""
    print("\n" + "="*80)
    print("TEST 1: Freeze Mode Gradient Zeroing")
    print("="*80)

    base_dim = 256
    expanded_dim = 512
    model, trainer, config = create_test_model_and_trainer(
        context_dim=expanded_dim,
        freeze_base_dims=True,
        base_context_dim=base_dim
    )

    print(f"\n📦 Created model: ctx={expanded_dim}, freeze_base_dims=True")
    print(f"   Base dims (0:{base_dim}): Should be FROZEN")
    print(f"   New dims ({base_dim}:{expanded_dim}): Should be TRAINABLE")

    # Get a batch
    inputs, targets = next(iter(trainer.train_dataloader))

    # Forward pass
    model.train()
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs

    # Backward pass
    from src.evaluation.metrics import compute_loss
    loss = compute_loss(logits, targets, pad_idx=0)
    loss.backward()

    # Apply freeze
    trainer.freeze_base_gradients()

    # Check gradients
    print(f"\n🔍 Verifying frozen gradients...")
    success = True

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        if 'context_update.weight' in name:
            # Shape: [expanded_dim, hidden_dim]
            base_grads = param.grad[:base_dim, :]
            new_grads = param.grad[base_dim:, :]

            if torch.allclose(base_grads, torch.zeros_like(base_grads)):
                print(f"   ✓ {name}: Base dims FROZEN ({base_grads.shape})")
            else:
                print(f"   ✗ {name}: Base dims NOT frozen (max grad: {base_grads.abs().max():.6f})")
                success = False

            if not torch.allclose(new_grads, torch.zeros_like(new_grads)):
                print(f"   ✓ {name}: New dims TRAINABLE (max grad: {new_grads.abs().max():.6f})")
            else:
                print(f"   ✗ {name}: New dims have zero gradients (should be trainable!)")
                success = False

        elif 'context_update.bias' in name:
            # Shape: [expanded_dim]
            base_grads = param.grad[:base_dim]
            new_grads = param.grad[base_dim:]

            if torch.allclose(base_grads, torch.zeros_like(base_grads)):
                print(f"   ✓ {name}: Base dims FROZEN ({base_grads.shape})")
            else:
                print(f"   ✗ {name}: Base dims NOT frozen")
                success = False

    if success:
        print(f"\n✅ TEST 1 PASSED: Base dimensions frozen, new dimensions trainable")
    else:
        print(f"\n❌ TEST 1 FAILED")

    return success


def test_finetune_mode_gradients():
    """Test that fine-tune mode allows all gradients"""
    print("\n" + "="*80)
    print("TEST 2: Fine-tune Mode Gradient Flow")
    print("="*80)

    base_dim = 256
    expanded_dim = 512
    model, trainer, config = create_test_model_and_trainer(
        context_dim=expanded_dim,
        freeze_base_dims=False,  # Fine-tune mode
        base_context_dim=base_dim
    )

    print(f"\n📦 Created model: ctx={expanded_dim}, freeze_base_dims=False")
    print(f"   All dims (0:{expanded_dim}): Should be TRAINABLE")

    # Get a batch
    inputs, targets = next(iter(trainer.train_dataloader))

    # Forward pass
    model.train()
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        logits = outputs

    # Backward pass
    from src.evaluation.metrics import compute_loss
    loss = compute_loss(logits, targets, pad_idx=0)
    loss.backward()

    # Apply freeze (should do nothing in fine-tune mode)
    trainer.freeze_base_gradients()

    # Check gradients
    print(f"\n🔍 Verifying all gradients trainable...")
    success = True

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        if 'context_update.weight' in name:
            # All gradients should be non-zero
            all_grads = param.grad

            if not torch.allclose(all_grads, torch.zeros_like(all_grads)):
                print(f"   ✓ {name}: All dims TRAINABLE (max grad: {all_grads.abs().max():.6f})")
            else:
                print(f"   ✗ {name}: Has zero gradients (unexpected!)")
                success = False

    if success:
        print(f"\n✅ TEST 2 PASSED: All dimensions trainable in fine-tune mode")
    else:
        print(f"\n❌ TEST 2 FAILED")

    return success


def test_parameter_count():
    """Test parameter counting logic"""
    print("\n" + "="*80)
    print("TEST 3: Parameter Count Analysis")
    print("="*80)

    base_dim = 256
    expanded_dim = 512

    # Test freeze mode
    print(f"\n📊 Freeze Mode Analysis:")
    model, trainer, config = create_test_model_and_trainer(
        context_dim=expanded_dim,
        freeze_base_dims=True,
        base_context_dim=base_dim
    )

    trainer.print_trainable_params()

    # Test fine-tune mode
    print(f"\n📊 Fine-tune Mode Analysis:")
    model, trainer, config = create_test_model_and_trainer(
        context_dim=expanded_dim,
        freeze_base_dims=False,
        base_context_dim=base_dim
    )

    trainer.print_trainable_params()

    print(f"\n✅ TEST 3 PASSED: Parameter analysis displayed")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("Gradient Freezing Test Suite")
    print("="*80)

    results = []

    # Run tests
    results.append(("Freeze Mode Gradients", test_freeze_mode_gradients()))
    results.append(("Fine-tune Mode Gradients", test_finetune_mode_gradients()))
    results.append(("Parameter Count", test_parameter_count()))

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
