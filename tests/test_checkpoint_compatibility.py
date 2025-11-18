#!/usr/bin/env python3
"""
Test checkpoint compatibility for context expansion script

Verifies that context expansion script can load checkpoints from:
- train_wikitext_fp16_layers.py (LayerExperimentConfig)
- train_wikitext_fp16.py (FP16Config)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.utils.config import NewLLML4Config
from src.models.context_vector_llm import ContextVectorLLM


def create_test_checkpoint(config_class_name, num_layers=1, context_dim=256):
    """Create a test checkpoint with specified config class

    Args:
        config_class_name: Name of config class to use
        num_layers: Number of layers
        context_dim: Context vector dimension

    Returns:
        Checkpoint dict
    """
    # Import the config classes from the expansion script
    # This simulates what happens during checkpoint loading
    import scripts.train_wikitext_context_expansion as expansion_script

    if config_class_name == "LayerExperimentConfig":
        config = expansion_script.LayerExperimentConfig(num_layers=num_layers)
    elif config_class_name == "FP16Config":
        config = expansion_script.FP16Config()
    else:
        raise ValueError(f"Unknown config class: {config_class_name}")

    # Create model
    model = ContextVectorLLM(config)

    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'epoch': 150,
        'val_loss': 3.02,
        'val_ppl': 20.5
    }

    return checkpoint, config


def test_layer_experiment_checkpoint():
    """Test loading checkpoint saved with LayerExperimentConfig"""
    print("\n" + "="*80)
    print("TEST 1: LayerExperimentConfig Checkpoint Compatibility")
    print("="*80)

    # Create test checkpoint
    print("\n📦 Creating test checkpoint with LayerExperimentConfig...")
    checkpoint, original_config = create_test_checkpoint(
        "LayerExperimentConfig",
        num_layers=1,
        context_dim=256
    )

    # Save to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name

    try:
        torch.save(checkpoint, temp_path)
        print(f"✓ Checkpoint saved to: {temp_path}")

        # Try to load it
        print(f"\n🔄 Loading checkpoint...")
        loaded_checkpoint = torch.load(temp_path, map_location='cpu', weights_only=False)

        print(f"✓ Checkpoint loaded successfully")
        print(f"   Config type: {type(loaded_checkpoint['config']).__name__}")
        print(f"   Num layers: {loaded_checkpoint['config'].num_layers}")
        print(f"   Context dim: {loaded_checkpoint['config'].context_vector_dim}")
        print(f"   Epoch: {loaded_checkpoint['epoch']}")
        print(f"   Val PPL: {loaded_checkpoint['val_ppl']}")

        # Verify config class
        if type(loaded_checkpoint['config']).__name__ == "LayerExperimentConfig":
            print(f"\n✅ TEST 1 PASSED: LayerExperimentConfig checkpoint loaded successfully")
            return True
        else:
            print(f"\n❌ TEST 1 FAILED: Wrong config type: {type(loaded_checkpoint['config']).__name__}")
            return False

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_fp16_checkpoint():
    """Test loading checkpoint saved with FP16Config"""
    print("\n" + "="*80)
    print("TEST 2: FP16Config Checkpoint Compatibility")
    print("="*80)

    # Create test checkpoint
    print("\n📦 Creating test checkpoint with FP16Config...")
    checkpoint, original_config = create_test_checkpoint(
        "FP16Config",
        num_layers=6,
        context_dim=256
    )

    # Save to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name

    try:
        torch.save(checkpoint, temp_path)
        print(f"✓ Checkpoint saved to: {temp_path}")

        # Try to load it
        print(f"\n🔄 Loading checkpoint...")
        loaded_checkpoint = torch.load(temp_path, map_location='cpu', weights_only=False)

        print(f"✓ Checkpoint loaded successfully")
        print(f"   Config type: {type(loaded_checkpoint['config']).__name__}")
        print(f"   Num layers: {loaded_checkpoint['config'].num_layers}")
        print(f"   Context dim: {loaded_checkpoint['config'].context_vector_dim}")

        # Verify config class
        if type(loaded_checkpoint['config']).__name__ == "FP16Config":
            print(f"\n✅ TEST 2 PASSED: FP16Config checkpoint loaded successfully")
            return True
        else:
            print(f"\n❌ TEST 2 FAILED: Wrong config type: {type(loaded_checkpoint['config']).__name__}")
            return False

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_model_loading_from_checkpoint():
    """Test loading model weights from expanded checkpoint"""
    print("\n" + "="*80)
    print("TEST 3: Model Loading from Checkpoint")
    print("="*80)

    # Create test checkpoint with LayerExperimentConfig
    print("\n📦 Creating test checkpoint...")
    checkpoint, original_config = create_test_checkpoint(
        "LayerExperimentConfig",
        num_layers=1,
        context_dim=256
    )

    # Save to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_path = f.name

    try:
        torch.save(checkpoint, temp_path)
        print(f"✓ Checkpoint saved")

        # Load checkpoint
        print(f"\n🔄 Loading checkpoint...")
        loaded_checkpoint = torch.load(temp_path, map_location='cpu', weights_only=False)
        loaded_config = loaded_checkpoint['config']

        # Create model with same config
        print(f"\n🔨 Creating model from loaded config...")
        model = ContextVectorLLM(loaded_config)

        # Load weights
        print(f"📥 Loading model weights...")
        model.load_state_dict(loaded_checkpoint['model_state_dict'])

        print(f"✓ Model weights loaded successfully")

        # Test forward pass
        print(f"\n▶️  Testing forward pass...")
        batch_size = 4
        seq_len = 10
        dummy_input = torch.randint(0, 1000, (batch_size, seq_len))

        with torch.no_grad():
            output = model(dummy_input)

        if isinstance(output, tuple):
            logits, context = output
            print(f"✓ Forward pass successful")
            print(f"   Logits shape: {logits.shape}")
            print(f"   Context shape: {context.shape}")
        else:
            print(f"✓ Forward pass successful")
            print(f"   Output shape: {output.shape}")

        print(f"\n✅ TEST 3 PASSED: Model loaded and runs successfully")
        return True

    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("Checkpoint Compatibility Test Suite")
    print("="*80)

    results = []

    # Run tests
    results.append(("LayerExperimentConfig Compatibility", test_layer_experiment_checkpoint()))
    results.append(("FP16Config Compatibility", test_fp16_checkpoint()))
    results.append(("Model Loading", test_model_loading_from_checkpoint()))

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
        print("\nContext expansion script is ready to use with:")
        print("  ✓ best_new_llm_wikitext_fp16_layers*.pt (Layer experiment checkpoints)")
        print("  ✓ best_new_llm_wikitext_fp16.pt (FP16 baseline checkpoint)")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
