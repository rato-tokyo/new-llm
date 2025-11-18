#!/usr/bin/env python3
"""
Test HH-RLHF training script components

Verifies:
1. HHRLHFTrainConfig creation
2. HHRLHFDataset creation with mock data
3. Model creation with HH-RLHF config
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch


def test_hh_rlhf_config():
    """Test HHRLHFTrainConfig creation"""
    print("\n" + "="*80)
    print("TEST 1: HHRLHFTrainConfig Creation")
    print("="*80)

    # Import config
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from train_hh_rlhf import HHRLHFTrainConfig

    # Create config
    config = HHRLHFTrainConfig(num_layers=1)

    print(f"\n✓ Config created")
    print(f"   Num layers: {config.num_layers}")
    print(f"   Max seq length: {config.max_seq_length}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Vocab size: {config.vocab_size}")

    # Verify
    assert config.num_layers == 1
    assert config.max_seq_length == 128  # Longer for dialog
    assert config.vocab_size == 1000

    print(f"\n✅ TEST 1 PASSED")
    return True


def test_hh_rlhf_dataset_mock():
    """Test HHRLHFDataset with mock data"""
    print("\n" + "="*80)
    print("TEST 2: HHRLHFDataset with Mock Data")
    print("="*80)

    from src.training.hh_rlhf_dataset import HHRLHFDataset
    from src.training.dataset import SimpleTokenizer

    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.build_vocab(["hello world", "how are you", "human assistant conversation"])

    # Mock HH-RLHF data
    mock_data = [
        {
            'chosen': 'Human: What is AI?\n\nAssistant: Artificial Intelligence is...',
            'rejected': 'Human: What is AI?\n\nAssistant: I dont know'
        },
        {
            'chosen': 'Human: How do I learn Python?\n\nAssistant: Start with basics...',
            'rejected': 'Human: How do I learn Python?\n\nAssistant: Use Google'
        }
    ]

    # Create dataset
    dataset = HHRLHFDataset(mock_data, tokenizer, max_length=128)

    print(f"\n✓ Dataset created")
    print(f"   Num sequences: {len(dataset)}")
    print(f"   Sequence length: {len(dataset[0][0])}")

    # Verify
    assert len(dataset) == 2
    assert len(dataset[0][0]) == 127  # max_length - 1 (for input)
    assert len(dataset[0][1]) == 127  # max_length - 1 (for target)

    print(f"\n✅ TEST 2 PASSED")
    return True


def test_model_creation():
    """Test model creation with HH-RLHF config"""
    print("\n" + "="*80)
    print("TEST 3: Model Creation with HH-RLHF Config")
    print("="*80)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from train_hh_rlhf import HHRLHFTrainConfig
    from src.models.context_vector_llm import ContextVectorLLM

    # Create config
    config = HHRLHFTrainConfig(num_layers=1)
    config.device = 'cpu'  # CPU for testing

    # Create model
    model = ContextVectorLLM(config).to(config.device)

    print(f"\n✓ Model created")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Test forward pass
    print(f"\n▶️  Testing forward pass...")
    batch_size = 4
    seq_len = 127  # max_seq_length - 1
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

    print(f"\n✅ TEST 3 PASSED")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("HH-RLHF Training Script Test Suite")
    print("="*80)

    results = []

    # Run tests
    results.append(("HHRLHFTrainConfig", test_hh_rlhf_config()))
    results.append(("HHRLHFDataset Mock", test_hh_rlhf_dataset_mock()))
    results.append(("Model Creation", test_model_creation()))

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
        print("\nHH-RLHF training script is ready to use:")
        print("  python scripts/train_hh_rlhf.py --num_layers 1")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
