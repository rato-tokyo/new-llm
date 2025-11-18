#!/usr/bin/env python3
"""
Test UltraChat training script components

Verifies:
1. UltraChatTrainConfig creation
2. UltraChatDataset creation with mock data
3. Model creation with UltraChat config
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch


def test_ultrachat_config():
    """Test UltraChatTrainConfig creation"""
    print("\n" + "="*80)
    print("TEST 1: UltraChatTrainConfig Creation")
    print("="*80)

    # Import config
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from train_ultrachat import UltraChatTrainConfig

    # Create config
    config = UltraChatTrainConfig(num_layers=1)

    print(f"\n✓ Config created")
    print(f"   Num layers: {config.num_layers}")
    print(f"   Max seq length: {config.max_seq_length}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Num epochs: {config.num_epochs}")

    # Verify
    assert config.num_layers == 1
    assert config.max_seq_length == 128  # Longer for dialog
    assert config.vocab_size == 1000
    assert config.num_epochs == 50  # Fewer epochs for large dataset

    print(f"\n✅ TEST 1 PASSED")
    return True


def test_ultrachat_dataset_mock():
    """Test UltraChatDataset with mock data"""
    print("\n" + "="*80)
    print("TEST 2: UltraChatDataset with Mock Data")
    print("="*80)

    from src.training.ultrachat_dataset import UltraChatDataset
    from src.training.dataset import SimpleTokenizer

    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.build_vocab([
        "hello world conversation",
        "how are you today",
        "user assistant dialogue system"
    ])

    # Mock UltraChat data (simulating the format)
    mock_data = [
        {
            'data': [
                {'content': 'User: What is machine learning?'},
                {'content': 'Assistant: Machine learning is a subset of AI...'},
                {'content': 'User: Can you give an example?'},
                {'content': 'Assistant: Sure! Image recognition is a common example...'}
            ]
        },
        {
            'data': [
                {'content': 'User: Explain quantum computing'},
                {'content': 'Assistant: Quantum computing uses quantum bits...'},
            ]
        }
    ]

    # Create dataset
    dataset = UltraChatDataset(mock_data, tokenizer, max_length=128)

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
    """Test model creation with UltraChat config"""
    print("\n" + "="*80)
    print("TEST 3: Model Creation with UltraChat Config")
    print("="*80)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
    from train_ultrachat import UltraChatTrainConfig
    from src.models.context_vector_llm import ContextVectorLLM

    # Create config
    config = UltraChatTrainConfig(num_layers=1)
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
    print("UltraChat Training Script Test Suite")
    print("="*80)

    results = []

    # Run tests
    results.append(("UltraChatTrainConfig", test_ultrachat_config()))
    results.append(("UltraChatDataset Mock", test_ultrachat_dataset_mock()))
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
        print("\nUltraChat training script is ready to use:")
        print("  python scripts/train_ultrachat.py --num_layers 1")
        print("  python scripts/train_ultrachat.py --num_layers 4 --max_samples 100000")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
