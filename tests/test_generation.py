#!/usr/bin/env python3
"""
Test text generation functionality

Verifies:
1. Model loading from checkpoint
2. Text generation works
3. Chat interface works
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.context_vector_llm import ContextVectorLLM
from src.training.dataset import SimpleTokenizer
from src.inference.generator import TextGenerator
from src.utils.config import NewLLMConfig


def test_generator_creation():
    """Test TextGenerator creation"""
    print("\n" + "="*80)
    print("TEST 1: TextGenerator Creation")
    print("="*80)

    # Create mock model and tokenizer
    config = NewLLMConfig()
    config.device = 'cpu'  # Use CPU for testing
    model = ContextVectorLLM(config)
    model.eval()

    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.build_vocab(["hello world", "how are you", "test generation"])

    # Create generator
    generator = TextGenerator(model, tokenizer, device='cpu')

    print(f"\n✓ TextGenerator created")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Vocab size: {len(tokenizer.word2idx)}")
    print(f"   Device: {generator.device}")

    print(f"\n✅ TEST 1 PASSED")
    return True


def test_text_generation():
    """Test basic text generation"""
    print("\n" + "="*80)
    print("TEST 2: Text Generation")
    print("="*80)

    # Create model and tokenizer
    config = NewLLMConfig()
    config.device = 'cpu'
    model = ContextVectorLLM(config)
    model.eval()

    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.build_vocab([
        "hello world how are you",
        "i am doing well thank you",
        "what is your name",
        "my name is assistant"
    ])

    generator = TextGenerator(model, tokenizer, device='cpu')

    # Test generation
    print(f"\n▶️  Testing generation...")
    prompt = "hello"
    print(f"   Prompt: '{prompt}'")

    try:
        generated = generator.generate(
            prompt,
            max_length=20,
            temperature=1.0
        )
        print(f"   Generated: '{generated}'")
        print(f"   Length: {len(generated.split())} words")

        # Verify output
        assert len(generated) > len(prompt), "Generated text should be longer than prompt"

        print(f"\n✓ Text generation successful")
        print(f"\n✅ TEST 2 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Generation failed: {e}")
        return False


def test_chat_interface():
    """Test chat interface"""
    print("\n" + "="*80)
    print("TEST 3: Chat Interface")
    print("="*80)

    # Create model and tokenizer
    config = NewLLMConfig()
    config.device = 'cpu'
    model = ContextVectorLLM(config)
    model.eval()

    tokenizer = SimpleTokenizer(vocab_size=1000)
    tokenizer.build_vocab([
        "human hello how are you",
        "assistant i am doing well",
        "human what is your name",
        "assistant my name is new llm"
    ])

    generator = TextGenerator(model, tokenizer, device='cpu')

    # Test chat
    print(f"\n▶️  Testing chat interface...")

    try:
        # First turn
        user_input1 = "hello"
        print(f"   User: '{user_input1}'")

        response1, context1 = generator.chat(user_input1, context="")
        print(f"   Assistant: '{response1}'")
        print(f"   Context length: {len(context1)} chars")

        # Second turn
        user_input2 = "how are you"
        print(f"\n   User: '{user_input2}'")

        response2, context2 = generator.chat(user_input2, context=context1)
        print(f"   Assistant: '{response2}'")
        print(f"   Context length: {len(context2)} chars")

        # Verify
        assert len(response1) > 0, "Response should not be empty"
        assert len(response2) > 0, "Response should not be empty"
        assert len(context2) > len(context1), "Context should grow with conversation"

        print(f"\n✓ Chat interface successful")
        print(f"\n✅ TEST 3 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Chat failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_loading():
    """Test loading from actual checkpoint (if available)"""
    print("\n" + "="*80)
    print("TEST 4: Checkpoint Loading (Optional)")
    print("="*80)

    # Check for Dolly checkpoint
    checkpoint_path = "checkpoints/best_new_llm_dolly_layers1.pt"

    if not os.path.exists(checkpoint_path):
        print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
        print(f"   This test is optional - skipping")
        print(f"\n⏭️  TEST 4 SKIPPED")
        return True

    print(f"\n▶️  Loading checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        config = checkpoint['config']
        tokenizer = checkpoint['tokenizer']

        print(f"✓ Checkpoint loaded")
        print(f"   Layers: {config.num_layers}")
        print(f"   Vocab size: {config.vocab_size}")
        print(f"   Tokenizer vocab: {len(tokenizer.word2idx)} words")

        # Create model
        model = ContextVectorLLM(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"✓ Model loaded")

        # Test generation
        generator = TextGenerator(model, tokenizer, device='cpu')

        prompt = "Human: Hello"
        print(f"\n▶️  Testing generation with trained model...")
        print(f"   Prompt: '{prompt}'")

        generated = generator.generate(prompt, max_length=30, temperature=0.8)
        print(f"   Generated: '{generated}'")

        print(f"\n✅ TEST 4 PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Checkpoint loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("Text Generation Test Suite")
    print("="*80)

    results = []

    # Run tests
    results.append(("TextGenerator Creation", test_generator_creation()))
    results.append(("Text Generation", test_text_generation()))
    results.append(("Chat Interface", test_chat_interface()))
    results.append(("Checkpoint Loading", test_checkpoint_loading()))

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
        print("\nText generation is ready to use:")
        print("  python scripts/chat.py --checkpoint checkpoints/best_new_llm_ultrachat_layers1.pt")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
