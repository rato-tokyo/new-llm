"""Test convergence on simple sentences (2-3 words)

This test verifies that the model can find fixed points for very simple inputs.
If this test fails, there is likely a bug in the implementation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.models.new_llm_flexible import NewLLMFlexible
from src.utils.dialogue_config import TinyDialogueConfig
from transformers import AutoTokenizer


def test_simple_sentence_convergence():
    """Test that simple 2-3 word sentences converge quickly"""

    print("=" * 60)
    print("Test: Simple Sentence Convergence")
    print("=" * 60)

    # Initialize model
    config = TinyDialogueConfig()
    config.device = "cpu"
    model = NewLLMFlexible(config).to(config.device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/DialoGPT-small",
        cache_dir="cache/tokenizer"
    )

    # Test cases: very simple sentences
    test_sentences = [
        "Hello world",
        "Good morning",
        "Thank you",
        "Yes please",
        "I agree",
        "No problem",
    ]

    print(f"\nTesting {len(test_sentences)} simple sentences...")
    print(f"Expected: Most tokens should converge within 50 iterations\n")

    all_passed = True
    total_converged = 0
    total_tokens = 0

    for i, sentence in enumerate(test_sentences):
        # Tokenize
        tokens = tokenizer.encode(sentence)
        input_ids = torch.tensor([tokens], dtype=torch.long, device=config.device)

        # Get fixed points
        with torch.no_grad():
            fixed_contexts, converged, num_iters = model.get_fixed_point_context(
                input_ids,
                max_iterations=200,
                tolerance=1e-2,  # Relaxed threshold
                warmup_iterations=100  # Long warmup for stability
            )

        # Statistics
        num_tokens = len(tokens)
        num_converged = converged.sum().item()
        convergence_rate = num_converged / num_tokens
        avg_iters = num_iters.float().mean().item()

        total_converged += num_converged
        total_tokens += num_tokens

        # Check result
        if convergence_rate >= 0.5:  # At least 50% should converge
            status = "✓ PASS"
            passed = True
        else:
            status = "✗ FAIL"
            passed = False
            all_passed = False

        print(f"{status} | \"{sentence}\"")
        print(f"      Tokens: {num_tokens} | Converged: {num_converged} ({convergence_rate:.1%}) | Avg iters: {avg_iters:.1f}")

    # Overall results
    overall_rate = total_converged / total_tokens
    print("\n" + "=" * 60)
    print("Overall Results:")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Converged: {total_converged} ({overall_rate:.1%})")
    print("=" * 60)

    if overall_rate >= 0.3:  # At least 30% overall
        print("\n✓ TEST PASSED")
        print("  Simple sentences show some convergence")
    else:
        print("\n✗ TEST FAILED")
        print("  ⚠️ WARNING: Implementation may have bugs!")
        print("  Even simple 2-3 word sentences fail to converge")
        all_passed = False

    return all_passed


def test_repeated_words():
    """Test that repeated words converge to same fixed point"""

    print("\n" + "=" * 60)
    print("Test: Repeated Word Consistency")
    print("=" * 60)

    # Initialize model
    config = TinyDialogueConfig()
    config.device = "cpu"
    model = NewLLMFlexible(config).to(config.device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/DialoGPT-small",
        cache_dir="cache/tokenizer"
    )

    # Test: "hello hello hello"
    sentence = "hello hello hello"
    tokens = tokenizer.encode(sentence)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=config.device)

    print(f"\nTesting: \"{sentence}\"")
    print(f"Tokens: {tokens}")

    # Get fixed points
    with torch.no_grad():
        fixed_contexts, converged, num_iters = model.get_fixed_point_context(
            input_ids,
            max_iterations=200,
            tolerance=1e-2,  # Relaxed threshold
            warmup_iterations=100  # Long warmup for stability
        )

    # Check if "hello" tokens have similar fixed points
    # Note: Tokens may be different due to tokenization (e.g., " hello" vs "hello")
    print(f"\nConvergence status:")
    for i, (token_id, conv, iters) in enumerate(zip(tokens, converged[0], num_iters[0])):
        conv_str = "✓" if conv else "✗"
        print(f"  Token {i} (id={token_id}): {conv_str} converged in {iters} iters")

    convergence_rate = converged.float().mean().item()
    print(f"\nOverall convergence: {convergence_rate:.1%}")

    if convergence_rate >= 0.5:
        print("✓ TEST PASSED")
        return True
    else:
        print("✗ TEST FAILED")
        print("  Repeated words should converge easily")
        return False


def main():
    """Run all simple convergence tests"""

    # Check if tokenizer exists
    if not os.path.exists("cache/tokenizer"):
        print("❌ ERROR: Tokenizer not found at cache/tokenizer")
        print("   Please run train_dialogue.py first to create tokenizer")
        return

    print("\n" + "=" * 60)
    print("SIMPLE CONVERGENCE TESTS")
    print("Purpose: Verify implementation with easy inputs")
    print("=" * 60)

    # Run tests
    test1_passed = test_simple_sentence_convergence()
    test2_passed = test_repeated_words()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Simple sentences: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"Repeated words:   {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print("=" * 60)

    if test1_passed and test2_passed:
        print("\n✓ ALL TESTS PASSED")
        print("  Implementation appears correct")
    else:
        print("\n✗ SOME TESTS FAILED")
        print("  ⚠️ Possible implementation bugs detected")
        print("  Please review:")
        print("  - Fixed-point iteration logic")
        print("  - Convergence threshold")
        print("  - Context update mechanism")


if __name__ == "__main__":
    main()
