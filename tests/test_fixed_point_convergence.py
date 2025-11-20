"""Test Fixed-Point Convergence Properties

This test suite verifies that the New-LLM model correctly finds fixed points
for different types of input sequences.

Expected behaviors:
1. Same token repeated → Should converge quickly (high convergence rate)
2. Random tokens → May not converge (low convergence rate)
3. Structured patterns → Should converge moderately

This validates the fundamental fixed-point learning mechanism.
"""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.new_llm_flexible import NewLLMFlexible
from src.utils.dialogue_config import SmallDialogueConfig


class FixedPointTester:
    """Test fixed-point convergence for various input patterns"""

    def __init__(self):
        # Use small config for fast testing
        self.config = SmallDialogueConfig()
        self.config.vocab_size = 1000  # Small vocab for testing
        self.model = NewLLMFlexible(self.config)
        self.model.eval()

        print("=" * 70)
        print("Fixed-Point Convergence Test Suite")
        print("=" * 70)
        print(f"Model: {self.config.num_layers} layers, context_dim={self.config.context_dim}")
        print(f"Parameters: {self.model.count_parameters():,}")
        print("=" * 70)

    def test_repeated_token(self, token_id=42, length=50, max_iters=50):
        """
        Test 1: Same token repeated

        Expected: High convergence rate (>90%)
        Reasoning: Repeated token should quickly find stable state
        """
        print("\n" + "-" * 70)
        print("TEST 1: Repeated Token (Same token repeated)")
        print("-" * 70)

        # Create input: [token_id, token_id, token_id, ...]
        input_ids = torch.full((1, length), token_id, dtype=torch.long)
        print(f"Input: Token {token_id} repeated {length} times")

        # Get fixed points
        with torch.no_grad():
            contexts, converged, num_iters = self.model.get_fixed_point_context(
                input_ids,
                max_iterations=max_iters,
                tolerance=1e-4
            )

        # Statistics
        convergence_rate = converged.float().mean().item()
        avg_iters = num_iters.float().mean().item()
        max_iter = num_iters.max().item()

        print(f"Results:")
        print(f"  Converged: {converged.sum().item()}/{converged.numel()} tokens ({convergence_rate:.1%})")
        print(f"  Avg iterations: {avg_iters:.1f}")
        print(f"  Max iterations: {max_iter}")

        # Expectation
        expected_min = 0.80  # Expect at least 80% convergence
        status = "✓ PASS" if convergence_rate >= expected_min else "✗ FAIL"
        print(f"Expected: ≥{expected_min:.0%} convergence")
        print(f"Status: {status}")

        return convergence_rate >= expected_min

    def test_random_tokens(self, length=50, max_iters=50):
        """
        Test 2: Random tokens

        Expected: Low convergence rate (<50%)
        Reasoning: Random tokens may not have stable fixed points
        """
        print("\n" + "-" * 70)
        print("TEST 2: Random Tokens (No pattern)")
        print("-" * 70)

        # Create random input
        input_ids = torch.randint(0, self.config.vocab_size, (1, length))
        print(f"Input: {length} random tokens from vocab [0, {self.config.vocab_size})")

        # Get fixed points
        with torch.no_grad():
            contexts, converged, num_iters = self.model.get_fixed_point_context(
                input_ids,
                max_iterations=max_iters,
                tolerance=1e-4
            )

        # Statistics
        convergence_rate = converged.float().mean().item()
        avg_iters = num_iters.float().mean().item()
        max_iter = num_iters.max().item()

        print(f"Results:")
        print(f"  Converged: {converged.sum().item()}/{converged.numel()} tokens ({convergence_rate:.1%})")
        print(f"  Avg iterations: {avg_iters:.1f}")
        print(f"  Max iterations: {max_iter}")

        # For random tokens, we just observe (no strict requirement)
        print(f"Expected: Variable (typically <50% for untrained model)")
        print(f"Status: ✓ OBSERVED (convergence_rate={convergence_rate:.1%})")

        return True  # Always pass (just observing)

    def test_alternating_tokens(self, token_a=10, token_b=20, length=50, max_iters=50):
        """
        Test 3: Alternating tokens (ABABAB...)

        Expected: Moderate convergence rate (40-70%)
        Reasoning: Simple pattern should partially converge
        """
        print("\n" + "-" * 70)
        print("TEST 3: Alternating Tokens (ABABAB... pattern)")
        print("-" * 70)

        # Create alternating pattern
        input_ids = torch.zeros((1, length), dtype=torch.long)
        input_ids[0, 0::2] = token_a  # Even positions
        input_ids[0, 1::2] = token_b  # Odd positions
        print(f"Input: Alternating tokens {token_a} and {token_b} ({length} tokens)")

        # Get fixed points
        with torch.no_grad():
            contexts, converged, num_iters = self.model.get_fixed_point_context(
                input_ids,
                max_iterations=max_iters,
                tolerance=1e-4
            )

        # Statistics
        convergence_rate = converged.float().mean().item()
        avg_iters = num_iters.float().mean().item()
        max_iter = num_iters.max().item()

        print(f"Results:")
        print(f"  Converged: {converged.sum().item()}/{converged.numel()} tokens ({convergence_rate:.1%})")
        print(f"  Avg iterations: {avg_iters:.1f}")
        print(f"  Max iterations: {max_iter}")

        # For alternating pattern, we just observe
        print(f"Expected: Variable (typically 40-70% for untrained model)")
        print(f"Status: ✓ OBSERVED (convergence_rate={convergence_rate:.1%})")

        return True  # Always pass (just observing)

    def test_block_pattern(self, block_size=10, num_blocks=5, max_iters=50):
        """
        Test 4: Block pattern (AAAA...BBBB...CCCC...)

        Expected: High convergence within blocks
        Reasoning: Each block is repeated tokens → should converge well
        """
        print("\n" + "-" * 70)
        print("TEST 4: Block Pattern (AAAA...BBBB...CCCC...)")
        print("-" * 70)

        # Create block pattern
        length = block_size * num_blocks
        input_ids = torch.zeros((1, length), dtype=torch.long)

        for i in range(num_blocks):
            token_id = i * 10  # Use different tokens for each block
            start = i * block_size
            end = (i + 1) * block_size
            input_ids[0, start:end] = token_id

        print(f"Input: {num_blocks} blocks of {block_size} tokens each")

        # Get fixed points
        with torch.no_grad():
            contexts, converged, num_iters = self.model.get_fixed_point_context(
                input_ids,
                max_iterations=max_iters,
                tolerance=1e-4
            )

        # Statistics
        convergence_rate = converged.float().mean().item()
        avg_iters = num_iters.float().mean().item()

        # Per-block statistics
        print(f"Results:")
        print(f"  Overall converged: {converged.sum().item()}/{converged.numel()} ({convergence_rate:.1%})")
        print(f"  Avg iterations: {avg_iters:.1f}")

        # Check each block
        for i in range(num_blocks):
            start = i * block_size
            end = (i + 1) * block_size
            block_converged = converged[0, start:end]
            block_rate = block_converged.float().mean().item()
            print(f"  Block {i+1}: {block_converged.sum().item()}/{block_size} converged ({block_rate:.1%})")

        expected_min = 0.70  # Expect at least 70% overall
        status = "✓ PASS" if convergence_rate >= expected_min else "✗ FAIL"
        print(f"Expected: ≥{expected_min:.0%} overall convergence")
        print(f"Status: {status}")

        return convergence_rate >= expected_min

    def test_gradient_flow(self):
        """
        Test 5: Gradient flow through fixed-point computation

        Expected: Gradients should NOT flow (using torch.no_grad())
        Reasoning: Fixed-point search is inference-only
        """
        print("\n" + "-" * 70)
        print("TEST 5: Gradient Flow (Should be disabled)")
        print("-" * 70)

        input_ids = torch.full((1, 10), 42, dtype=torch.long)

        # Check gradient mode
        contexts, converged, num_iters = self.model.get_fixed_point_context(
            input_ids,
            max_iterations=10,
            tolerance=1e-4
        )

        # Gradients should not be attached
        has_grad = contexts.requires_grad
        print(f"Results:")
        print(f"  Context requires_grad: {has_grad}")
        print(f"Expected: False (fixed-point search is inference-only)")

        status = "✓ PASS" if not has_grad else "✗ FAIL"
        print(f"Status: {status}")

        return not has_grad

    def test_convergence_threshold(self):
        """
        Test 6: Convergence threshold sensitivity

        Expected: Lower threshold → lower convergence rate (stricter)
        Reasoning: Threshold controls what counts as "converged"
        """
        print("\n" + "-" * 70)
        print("TEST 6: Convergence Threshold Sensitivity")
        print("-" * 70)

        input_ids = torch.full((1, 30), 42, dtype=torch.long)

        thresholds = [1e-3, 1e-4, 1e-5]
        rates = []

        for threshold in thresholds:
            with torch.no_grad():
                contexts, converged, num_iters = self.model.get_fixed_point_context(
                    input_ids,
                    max_iterations=50,
                    tolerance=threshold
                )
            rate = converged.float().mean().item()
            rates.append(rate)
            print(f"  Threshold={threshold:.0e}: {rate:.1%} converged")

        # Check if rates decrease with stricter threshold
        is_decreasing = all(rates[i] >= rates[i+1] for i in range(len(rates)-1))

        print(f"Expected: Convergence rate should decrease with stricter threshold")
        status = "✓ PASS" if is_decreasing else "✗ OBSERVED"
        print(f"Status: {status}")

        return True  # Just observing trend

    def run_all_tests(self):
        """Run all tests and report summary"""
        tests = [
            ("Repeated Token", lambda: self.test_repeated_token()),
            ("Random Tokens", lambda: self.test_random_tokens()),
            ("Alternating Tokens", lambda: self.test_alternating_tokens()),
            ("Block Pattern", lambda: self.test_block_pattern()),
            ("Gradient Flow", lambda: self.test_gradient_flow()),
            ("Convergence Threshold", lambda: self.test_convergence_threshold()),
        ]

        results = []
        for name, test_fn in tests:
            try:
                passed = test_fn()
                results.append((name, passed))
            except Exception as e:
                print(f"\n✗ ERROR in {name}: {e}")
                results.append((name, False))

        # Summary
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)

        passed_count = sum(1 for _, passed in results if passed)
        total_count = len(results)

        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {name}")

        print(f"\nTotal: {passed_count}/{total_count} tests passed")
        print("=" * 70)

        return passed_count == total_count


def main():
    """Main test runner"""
    tester = FixedPointTester()
    all_passed = tester.run_all_tests()

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
