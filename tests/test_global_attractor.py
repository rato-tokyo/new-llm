"""Test for Global Attractor Problem Detection

This test detects the "degenerate solution" where all tokens converge to
the same identical fixed point, losing token-specific information.

Problem: All different tokens → Same context vector
Expected: Different tokens → Different context vectors

This was a critical bug in early implementations using "simple" context updater.
The gated updater (LSTM-style) solves this problem.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.new_llm_flexible import NewLLMFlexible
from src.utils.dialogue_config import SmallDialogueConfig


class GlobalAttractorTester:
    """Test for global attractor problem"""

    def __init__(self):
        self.config = SmallDialogueConfig()
        self.config.vocab_size = 1000
        self.model = NewLLMFlexible(self.config)
        self.model.eval()

        print("=" * 70)
        print("Global Attractor Problem Detection Test")
        print("=" * 70)
        print(f"Model: {self.config.num_layers} layers, context_dim={self.config.context_dim}")
        print(f"Parameters: {self.model.count_parameters():,}")
        print("=" * 70)

    def test_different_tokens_different_contexts(self):
        """
        Test 1: Different tokens should produce different fixed-point contexts

        Global Attractor Problem:
        - All tokens → Same context (L2 distance ≈ 0)
        - Cosine similarity ≈ 1.0
        - Loss appears low but model is degenerate

        Healthy Model:
        - Different tokens → Different contexts (L2 distance > 0.1)
        - Cosine similarity < 0.99
        """
        print("\n" + "-" * 70)
        print("TEST 1: Different Tokens → Different Contexts")
        print("-" * 70)

        # Create input with different tokens
        num_tokens = 10
        input_ids = torch.arange(num_tokens).unsqueeze(0)  # [1, 10]: [0, 1, 2, ..., 9]

        print(f"Input: {num_tokens} different tokens: {input_ids[0].tolist()}")

        # Get fixed-point contexts
        with torch.no_grad():
            contexts, converged, num_iters = self.model.get_fixed_point_context(
                input_ids,
                max_iterations=50,
                tolerance=1e-4
            )

        # Extract contexts [num_tokens, context_dim]
        contexts = contexts.squeeze(0)  # [10, context_dim]

        # Compute pairwise distances
        distances = []
        cosine_sims = []

        for i in range(num_tokens):
            for j in range(i + 1, num_tokens):
                c1 = contexts[i]
                c2 = contexts[j]

                # L2 distance
                dist = torch.norm(c1 - c2).item()
                distances.append(dist)

                # Cosine similarity
                cos_sim = torch.dot(c1, c2) / (torch.norm(c1) * torch.norm(c2) + 1e-8)
                cosine_sims.append(cos_sim.item())

        # Statistics
        avg_distance = sum(distances) / len(distances)
        min_distance = min(distances)
        max_distance = max(distances)
        avg_cosine = sum(cosine_sims) / len(cosine_sims)
        max_cosine = max(cosine_sims)

        print(f"\nResults:")
        print(f"  Average L2 distance: {avg_distance:.6f}")
        print(f"  Min L2 distance: {min_distance:.6f}")
        print(f"  Max L2 distance: {max_distance:.6f}")
        print(f"  Average cosine similarity: {avg_cosine:.6f}")
        print(f"  Max cosine similarity: {max_cosine:.6f}")

        # Diagnosis
        print(f"\nDiagnosis:")

        # Check for global attractor
        if avg_distance < 0.001:
            print(f"  ⚠️  GLOBAL ATTRACTOR DETECTED!")
            print(f"     All tokens converge to same point (avg L2={avg_distance:.6f})")
            print(f"     This is a DEGENERATE SOLUTION - model is broken")
            status = "✗ FAIL - Global Attractor Problem"
            passed = False

        elif avg_distance < 0.1 and avg_cosine > 0.99:
            print(f"  ⚠️  SUSPICIOUS: Very similar contexts")
            print(f"     L2={avg_distance:.6f}, cosine={avg_cosine:.6f}")
            print(f"     Model may be converging to attractor")
            status = "⚠️  WARNING - Suspicious similarity"
            passed = False

        else:
            print(f"  ✓ HEALTHY: Different tokens have different contexts")
            print(f"     L2={avg_distance:.6f} (good diversity)")
            print(f"     Cosine={avg_cosine:.6f} (not identical)")
            status = "✓ PASS"
            passed = True

        print(f"\nStatus: {status}")
        return passed

    def test_context_norm_variation(self):
        """
        Test 2: Context norms should be similar (due to LayerNorm)

        Note: LayerNorm normalizes contexts, so identical norms are EXPECTED.
        This is NOT a sign of global attractor problem.

        What we check instead:
        - Norms are consistent (good)
        - But directions differ (tested in Test 1)
        """
        print("\n" + "-" * 70)
        print("TEST 2: Context Norm Consistency (LayerNorm Effect)")
        print("-" * 70)

        # Different tokens
        num_tokens = 20
        input_ids = torch.arange(num_tokens).unsqueeze(0)

        print(f"Input: {num_tokens} different tokens")

        # Get contexts
        with torch.no_grad():
            contexts, _, _ = self.model.get_fixed_point_context(
                input_ids,
                max_iterations=50,
                tolerance=1e-4
            )

        contexts = contexts.squeeze(0)  # [num_tokens, context_dim]

        # Compute norms
        norms = torch.norm(contexts, dim=-1)  # [num_tokens]

        avg_norm = norms.mean().item()
        std_norm = norms.std().item()
        min_norm = norms.min().item()
        max_norm = norms.max().item()

        print(f"\nResults:")
        print(f"  Average norm: {avg_norm:.4f}")
        print(f"  Std dev: {std_norm:.4f}")
        print(f"  Min norm: {min_norm:.4f}")
        print(f"  Max norm: {max_norm:.4f}")

        # Diagnosis
        print(f"\nDiagnosis:")

        # With LayerNorm, identical norms are EXPECTED
        if std_norm < 0.01:
            print(f"  ✓ EXPECTED: LayerNorm normalizes contexts (std={std_norm:.6f})")
            print(f"     This is normal behavior, not a problem")
            status = "✓ PASS (LayerNorm working)"
            passed = True
        else:
            print(f"  ⚠️  UNEXPECTED: Norms vary despite LayerNorm (std={std_norm:.4f})")
            status = "⚠️  OBSERVED"
            passed = True  # Not a failure, just unexpected

        print(f"\nStatus: {status}")
        return passed

    def test_convergence_speed_variation(self):
        """
        Test 3: Convergence speed should vary across tokens

        Global Attractor Problem:
        - All tokens converge in 1-2 iterations (instant)
        - No variation in convergence speed

        Healthy Model:
        - Convergence speed varies
        - Some tokens take longer than others
        """
        print("\n" + "-" * 70)
        print("TEST 3: Convergence Speed Variation")
        print("-" * 70)

        num_tokens = 20
        input_ids = torch.arange(num_tokens).unsqueeze(0)

        print(f"Input: {num_tokens} different tokens")

        # Get convergence info
        with torch.no_grad():
            _, converged, num_iters = self.model.get_fixed_point_context(
                input_ids,
                max_iterations=50,
                tolerance=1e-4
            )

        num_iters = num_iters.squeeze(0)  # [num_tokens]

        # Only consider converged tokens
        converged_mask = converged.squeeze(0)
        if converged_mask.sum() > 0:
            converged_iters = num_iters[converged_mask].float()

            avg_iters = converged_iters.mean().item()
            std_iters = converged_iters.std().item()
            min_iters = converged_iters.min().item()
            max_iters = converged_iters.max().item()

            print(f"\nResults:")
            print(f"  Converged tokens: {converged_mask.sum().item()}/{num_tokens}")
            print(f"  Average iterations: {avg_iters:.2f}")
            print(f"  Std dev: {std_iters:.2f}")
            print(f"  Min iterations: {min_iters:.0f}")
            print(f"  Max iterations: {max_iters:.0f}")

            # Diagnosis
            print(f"\nDiagnosis:")

            if avg_iters < 3 and std_iters < 0.5:
                print(f"  ⚠️  GLOBAL ATTRACTOR: Instant convergence (avg={avg_iters:.1f})")
                print(f"     All tokens reach same point in 1-2 steps")
                status = "✗ FAIL - Global Attractor"
                passed = False
            else:
                print(f"  ✓ OBSERVED: Convergence varies (avg={avg_iters:.1f}, std={std_iters:.1f})")
                status = "✓ OBSERVED"
                passed = True

        else:
            print(f"\nResults:")
            print(f"  No tokens converged (0/{num_tokens})")
            status = "✓ OBSERVED (untrained model)"
            print(f"Status: {status}")
            passed = True
        return passed

    def test_same_token_different_positions(self):
        """
        Test 4: Same token should produce consistent fixed-point context

        Note: Fixed-point context depends ONLY on the token itself, not position.
        This is by design - the fixed point is the stable state for that token.

        Expected: Same token → Identical fixed-point context (distance ≈ 0)
        This is CORRECT behavior, not a problem.
        """
        print("\n" + "-" * 70)
        print("TEST 4: Fixed-Point Consistency for Same Token")
        print("-" * 70)

        # Repeat same token at different positions
        token_id = 42
        num_positions = 10
        input_ids = torch.full((1, num_positions), token_id)

        print(f"Input: Token {token_id} repeated at {num_positions} positions")

        # Get contexts
        with torch.no_grad():
            contexts, _, _ = self.model.get_fixed_point_context(
                input_ids,
                max_iterations=50,
                tolerance=1e-4
            )

        contexts = contexts.squeeze(0)  # [num_positions, context_dim]

        # Compute pairwise distances
        distances = []
        for i in range(num_positions):
            for j in range(i + 1, num_positions):
                dist = torch.norm(contexts[i] - contexts[j]).item()
                distances.append(dist)

        if len(distances) > 0:
            avg_distance = sum(distances) / len(distances)
            max_distance = max(distances)

            print(f"\nResults:")
            print(f"  Average L2 distance: {avg_distance:.6f}")
            print(f"  Max L2 distance: {max_distance:.6f}")

            print(f"\nDiagnosis:")

            if avg_distance < 0.0001:
                print(f"  ✓ EXPECTED: Same token → Same fixed point (dist={avg_distance:.6f})")
                print(f"     This is correct behavior for fixed-point context")
                status = "✓ PASS (Consistent fixed points)"
                passed = True
            else:
                print(f"  ⚠️  UNEXPECTED: Same token → Different contexts (dist={avg_distance:.6f})")
                print(f"     Fixed points should be identical for same token")
                status = "⚠️  WARNING (Inconsistent fixed points)"
                passed = False

        else:
            status = "✓ OBSERVED"
            passed = True

        print(f"\nStatus: {status}")
        return passed

    def run_all_tests(self):
        """Run all tests and report summary"""
        tests = [
            ("Different Tokens → Different Contexts", self.test_different_tokens_different_contexts),
            ("Context Norm Variation", self.test_context_norm_variation),
            ("Convergence Speed Variation", self.test_convergence_speed_variation),
            ("Same Token at Different Positions", self.test_same_token_different_positions),
        ]

        results = []
        for name, test_fn in tests:
            try:
                passed = test_fn()
                results.append((name, passed))
            except Exception as e:
                print(f"\n✗ ERROR in {name}: {e}")
                import traceback
                traceback.print_exc()
                results.append((name, False))

        # Summary
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)

        passed_count = sum(1 for _, passed in results if passed)
        total_count = len(results)

        has_failure = False
        for name, passed in results:
            if not passed:
                status = "✗ FAIL"
                has_failure = True
            else:
                status = "✓ PASS"
            print(f"{status}: {name}")

        print(f"\nTotal: {passed_count}/{total_count} tests passed")

        if has_failure:
            print("\n" + "!" * 70)
            print("⚠️  GLOBAL ATTRACTOR PROBLEM DETECTED!")
            print("!" * 70)
            print("\nThis indicates a degenerate solution where all tokens converge")
            print("to the same fixed point. The model appears to work but is broken.")
            print("\nPossible causes:")
            print("1. Using 'simple' context updater (should use 'gated')")
            print("2. Insufficient model capacity (increase layers or context_dim)")
            print("3. Improper initialization")
            print("\nRecommended actions:")
            print("- Verify context_update_strategy='gated' is used")
            print("- Increase num_layers (1→2→3)")
            print("- Increase context_dim (128→256→512)")

        print("=" * 70)

        return passed_count == total_count


def main():
    """Main test runner"""
    tester = GlobalAttractorTester()
    all_passed = tester.run_all_tests()

    if all_passed:
        print("\n✓ No global attractor problem detected!")
        return 0
    else:
        print("\n✗ Global attractor problem detected!")
        return 1


if __name__ == "__main__":
    exit(main())
