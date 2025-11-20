"""Test Basic Model Functionality

This test suite verifies that the New-LLM model components work correctly
regardless of training status.

These are sanity checks for:
- Model creation
- Forward pass
- Output shapes
- Numerical stability
"""

import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.new_llm_flexible import NewLLMFlexible
from src.utils.dialogue_config import SmallDialogueConfig


class BasicFunctionalityTester:
    """Test basic model functionality"""

    def __init__(self):
        self.config = SmallDialogueConfig()
        self.config.vocab_size = 1000
        self.model = NewLLMFlexible(self.config)

        print("=" * 70)
        print("Basic Functionality Test Suite")
        print("=" * 70)
        print(f"Model: {self.config.num_layers} layers, context_dim={self.config.context_dim}")
        print(f"Parameters: {self.model.count_parameters():,}")
        print("=" * 70)

    def test_model_creation(self):
        """Test 1: Model can be created"""
        print("\n" + "-" * 70)
        print("TEST 1: Model Creation")
        print("-" * 70)

        try:
            model = NewLLMFlexible(self.config)
            param_count = model.count_parameters()

            print(f"Results:")
            print(f"  Model created successfully")
            print(f"  Parameters: {param_count:,}")
            print(f"Status: ✓ PASS")
            return True

        except Exception as e:
            print(f"Status: ✗ FAIL - {e}")
            return False

    def test_forward_pass(self):
        """Test 2: Forward pass produces correct shapes"""
        print("\n" + "-" * 70)
        print("TEST 2: Forward Pass (Shape Verification)")
        print("-" * 70)

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        try:
            logits = self.model(input_ids)

            expected_shape = (batch_size, seq_len, self.config.vocab_size)
            actual_shape = tuple(logits.shape)

            print(f"Results:")
            print(f"  Input shape: {tuple(input_ids.shape)}")
            print(f"  Output shape: {actual_shape}")
            print(f"  Expected shape: {expected_shape}")

            if actual_shape == expected_shape:
                print(f"Status: ✓ PASS")
                return True
            else:
                print(f"Status: ✗ FAIL - Shape mismatch")
                return False

        except Exception as e:
            print(f"Status: ✗ FAIL - {e}")
            return False

    def test_context_trajectory(self):
        """Test 3: Context trajectory is returned when requested"""
        print("\n" + "-" * 70)
        print("TEST 3: Context Trajectory")
        print("-" * 70)

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        try:
            logits, contexts = self.model(input_ids, return_context_trajectory=True)

            expected_context_shape = (batch_size, seq_len, self.config.context_dim)
            actual_context_shape = tuple(contexts.shape)

            print(f"Results:")
            print(f"  Logits shape: {tuple(logits.shape)}")
            print(f"  Context shape: {actual_context_shape}")
            print(f"  Expected context shape: {expected_context_shape}")

            if actual_context_shape == expected_context_shape:
                print(f"Status: ✓ PASS")
                return True
            else:
                print(f"Status: ✗ FAIL - Shape mismatch")
                return False

        except Exception as e:
            print(f"Status: ✗ FAIL - {e}")
            return False

    def test_numerical_stability(self):
        """Test 4: No NaN or Inf in outputs"""
        print("\n" + "-" * 70)
        print("TEST 4: Numerical Stability (No NaN/Inf)")
        print("-" * 70)

        input_ids = torch.randint(0, self.config.vocab_size, (2, 20))

        try:
            logits = self.model(input_ids)

            has_nan = torch.isnan(logits).any().item()
            has_inf = torch.isinf(logits).any().item()

            print(f"Results:")
            print(f"  Contains NaN: {has_nan}")
            print(f"  Contains Inf: {has_inf}")

            if not has_nan and not has_inf:
                print(f"Status: ✓ PASS")
                return True
            else:
                print(f"Status: ✗ FAIL - Numerical instability detected")
                return False

        except Exception as e:
            print(f"Status: ✗ FAIL - {e}")
            return False

    def test_determinism(self):
        """Test 5: Same input produces same output"""
        print("\n" + "-" * 70)
        print("TEST 5: Determinism (Same input → Same output)")
        print("-" * 70)

        input_ids = torch.randint(0, self.config.vocab_size, (1, 15))

        try:
            self.model.eval()
            with torch.no_grad():
                logits1 = self.model(input_ids)
                logits2 = self.model(input_ids)

            max_diff = (logits1 - logits2).abs().max().item()

            print(f"Results:")
            print(f"  Max difference: {max_diff:.2e}")

            if max_diff < 1e-6:
                print(f"Status: ✓ PASS")
                return True
            else:
                print(f"Status: ✗ FAIL - Non-deterministic output")
                return False

        except Exception as e:
            print(f"Status: ✗ FAIL - {e}")
            return False

    def test_batch_independence(self):
        """Test 6: Batch elements are processed independently"""
        print("\n" + "-" * 70)
        print("TEST 6: Batch Independence")
        print("-" * 70)

        seq_len = 10
        input1 = torch.randint(0, self.config.vocab_size, (1, seq_len))
        input2 = torch.randint(0, self.config.vocab_size, (1, seq_len))
        input_batch = torch.cat([input1, input2], dim=0)  # [2, seq_len]

        try:
            self.model.eval()
            with torch.no_grad():
                # Process separately
                logits1 = self.model(input1)
                logits2 = self.model(input2)

                # Process as batch
                logits_batch = self.model(input_batch)

            # Compare
            diff1 = (logits_batch[0] - logits1[0]).abs().max().item()
            diff2 = (logits_batch[1] - logits2[0]).abs().max().item()

            print(f"Results:")
            print(f"  Max diff (sample 1): {diff1:.2e}")
            print(f"  Max diff (sample 2): {diff2:.2e}")

            if diff1 < 1e-5 and diff2 < 1e-5:
                print(f"Status: ✓ PASS")
                return True
            else:
                print(f"Status: ✗ FAIL - Batch processing inconsistent")
                return False

        except Exception as e:
            print(f"Status: ✗ FAIL - {e}")
            return False

    def test_context_initialization(self):
        """Test 7: Context starts from zero"""
        print("\n" + "-" * 70)
        print("TEST 7: Context Initialization (Starts from zero)")
        print("-" * 70)

        input_ids = torch.randint(0, self.config.vocab_size, (1, 5))

        try:
            self.model.eval()
            with torch.no_grad():
                _, contexts = self.model(input_ids, return_context_trajectory=True)

            # First context should be result of processing first token with zero context
            # We can't check if it's exactly zero (it gets updated),
            # but we can check the shape and that it's not all zeros
            first_context = contexts[0, 0, :]
            is_all_zeros = (first_context == 0).all().item()

            print(f"Results:")
            print(f"  First context shape: {tuple(first_context.shape)}")
            print(f"  First context all zeros: {is_all_zeros}")
            print(f"  First context norm: {first_context.norm().item():.4f}")

            # After processing first token, context should not be all zeros
            if not is_all_zeros:
                print(f"Status: ✓ PASS (Context updated after first token)")
                return True
            else:
                print(f"Status: ✗ FAIL - Context not updated")
                return False

        except Exception as e:
            print(f"Status: ✗ FAIL - {e}")
            return False

    def test_fixed_point_shapes(self):
        """Test 8: Fixed-point computation returns correct shapes"""
        print("\n" + "-" * 70)
        print("TEST 8: Fixed-Point Computation (Shape Verification)")
        print("-" * 70)

        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))

        try:
            with torch.no_grad():
                contexts, converged, num_iters = self.model.get_fixed_point_context(
                    input_ids,
                    max_iterations=5
                )

            expected_context_shape = (batch_size, seq_len, self.config.context_dim)
            expected_flag_shape = (batch_size, seq_len)

            print(f"Results:")
            print(f"  Contexts shape: {tuple(contexts.shape)}")
            print(f"  Converged shape: {tuple(converged.shape)}")
            print(f"  Num iters shape: {tuple(num_iters.shape)}")

            shapes_ok = (
                tuple(contexts.shape) == expected_context_shape and
                tuple(converged.shape) == expected_flag_shape and
                tuple(num_iters.shape) == expected_flag_shape
            )

            if shapes_ok:
                print(f"Status: ✓ PASS")
                return True
            else:
                print(f"Status: ✗ FAIL - Shape mismatch")
                return False

        except Exception as e:
            print(f"Status: ✗ FAIL - {e}")
            return False

    def run_all_tests(self):
        """Run all tests and report summary"""
        tests = [
            ("Model Creation", self.test_model_creation),
            ("Forward Pass", self.test_forward_pass),
            ("Context Trajectory", self.test_context_trajectory),
            ("Numerical Stability", self.test_numerical_stability),
            ("Determinism", self.test_determinism),
            ("Batch Independence", self.test_batch_independence),
            ("Context Initialization", self.test_context_initialization),
            ("Fixed-Point Shapes", self.test_fixed_point_shapes),
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
    tester = BasicFunctionalityTester()
    all_passed = tester.run_all_tests()

    if all_passed:
        print("\n✓ All basic functionality tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())
