#!/usr/bin/env python3
"""
Unified test suite for New-LLM project

Runs all tests and provides a comprehensive summary.

Usage:
    python tests/test_all.py
    python tests/test_all.py --fast  # Skip slow tests
"""

import sys
import os
import argparse

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def run_checkpoint_compatibility_tests():
    """Run checkpoint compatibility tests"""
    from tests.test_checkpoint_compatibility import main
    print("\n" + "="*80)
    print("Running Checkpoint Compatibility Tests")
    print("="*80)
    return main()


def run_context_expansion_tests():
    """Run context expansion tests"""
    from tests.test_context_expansion import main
    print("\n" + "="*80)
    print("Running Context Expansion Tests")
    print("="*80)
    return main()


def run_dolly_training_tests():
    """Run Dolly training tests"""
    from tests.test_dolly_training import main
    print("\n" + "="*80)
    print("Running Dolly Training Tests")
    print("="*80)
    return main()


def run_gradient_freezing_tests():
    """Run gradient freezing tests"""
    from tests.test_gradient_freezing import main
    print("\n" + "="*80)
    print("Running Gradient Freezing Tests")
    print("="*80)
    return main()


def main():
    """Run all tests"""
    parser = argparse.ArgumentParser(description='Run all New-LLM tests')
    parser.add_argument('--fast', action='store_true',
                       help='Skip slow tests')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("New-LLM Test Suite")
    print("="*80)

    results = []

    # Run all test modules
    try:
        results.append(("Checkpoint Compatibility", run_checkpoint_compatibility_tests()))
    except Exception as e:
        print(f"❌ Checkpoint Compatibility tests failed: {e}")
        results.append(("Checkpoint Compatibility", False))

    try:
        results.append(("Context Expansion", run_context_expansion_tests()))
    except Exception as e:
        print(f"❌ Context Expansion tests failed: {e}")
        results.append(("Context Expansion", False))

    try:
        results.append(("Dolly Training", run_dolly_training_tests()))
    except Exception as e:
        print(f"❌ Dolly Training tests failed: {e}")
        results.append(("Dolly Training", False))

    if not args.fast:
        try:
            results.append(("Gradient Freezing", run_gradient_freezing_tests()))
        except Exception as e:
            print(f"❌ Gradient Freezing tests failed: {e}")
            results.append(("Gradient Freezing", False))

    # Summary
    print("\n" + "="*80)
    print("OVERALL TEST SUMMARY")
    print("="*80)

    for test_module, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_module}")

    all_passed = all(passed for _, passed in results)

    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED")
        print(f"\nRan {len(results)} test modules successfully.")
    else:
        failed_count = sum(1 for _, passed in results if not passed)
        print(f"❌ {failed_count} TEST MODULE(S) FAILED")
        print(f"\nPassed: {len(results) - failed_count}/{len(results)}")
    print("="*80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
