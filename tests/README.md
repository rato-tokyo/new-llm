# Test Suite for New-LLM

This directory contains comprehensive tests for the New-LLM implementation.

## Test Files

### 1. `test_basic_functionality.py`

**Purpose**: Verify core model functionality

**Tests**:
1. ✓ Model Creation - Model can be instantiated
2. ✓ Forward Pass - Correct output shapes
3. ✓ Context Trajectory - Context vectors returned when requested
4. ✓ Numerical Stability - No NaN or Inf in outputs
5. ✓ Determinism - Same input produces same output
6. ✓ Batch Independence - Batch processing matches individual processing
7. ✓ Context Initialization - Context updates correctly from zero
8. ✓ Fixed-Point Shapes - Fixed-point computation returns correct shapes

**Status**: ✅ All 8 tests pass

**Run**:
```bash
python3 tests/test_basic_functionality.py
```

---

### 2. `test_fixed_point_convergence.py`

**Purpose**: Verify fixed-point convergence behavior

**Tests**:
1. Repeated Token - Same token repeated should converge quickly
2. Random Tokens - Random tokens may not converge (observation)
3. Alternating Tokens - Simple pattern convergence (observation)
4. Block Pattern - Blocks of repeated tokens should converge
5. ✓ Gradient Flow - Fixed-point search should not compute gradients
6. ✓ Convergence Threshold - Threshold sensitivity check

**Status**: ⚠️ 4/6 pass (convergence tests fail on untrained model - expected)

**Run**:
```bash
python3 tests/test_fixed_point_convergence.py
```

**Note**: Tests 1, 3, 4 expect convergence, which only occurs after training.
These tests will pass once the model is trained on dialogue data.

---

### 3. `test_global_attractor.py` ⭐ **NEW**

**Purpose**: Detect global attractor problem (degenerate solution)

**Tests**:
1. ✓ Different Tokens → Different Contexts - Verify token diversity
2. ✓ Context Norm Consistency - Verify LayerNorm working correctly
3. ✓ Convergence Speed Variation - Observe convergence patterns
4. ✓ Fixed-Point Consistency - Same token → same fixed point

**Status**: ✅ All 4 tests pass

**Run**:
```bash
python3 tests/test_global_attractor.py
```

**What is Global Attractor Problem?**

A critical bug where **all tokens converge to the same identical fixed point**, losing token-specific information. The model appears to work (low loss) but is actually broken.

**Symptoms**:
- ❌ All different tokens → Same context (L2 distance < 0.001)
- ❌ Cosine similarity > 0.999 between all tokens
- ❌ Loss appears good but predictions are meaningless

**Causes**:
1. Using "simple" context updater (overwrites previous context)
2. Insufficient model capacity
3. Poor initialization

**Current Implementation**: ✅ Uses gated updater (LSTM-style) - Safe from this problem

---

## Test Results Summary

### Untrained Model

| Test Suite | Pass Rate | Notes |
|------------|-----------|-------|
| **Basic Functionality** | 8/8 (100%) | ✅ All pass |
| **Fixed-Point Convergence** | 4/6 (67%) | ⚠️ Expected for untrained model |
| **Global Attractor Detection** | 4/4 (100%) | ✅ No problem detected |

### After Training (Expected)

| Test Suite | Pass Rate | Expected Outcome |
|------------|-----------|------------------|
| **Basic Functionality** | 8/8 (100%) | ✅ Should still pass |
| **Fixed-Point Convergence** | 6/6 (100%) | ✅ Convergence tests should pass |
| **Global Attractor Detection** | 4/4 (100%) | ✅ Should still pass |

---

## Understanding the Results

### Why convergence tests fail on untrained models

**Reason**: Fixed-point convergence requires the model to learn stable dynamics.

An untrained model has:
- ❌ Random weights
- ❌ No learned attractors
- ❌ Unstable dynamics

A trained model has:
- ✅ Learned weights
- ✅ Stable attractors (fixed points)
- ✅ Convergent dynamics

**Expected Behavior**:
1. **Before training**: 0% convergence on repeated tokens
2. **After Phase 1 training**: >90% convergence on repeated tokens
3. **After Phase 2 training**: >95% convergence on dialogue data

---

## How to Use These Tests

### 1. Development Workflow

**Before making changes**:
```bash
python3 tests/test_basic_functionality.py
```
→ Ensure basic functionality works

**After making changes**:
```bash
python3 tests/test_basic_functionality.py
```
→ Verify changes didn't break anything

### 2. Training Workflow

**Before training**:
```bash
python3 tests/test_fixed_point_convergence.py
```
→ Baseline convergence (expect 0%)

**After Phase 1 training**:
```bash
python3 tests/test_fixed_point_convergence.py
```
→ Check convergence rate (expect >90%)

**After Phase 2 training**:
```bash
python3 tests/test_fixed_point_convergence.py
```
→ Verify convergence maintained (expect >95%)

---

## Adding New Tests

### Test Template

```python
def test_new_feature(self):
    """Test description"""
    print("\n" + "-" * 70)
    print("TEST N: Test Name")
    print("-" * 70)

    # Test logic
    try:
        # ... test code ...

        print(f"Results:")
        # ... print results ...

        if condition:
            print(f"Status: ✓ PASS")
            return True
        else:
            print(f"Status: ✗ FAIL")
            return False

    except Exception as e:
        print(f"Status: ✗ FAIL - {e}")
        return False
```

### Best Practices

1. **Clear test names**: Describe what is being tested
2. **Expected behavior**: Document what should happen
3. **Failure messages**: Explain why a test might fail
4. **Isolation**: Tests should not depend on each other
5. **Cleanup**: Clean up any temporary files/state

---

## Troubleshooting

### Test fails with "ModuleNotFoundError"

**Solution**: Make sure you're running from project root
```bash
cd /path/to/new-llm
python3 tests/test_basic_functionality.py
```

### Fixed-point tests all fail

**Solution**: This is expected for untrained models. Train first:
```bash
python3 train_dialogue.py
```

### Tests pass but training fails

**Solution**: Tests verify basic functionality, not training logic.
Check `train_dialogue.py` for training-specific issues.

---

## CI/CD Integration (Future)

These tests can be integrated into GitHub Actions:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run basic tests
        run: python3 tests/test_basic_functionality.py
```

---

## Summary

✅ **Basic functionality tests**: Always should pass
⚠️ **Convergence tests**: Pass after training

Use these tests to:
- Verify implementation correctness
- Catch regressions
- Validate training progress
