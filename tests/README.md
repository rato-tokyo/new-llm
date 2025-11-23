# Tests Directory

This directory contains test scripts for the New-LLM project.

## Available Tests

### test_refactored.py

Quick development test for the refactored CVFP implementation.

**Purpose**: Verify that the CVFPLayer architecture works correctly with minimal data.

**Usage**:
```bash
python3 tests/test_refactored.py
```

**Features**:
- Tests with only 10 tokens (fast execution)
- Validates distribution regularization
- Checks for identity mapping issues
- Analyzes context diversity

**Expected Output**:
- Model creation confirmation
- Phase 1 training results
- Distribution loss metrics
- Identity mapping similarity
- Context diversity analysis

## Running Tests

All tests should be run from the project root directory:

```bash
# From project root
cd /path/to/new-llm
python3 tests/test_refactored.py
```
