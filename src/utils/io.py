"""
I/O utilities for new-llm.

Provides centralized output functions used across the codebase.
"""

import sys


def print_flush(*args, **kwargs):
    """Print with immediate flush for Colab/Jupyter compatibility."""
    print(*args, **kwargs)
    sys.stdout.flush()
