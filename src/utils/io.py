"""
I/O utilities for new-llm.

Provides centralized output functions used across the codebase.
"""

import sys
from typing import Any


def print_flush(*args: Any, **kwargs: Any) -> None:
    """Print with immediate flush for Colab/Jupyter compatibility."""
    print(*args, **kwargs)
    sys.stdout.flush()
