"""Loss functions for New-LLM."""

from .diversity import (
    DIVERSITY_ALGORITHMS,
    ALGORITHM_DESCRIPTIONS,
    DEFAULT_ALGORITHM,
    oacd_loss,
)

__all__ = [
    'DIVERSITY_ALGORITHMS',
    'ALGORITHM_DESCRIPTIONS',
    'DEFAULT_ALGORITHM',
    'oacd_loss',
]
