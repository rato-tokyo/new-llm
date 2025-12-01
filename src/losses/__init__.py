"""Loss functions for New-LLM."""

from .diversity import (
    DIVERSITY_ALGORITHMS,
    ALGORITHM_DESCRIPTIONS,
    mcdl_loss,
    odcm_loss,
    oacd_loss,
    wmse_loss,
)

__all__ = [
    'DIVERSITY_ALGORITHMS',
    'ALGORITHM_DESCRIPTIONS',
    'mcdl_loss',
    'odcm_loss',
    'oacd_loss',
    'wmse_loss',
]
