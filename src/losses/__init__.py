"""Loss functions for New-LLM."""

from .diversity import (
    DIVERSITY_ALGORITHMS,
    ALGORITHM_DESCRIPTIONS,
    HIGH_COST_ALGORITHMS,
    mcdl_loss,
    odcm_loss,
    sdl_loss,
    nuc_loss,
    wmse_loss,
)

__all__ = [
    'DIVERSITY_ALGORITHMS',
    'ALGORITHM_DESCRIPTIONS',
    'HIGH_COST_ALGORITHMS',
    'mcdl_loss',
    'odcm_loss',
    'sdl_loss',
    'nuc_loss',
    'wmse_loss',
]
