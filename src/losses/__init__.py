"""Loss functions for New-LLM."""

from .diversity import (
    DIVERSITY_ALGORITHMS,
    ALGORITHM_DESCRIPTIONS,
    HIGH_COST_ALGORITHMS,
    mcdl_loss,
    odcm_loss,
    due_loss,
    ctm_loss,
    udel_loss,
    sdl_loss,
    unif_loss,
    decorr_loss,
    nuc_loss,
    hsic_loss,
    infonce_loss,
    wmse_loss,
)

__all__ = [
    'DIVERSITY_ALGORITHMS',
    'ALGORITHM_DESCRIPTIONS',
    'HIGH_COST_ALGORITHMS',
    'mcdl_loss',
    'odcm_loss',
    'due_loss',
    'ctm_loss',
    'udel_loss',
    'sdl_loss',
    'unif_loss',
    'decorr_loss',
    'nuc_loss',
    'hsic_loss',
    'infonce_loss',
    'wmse_loss',
]
