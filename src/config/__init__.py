"""
Experiment configuration defaults.
"""

from .experiment_defaults import (
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP,
    DEFAULT_LR,
    DEFAULT_KV_DIM,
    DEFAULT_ALIBI_SLOPE,
    DEFAULT_ROTARY_PCT,
)

__all__ = [
    "EARLY_STOPPING_PATIENCE",
    "GRADIENT_CLIP",
    "DEFAULT_LR",
    "DEFAULT_KV_DIM",
    "DEFAULT_ALIBI_SLOPE",
    "DEFAULT_ROTARY_PCT",
]
