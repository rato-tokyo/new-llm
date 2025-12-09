"""
Senri Configuration Module

Senri: Japanese LLM with Compressive Memory
"""

from .senri import SenriConfig, ModelTypeLiteral
from .pythia import (
    PythiaConfig,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP,
)
from .open_calm import (
    OPEN_CALM_TOKENIZER,
    OPEN_CALM_VOCAB_SIZE,
)

__all__ = [
    # Main config
    "SenriConfig",
    "ModelTypeLiteral",
    # Legacy config (for English baseline)
    "PythiaConfig",
    # OpenCALM tokenizer constants
    "OPEN_CALM_TOKENIZER",
    "OPEN_CALM_VOCAB_SIZE",
    # Constants
    "EARLY_STOPPING_PATIENCE",
    "GRADIENT_CLIP",
]
