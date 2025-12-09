"""
Senri Configuration Module

Senri: Japanese LLM with Compressive Memory
"""

from .senri import SenriConfig
from .pythia import (
    PythiaConfig,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP,
)
from .open_calm import (
    OpenCalmConfig,
    OPEN_CALM_TOKENIZER,
    OPEN_CALM_VOCAB_SIZE,
)
from .experiment import ExperimentConfig
from .models import (
    InfiniConfig,
    MultiMemoryConfig,
    ModelConfigType,
    ModelTypeLiteral,
)

__all__ = [
    # Main config (recommended)
    "SenriConfig",
    # Legacy configs
    "PythiaConfig",
    "OpenCalmConfig",
    # OpenCALM tokenizer constants
    "OPEN_CALM_TOKENIZER",
    "OPEN_CALM_VOCAB_SIZE",
    # Experimental model configs
    "InfiniConfig",
    "MultiMemoryConfig",
    "ModelConfigType",
    "ModelTypeLiteral",
    # Experiment config
    "ExperimentConfig",
    # Constants
    "EARLY_STOPPING_PATIENCE",
    "GRADIENT_CLIP",
]
