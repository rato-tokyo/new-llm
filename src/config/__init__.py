"""
New-LLM Configuration Module
"""

from .pythia import (
    PythiaConfig,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP,
)
from .experiment import ExperimentConfig
from .models import (
    InfiniConfig,
    MultiMemoryConfig,
    ModelConfigType,
    ModelTypeLiteral,
)

__all__ = [
    # Base model config
    "PythiaConfig",
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
