"""
New-LLM Configuration Module
"""

from .pythia import (
    PythiaConfig,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP,
)
from .open_calm import OpenCalmConfig
from .experiment import ExperimentConfig
from .models import (
    InfiniConfig,
    MultiMemoryConfig,
    ModelConfigType,
    ModelTypeLiteral,
)

__all__ = [
    # Base model configs
    "PythiaConfig",
    "OpenCalmConfig",
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
