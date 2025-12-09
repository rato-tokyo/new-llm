"""
Senri Configuration Module

定数、モデル設定、実験設定を提供。

モデル作成:
    from src.config import SENRI_CONFIG, create_model_from_config

    model = create_model_from_config(SENRI_CONFIG)
"""

# Constants
from .constants import (
    OPEN_CALM_TOKENIZER,
    OPEN_CALM_VOCAB_SIZE,
    PYTHIA_TOKENIZER,
)

# Model configurations
from .models import (
    ModelConfig,
    PYTHIA_CONFIG,
    SENRI_CONFIG,
    SENRI_MULTI_MEMORY_CONFIG,
    SENRI_ONLY_CONFIG,
    create_model_from_config,
)

# Experiment configurations
from .experiments import ExperimentConfig

__all__ = [
    # Constants
    "OPEN_CALM_TOKENIZER",
    "OPEN_CALM_VOCAB_SIZE",
    "PYTHIA_TOKENIZER",
    # Model config
    "ModelConfig",
    "PYTHIA_CONFIG",
    "SENRI_CONFIG",
    "SENRI_MULTI_MEMORY_CONFIG",
    "SENRI_ONLY_CONFIG",
    "create_model_from_config",
    # Experiment config
    "ExperimentConfig",
]
