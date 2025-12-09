"""
Senri Configuration Module

定数、モデルプリセット、実験設定を提供。

モデル作成:
    from src.config import SENRI_MODEL, PYTHIA_MODEL

    model = SENRI_MODEL()
    model = PYTHIA_MODEL()

    # または直接レイヤー指定
    from src.models import SenriModel, SenriLayer, PythiaLayer

    model = SenriModel([
        SenriLayer(),
        PythiaLayer(),
        PythiaLayer(),
        PythiaLayer(),
        PythiaLayer(),
        PythiaLayer(),
    ])
"""

# Constants
from .constants import (
    OPEN_CALM_TOKENIZER,
    OPEN_CALM_VOCAB_SIZE,
    PYTHIA_TOKENIZER,
)

# Model presets
from .models import (
    PYTHIA_MODEL,
    SENRI_MODEL,
    SENRI_MULTI_MEMORY_MODEL,
    SENRI_ONLY_MODEL,
    MODEL_PRESETS,
    create_model,
)

# Experiment configurations
from .experiments import ExperimentConfig

__all__ = [
    # Constants
    "OPEN_CALM_TOKENIZER",
    "OPEN_CALM_VOCAB_SIZE",
    "PYTHIA_TOKENIZER",
    # Model presets
    "PYTHIA_MODEL",
    "SENRI_MODEL",
    "SENRI_MULTI_MEMORY_MODEL",
    "SENRI_ONLY_MODEL",
    "MODEL_PRESETS",
    "create_model",
    # Experiment config
    "ExperimentConfig",
]
