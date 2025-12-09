"""
Senri Configuration Module

定数と実験設定を提供。

モデル作成は src.models を使用:
    from src.models import TransformerLM, senri_layers, pythia_layers

    model = TransformerLM(layers=senri_layers(), vocab_size=52000)
"""

# Constants
from .constants import (
    OPEN_CALM_TOKENIZER,
    OPEN_CALM_VOCAB_SIZE,
    PYTHIA_TOKENIZER,
)

# Experiment configurations
from .experiments import ExperimentConfig

__all__ = [
    # Constants
    "OPEN_CALM_TOKENIZER",
    "OPEN_CALM_VOCAB_SIZE",
    "PYTHIA_TOKENIZER",
    # Experiment config
    "ExperimentConfig",
]
