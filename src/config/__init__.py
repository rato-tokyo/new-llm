"""
Senri Configuration Module

定数、モデルプリセット、実験設定を提供。
全ての値は constants.py で一元管理され、デフォルト値は使用しない。
PythiaLayerとSenriLayerは独立した設定を持つ。

モデル作成:
    from src.config import SENRI_MODEL, PYTHIA_MODEL

    model = SENRI_MODEL()
    model = PYTHIA_MODEL()

設定値の参照:
    from src.config import PYTHIA_HIDDEN_SIZE, SENRI_HIDDEN_SIZE

    # PythiaLayerとSenriLayerは独立した設定
"""

# Constants - 全ての設定値
from .constants import (
    # トークナイザー
    OPEN_CALM_TOKENIZER,
    OPEN_CALM_VOCAB_SIZE,
    PYTHIA_TOKENIZER,
    # Model共通
    MODEL_HIDDEN_SIZE,
    MODEL_VOCAB_SIZE,
    # PythiaLayer専用
    PYTHIA_HIDDEN_SIZE,
    PYTHIA_NUM_HEADS,
    PYTHIA_INTERMEDIATE_SIZE,
    PYTHIA_ROTARY_PCT,
    PYTHIA_MAX_POSITION_EMBEDDINGS,
    # SenriLayer専用
    SENRI_HIDDEN_SIZE,
    SENRI_NUM_HEADS,
    SENRI_INTERMEDIATE_SIZE,
    SENRI_NUM_MEMORIES,
    SENRI_MEMORY_HEAD_DIM,
    SENRI_USE_DELTA_RULE,
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
    # Tokenizer constants
    "OPEN_CALM_TOKENIZER",
    "OPEN_CALM_VOCAB_SIZE",
    "PYTHIA_TOKENIZER",
    # Model common constants
    "MODEL_HIDDEN_SIZE",
    "MODEL_VOCAB_SIZE",
    # PythiaLayer constants
    "PYTHIA_HIDDEN_SIZE",
    "PYTHIA_NUM_HEADS",
    "PYTHIA_INTERMEDIATE_SIZE",
    "PYTHIA_ROTARY_PCT",
    "PYTHIA_MAX_POSITION_EMBEDDINGS",
    # SenriLayer constants
    "SENRI_HIDDEN_SIZE",
    "SENRI_NUM_HEADS",
    "SENRI_INTERMEDIATE_SIZE",
    "SENRI_NUM_MEMORIES",
    "SENRI_MEMORY_HEAD_DIM",
    "SENRI_USE_DELTA_RULE",
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
