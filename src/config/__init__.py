"""
Senri Configuration Module

定数、モデルプリセット、実験設定を提供。
全ての値は constants.py で一元管理され、デフォルト値は使用しない。

モデル作成:
    from src.config import SENRI_MODEL, PYTHIA_MODEL

    model = SENRI_MODEL()
    model = PYTHIA_MODEL()

設定値の参照:
    from src.config import MODEL_HIDDEN_SIZE, MODEL_NUM_HEADS

    # 設定値は constants.py に集約されている
"""

# Constants - 全ての設定値
from .constants import (
    # トークナイザー
    OPEN_CALM_TOKENIZER,
    OPEN_CALM_VOCAB_SIZE,
    PYTHIA_TOKENIZER,
    # モデルアーキテクチャ
    MODEL_HIDDEN_SIZE,
    MODEL_NUM_HEADS,
    MODEL_INTERMEDIATE_SIZE,
    MODEL_NUM_LAYERS,
    # SenriLayer
    SENRI_NUM_MEMORIES,
    SENRI_MEMORY_HEAD_DIM,
    SENRI_USE_DELTA_RULE,
    # PythiaLayer
    PYTHIA_ROTARY_PCT,
    PYTHIA_MAX_POSITION_EMBEDDINGS,
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
    # Model architecture constants
    "MODEL_HIDDEN_SIZE",
    "MODEL_NUM_HEADS",
    "MODEL_INTERMEDIATE_SIZE",
    "MODEL_NUM_LAYERS",
    # SenriLayer constants
    "SENRI_NUM_MEMORIES",
    "SENRI_MEMORY_HEAD_DIM",
    "SENRI_USE_DELTA_RULE",
    # PythiaLayer constants
    "PYTHIA_ROTARY_PCT",
    "PYTHIA_MAX_POSITION_EMBEDDINGS",
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
