"""
Senri Configuration Module

Senri: Japanese LLM with Compressive Memory

設定は3つのモジュールに分離:
- layers.py: レイヤー構成（SenriLayerConfig, PythiaLayerConfig）
- experiment.py: 実験設定（訓練、評価）
- open_calm.py: トークナイザー定数

Usage:
    from src.config import (
        # レイヤー設定
        SenriLayerConfig,
        PythiaLayerConfig,
        default_senri_layers,
        default_pythia_layers,
        # 実験設定
        ExperimentConfig,
        # トークナイザー
        OPEN_CALM_TOKENIZER,
        OPEN_CALM_VOCAB_SIZE,
    )
    from src.models import create_model

    # モデル作成
    layers = default_senri_layers()
    model = create_model(layers)

    # 実験設定
    exp_config = ExperimentConfig(num_epochs=50)
"""

# Layer configurations
from .layers import (
    BaseLayerConfig,
    PythiaLayerConfig,
    SenriLayerConfig,
    LayerConfigType,
    default_senri_layers,
    default_pythia_layers,
)

# Experiment configuration
from .experiment import ExperimentConfig

# OpenCALM tokenizer constants
from .open_calm import (
    OPEN_CALM_TOKENIZER,
    OPEN_CALM_VOCAB_SIZE,
)

__all__ = [
    # Layer configs
    "BaseLayerConfig",
    "PythiaLayerConfig",
    "SenriLayerConfig",
    "LayerConfigType",
    "default_senri_layers",
    "default_pythia_layers",
    # Experiment config
    "ExperimentConfig",
    # OpenCALM constants
    "OPEN_CALM_TOKENIZER",
    "OPEN_CALM_VOCAB_SIZE",
]
