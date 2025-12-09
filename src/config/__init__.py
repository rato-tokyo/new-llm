"""
Senri Configuration Module

Senri: Japanese LLM with Compressive Memory

設定は4つのモジュールに分離:
- senri.py: Senriモデル全体の設定（SenriModelConfig）
- layers.py: レイヤー構成（SenriLayerConfig, PythiaLayerConfig）
- experiment.py: 実験設定（訓練、評価）
- open_calm.py: トークナイザー定数

Usage:
    from src.config import SenriModelConfig

    # 方法1: SenriModelConfigを使用（推奨）
    config = SenriModelConfig()
    model = config.create_model()

    # 方法2: LayerConfigリストを使用
    from src.config import default_senri_layers
    from src.models import create_model
    layers = default_senri_layers()
    model = create_model(layers)

    # 方法3: カスタム構成
    from src.config import SenriLayerConfig, PythiaLayerConfig
    layers = [
        SenriLayerConfig(use_multi_memory=True, num_memories=8),
        PythiaLayerConfig(),
        PythiaLayerConfig(),
    ]
    model = create_model(layers)
"""

# Senri model configuration (main entry point)
from .senri import SenriModelConfig

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
    # Model config
    "SenriModelConfig",
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
