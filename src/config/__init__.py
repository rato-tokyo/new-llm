"""
Senri Configuration Module

設定は3つのサブパッケージに分離:
- layers/: レイヤー設定（SenriLayerConfig, PythiaLayerConfig）
- models/: モデル設定（SenriModelConfig, PythiaModelConfig）
- experiments/: 実験設定（ExperimentConfig）
- constants.py: 定数（OPEN_CALM_TOKENIZER等）

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

# Constants
from .constants import (
    OPEN_CALM_TOKENIZER,
    OPEN_CALM_VOCAB_SIZE,
    PYTHIA_TOKENIZER,
)

# Layer configurations
from .layers import (
    BaseLayerConfig,
    PythiaLayerConfig,
    SenriLayerConfig,
    LayerConfigType,
    default_senri_layers,
    default_pythia_layers,
)

# Model configurations
from .models import (
    BaseModelConfig,
    PythiaModelConfig,
    SenriModelConfig,
)

# Experiment configurations
from .experiments import (
    ExperimentConfig,
)

__all__ = [
    # Constants
    "OPEN_CALM_TOKENIZER",
    "OPEN_CALM_VOCAB_SIZE",
    "PYTHIA_TOKENIZER",
    # Layer configs
    "BaseLayerConfig",
    "PythiaLayerConfig",
    "SenriLayerConfig",
    "LayerConfigType",
    "default_senri_layers",
    "default_pythia_layers",
    # Model configs
    "BaseModelConfig",
    "PythiaModelConfig",
    "SenriModelConfig",
    # Experiment configs
    "ExperimentConfig",
]
