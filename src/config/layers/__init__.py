"""
Layer Configuration Package

レイヤー設定をまとめてエクスポート。
"""

from typing import Union

from .base import BaseLayerConfig
from .pythia import PythiaLayerConfig
from .senri import SenriLayerConfig

# Type alias
LayerConfigType = Union[PythiaLayerConfig, SenriLayerConfig]


def default_senri_layers(
    num_senri: int = 1,
    num_pythia: int = 5,
    **senri_kwargs,
) -> list[LayerConfigType]:
    """デフォルトのSenriレイヤー構成を生成

    Args:
        num_senri: Senriレイヤー数（先頭に配置）
        num_pythia: Pythiaレイヤー数
        **senri_kwargs: SenriLayerConfigに渡す追加引数

    Returns:
        レイヤー設定のリスト
    """
    return [
        SenriLayerConfig(**senri_kwargs) for _ in range(num_senri)
    ] + [
        PythiaLayerConfig() for _ in range(num_pythia)
    ]


def default_pythia_layers(num_layers: int = 6) -> list[LayerConfigType]:
    """Pythiaのみのレイヤー構成を生成（ベースライン用）

    Args:
        num_layers: レイヤー数

    Returns:
        PythiaLayerConfigのリスト
    """
    return [PythiaLayerConfig() for _ in range(num_layers)]


__all__ = [
    "BaseLayerConfig",
    "PythiaLayerConfig",
    "SenriLayerConfig",
    "LayerConfigType",
    "default_senri_layers",
    "default_pythia_layers",
]
