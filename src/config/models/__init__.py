"""
Model Configuration Package

モデル設定をまとめてエクスポート。
"""

from .base import BaseModelConfig
from .pythia import PythiaModelConfig
from .senri import SenriModelConfig

__all__ = [
    "BaseModelConfig",
    "PythiaModelConfig",
    "SenriModelConfig",
]
