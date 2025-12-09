"""
Model Configuration Package

モデル設定をまとめてエクスポート。
"""

from .pythia import PythiaModelConfig
from .senri import SenriModelConfig

__all__ = [
    "PythiaModelConfig",
    "SenriModelConfig",
]
