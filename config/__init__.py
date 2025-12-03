"""
New-LLM 設定モジュール (Configuration Module)

Pythia-70M + Context-Pythia用の設定。

使い方:
    from config import Phase1Config, PythiaConfig, ContextPythiaConfig
"""

from .phase1 import Phase1Config
from .pythia import PythiaConfig, ContextPythiaConfig


__all__ = [
    "Phase1Config",
    "PythiaConfig",
    "ContextPythiaConfig",
]
