"""
New-LLM 設定モジュール (Configuration Module)

使い方:
    from config import PythiaConfig, ContextPythiaConfig, Phase1Config
    config = PythiaConfig()
    phase1_config = Phase1Config()
"""

from .phase1 import Phase1Config
from .pythia import PythiaConfig, ContextPythiaConfig

__all__ = [
    "Phase1Config",
    "PythiaConfig",
    "ContextPythiaConfig",
]
