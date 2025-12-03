"""
New-LLM 設定モジュール (Configuration Module)

使い方:
    from config import PythiaConfig, ContextPythiaConfig
    config = PythiaConfig()
"""

from .pythia import PythiaConfig, ContextPythiaConfig

__all__ = [
    "PythiaConfig",
    "ContextPythiaConfig",
]
