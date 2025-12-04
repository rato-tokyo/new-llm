"""
New-LLM 設定モジュール (Configuration Module)

Pythia-70M + DProj-Pythia用の設定。

使い方:
    from config import DProjTrainingConfig, PythiaConfig, DProjPythiaConfig
"""

from .dproj import DProjTrainingConfig
from .pythia import PythiaConfig, DProjPythiaConfig

# Backward compatibility aliases
from .dproj import Phase1Config
from .pythia import ContextPythiaConfig


__all__ = [
    "DProjTrainingConfig",
    "PythiaConfig",
    "DProjPythiaConfig",
    # Backward compatibility
    "Phase1Config",
    "ContextPythiaConfig",
]
