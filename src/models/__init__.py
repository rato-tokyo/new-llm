"""New-LLM Models

Pythia-70M + DProj-Pythia architecture for KV cache compression.
"""

from .dproj import DiverseProjection, DiverseProjectionLayer
from .pythia import PythiaModel
from .dproj_pythia import DProjPythiaModel

__all__ = [
    'DiverseProjection',
    'DiverseProjectionLayer',
    'PythiaModel',
    'DProjPythiaModel',
]
