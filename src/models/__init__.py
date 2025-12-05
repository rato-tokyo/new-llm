"""New-LLM Models

MLA-Pythia architecture for KV cache compression.
Unified position encoding system (RoPE, ALiBi, NoPE).
KA cache for inference optimization.
"""

from .pythia import PythiaModel
from .mla_pythia import MLAPythiaModel
from .mla import MLAAttention, MLALayer
from .alibi import ALiBiCache, build_alibi_bias, build_alibi_bias_causal
from .position_encoding import (
    PositionEncodingType,
    PositionEncodingConfig,
    PositionEncoding,
    NoPositionEncoding,
    RotaryPositionEncoding,
    ALiBiPositionEncoding,
    create_position_encoding,
)
from .unified_pythia import UnifiedPythiaModel
from .ka_cache import KACachePythiaModel, KACache, KVCache
from .ka_adapter import KAAdapterPythiaModel, KAAdapter, KAAdapterAttention
from .ka_train import KATrainPythiaModel, KATrainAttention

__all__ = [
    # Legacy models
    'PythiaModel',
    'MLAPythiaModel',
    'MLAAttention',
    'MLALayer',
    'ALiBiCache',
    'build_alibi_bias',
    'build_alibi_bias_causal',
    # Unified position encoding system
    'PositionEncodingType',
    'PositionEncodingConfig',
    'PositionEncoding',
    'NoPositionEncoding',
    'RotaryPositionEncoding',
    'ALiBiPositionEncoding',
    'create_position_encoding',
    # Unified model
    'UnifiedPythiaModel',
    # KA cache
    'KACachePythiaModel',
    'KACache',
    'KVCache',
    # KA adapter
    'KAAdapterPythiaModel',
    'KAAdapter',
    'KAAdapterAttention',
    # KA train (案2)
    'KATrainPythiaModel',
    'KATrainAttention',
]
