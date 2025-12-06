"""
Transformer Layer Implementations

全レイヤータイプを提供:
- PythiaLayer: 標準Pythia (RoPE + Softmax Attention)
- InfiniLayer: Infini-Attention (Memory + Linear Attention)
- MultiMemoryLayer: 複数独立メモリ
- HierarchicalLayer: 階層的メモリ

使用例:
    from src.models.layers import PythiaLayer, InfiniLayer

    # 標準Pythiaレイヤー
    layer = PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048)

    # Infini-Attentionレイヤー
    layer = InfiniLayer(hidden_size=512, num_heads=8, intermediate_size=2048)
"""

from .base import BaseLayer
from .pythia import PythiaLayer, PythiaAttention
from .infini import InfiniLayer, InfiniAttention
from .multi_memory import MultiMemoryLayer, MultiMemoryAttention
from .hierarchical import HierarchicalLayer, HierarchicalAttention

__all__ = [
    # Base
    'BaseLayer',
    # Pythia
    'PythiaLayer',
    'PythiaAttention',
    # Infini
    'InfiniLayer',
    'InfiniAttention',
    # Multi-Memory
    'MultiMemoryLayer',
    'MultiMemoryAttention',
    # Hierarchical
    'HierarchicalLayer',
    'HierarchicalAttention',
]
