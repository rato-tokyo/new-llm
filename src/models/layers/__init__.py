"""
Transformer Layer Implementations

全レイヤータイプを提供:
- PythiaLayer: 標準Pythia (RoPE + Softmax Attention)
- SenriLayer: 圧縮メモリ + Linear Attention (統一版)

使用例:
    from src.models.layers import PythiaLayer, SenriLayer

    # 標準Pythiaレイヤー
    layer = PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048)

    # Senriレイヤー（メモリ1つ = 旧InfiniAttention相当）
    layer = SenriLayer(hidden_size=512, num_heads=8, intermediate_size=2048)

    # Senriレイヤー（複数メモリ = 旧MultiMemory相当）
    layer = SenriLayer(hidden_size=512, num_heads=8, intermediate_size=2048, num_memories=4)
"""

from .base import BaseLayer
from .pythia import PythiaLayer, PythiaAttention
from .senri import SenriLayer, SenriAttention

__all__ = [
    # Base
    'BaseLayer',
    # Pythia
    'PythiaLayer',
    'PythiaAttention',
    # Senri
    'SenriLayer',
    'SenriAttention',
]
