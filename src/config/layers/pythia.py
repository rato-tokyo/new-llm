"""
Pythia Layer Configuration

標準的なTransformerレイヤー（RoPE + Softmax Attention）の設定。
"""

from dataclasses import dataclass

from .base import BaseLayerConfig


@dataclass
class PythiaLayerConfig(BaseLayerConfig):
    """Pythiaレイヤー設定

    標準的なTransformerレイヤー（RoPE + Softmax Attention）。

    Attributes:
        rotary_pct: Rotary Embeddingの割合（0.0でNoPE）
        max_position_embeddings: 最大シーケンス長
    """

    rotary_pct: float = 0.25
    max_position_embeddings: int = 2048
