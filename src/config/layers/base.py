"""
Base Layer Configuration

全レイヤー共通の設定を定義。
"""

from dataclasses import dataclass


@dataclass
class BaseLayerConfig:
    """レイヤー設定の基底クラス

    全レイヤー共通の設定を定義。
    """

    hidden_size: int = 512
    num_attention_heads: int = 8
    intermediate_size: int = 2048
