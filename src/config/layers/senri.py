"""
Senri Layer Configuration

Infini-Attention + Multi-Memory を組み合わせた特殊レイヤーの設定。
"""

from dataclasses import dataclass

from .base import BaseLayerConfig


@dataclass
class SenriLayerConfig(BaseLayerConfig):
    """Senriレイヤー設定

    Infini-Attention + Multi-Memory を組み合わせた特殊レイヤー。
    圧縮メモリによる長期記憶とLinear Attentionを持つ。

    Attributes:
        num_memory_banks: Infiniメモリバンク数
        segments_per_bank: バンクあたりのセグメント数
        num_memories: MultiMemoryのメモリ数（Detail Memory）
        use_delta_rule: Delta Rule更新を使用
        use_multi_memory: MultiMemoryを使用するか（FalseならInfiniのみ）
    """

    # Infini-Attention設定
    num_memory_banks: int = 1
    segments_per_bank: int = 4

    # Multi-Memory設定
    num_memories: int = 4
    use_multi_memory: bool = False  # デフォルトはInfiniのみ

    # 共通メモリ設定
    use_delta_rule: bool = True
