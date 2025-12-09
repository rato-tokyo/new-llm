"""
Senri Memory Module

圧縮メモリの管理を担当:
- CompressiveMemory: 基本的な圧縮メモリ（Linear Attention形式）
- FreezableMemoryMixin: freeze/unfreeze機能
- MemoryExportMixin: export/import機能

使用例:
    from src.models.memory import CompressiveMemory

    memory = CompressiveMemory(
        hidden_size=512,
        num_memories=4,
        use_delta_rule=True,
    )

    # メモリ更新
    memory.update(keys, values)

    # メモリ検索
    output = memory.retrieve(queries)

    # freeze
    memory.freeze([0, 1])  # メモリ0, 1をfreeze
"""

from .base import CompressiveMemory
from .mixins import FreezableMemoryMixin, MemoryExportMixin

__all__ = [
    "CompressiveMemory",
    "FreezableMemoryMixin",
    "MemoryExportMixin",
]
