"""
Data Providers - データローダーの統一インターフェース

memory: 全データをメモリに展開（小〜中規模）
"""

from .base import DataProvider
from .memory import MemoryDataProvider


__all__ = [
    "DataProvider",
    "MemoryDataProvider",
]
