"""
Providers - 依存性注入によるデータローダー切り替え

Usage:
    from src.providers import create_data_provider

    # データプロバイダー
    data_provider = create_data_provider("memory", config)  # or "storage"
"""

from .data import (
    DataProvider,
    MemoryDataProvider,
    StorageDataProvider,
    create_data_provider,
)

__all__ = [
    # Data providers
    "DataProvider",
    "MemoryDataProvider",
    "StorageDataProvider",
    "create_data_provider",
]
