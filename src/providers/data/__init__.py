"""
Data Providers - データローダーの統一インターフェース

memory: 全データをメモリに展開（小〜中規模）
storage: mmapでディスクから直接読み込み（大規模）
"""

from .base import DataProvider
from .memory import MemoryDataProvider
from .storage import StorageDataProvider


def create_data_provider(
    mode: str,
    config,
    shuffle_samples: bool = False,
    shuffle_seed: int = 42
) -> DataProvider:
    """
    データプロバイダーを作成

    Args:
        mode: "memory" or "storage"
        config: ResidualConfig
        shuffle_samples: サンプルシャッフル（memoryモードのみ）
        shuffle_seed: シャッフル用シード

    Returns:
        DataProvider
    """
    if mode == "memory":
        return MemoryDataProvider(
            config,
            shuffle_samples=shuffle_samples,
            shuffle_seed=shuffle_seed
        )
    elif mode == "storage":
        return StorageDataProvider(config)
    else:
        raise ValueError(f"Unknown data mode: {mode}")


__all__ = [
    "DataProvider",
    "MemoryDataProvider",
    "StorageDataProvider",
    "create_data_provider",
]
