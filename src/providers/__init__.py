"""
Providers - 依存性注入によるデータローダー・トレーナー切り替え

Usage:
    from src.providers import create_data_provider, create_phase1_trainer

    # データプロバイダー
    data_provider = create_data_provider("memory", config)  # or "storage"

    # Phase1トレーナー
    trainer = create_phase1_trainer("memory", model, config, device)
"""

from .data import (
    DataProvider,
    MemoryDataProvider,
    StorageDataProvider,
    create_data_provider,
)
from .trainer import Phase1Trainer, MemoryPhase1Trainer, StoragePhase1Trainer, create_phase1_trainer

__all__ = [
    # Data providers
    "DataProvider",
    "MemoryDataProvider",
    "StorageDataProvider",
    "create_data_provider",
    # Trainers
    "Phase1Trainer",
    "MemoryPhase1Trainer",
    "StoragePhase1Trainer",
    "create_phase1_trainer",
]
