"""
Phase 1 Trainers - CVFP固定点学習トレーナー

memory: 全データをメモリに展開して処理（小〜中規模）
storage: mmapでディスクから直接処理（大規模）
"""

from .base import Phase1Trainer
from .memory import MemoryPhase1Trainer
from .storage import StoragePhase1Trainer


def create_phase1_trainer(mode: str, model, config, device) -> Phase1Trainer:
    """
    Phase 1トレーナーを作成

    Args:
        mode: "memory" or "storage"
        model: LLMモデル
        config: ResidualConfig
        device: 計算デバイス

    Returns:
        Phase1Trainer
    """
    if mode == "memory":
        return MemoryPhase1Trainer(model, config, device)
    elif mode == "storage":
        return StoragePhase1Trainer(model, config, device)
    else:
        raise ValueError(f"Unknown trainer mode: {mode}")


__all__ = [
    "Phase1Trainer",
    "MemoryPhase1Trainer",
    "StoragePhase1Trainer",
    "create_phase1_trainer",
]
