"""
New-LLM 設定モジュール (Configuration Module)

使い方:
    from config import Config
    config = Config()

    # または個別設定
    from config.base import BaseConfig
    from config.phase1 import Phase1Config
    from config.phase2 import Phase2Config

    # 実験用ラッパー
    from config.experiment import DataConfig, Phase1TrainerConfig, Phase2TrainerConfig
"""

from .base import BaseConfig
from .phase1 import Phase1Config
from .phase2 import Phase2Config
from .experiment import DataConfig, Phase1TrainerConfig, Phase2TrainerConfig


class Config(BaseConfig):
    """
    統合設定クラス

    BaseConfig + Phase1Config + Phase2Config を統合。
    """

    # ========== Phase 1 設定 ==========
    phase1_max_iterations = Phase1Config.max_iterations
    phase1_convergence_threshold = Phase1Config.convergence_threshold
    phase1_learning_rate = Phase1Config.learning_rate
    phase1_batch_size = Phase1Config.batch_size
    phase1_gradient_clip = Phase1Config.gradient_clip
    phase1_context_noise = Phase1Config.context_noise
    phase1_early_stopping = Phase1Config.early_stopping
    phase1_early_stopping_threshold = Phase1Config.early_stopping_threshold

    # ========== Phase 2 設定 ==========
    phase2_learning_rate = Phase2Config.learning_rate
    phase2_epochs = Phase2Config.epochs
    phase2_patience = Phase2Config.patience
    phase2_gradient_clip = Phase2Config.gradient_clip
    phase2_batch_size = Phase2Config.batch_size
    phase2_min_batch_size = Phase2Config.min_batch_size
    phase2_max_batch_size = Phase2Config.max_batch_size
    phase2_memory_safety_factor = Phase2Config.memory_safety_factor
    phase2_freeze_embedding = Phase2Config.freeze_embedding

    @property
    def effective_phase2_batch_size(self) -> int:
        """Phase 2のバッチサイズを取得（GPUメモリベース自動計算）"""
        return Phase2Config.get_effective_batch_size(self.phase2_batch_size)


__all__ = [
    "Config",
    "BaseConfig",
    "Phase1Config",
    "Phase2Config",
    "DataConfig",
    "Phase1TrainerConfig",
    "Phase2TrainerConfig",
]
