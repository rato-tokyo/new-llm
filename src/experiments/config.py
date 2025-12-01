"""
実験用設定クラス (Experiment Configuration Classes)

各トレーナー・データプロバイダー用の設定ラッパー。
ResidualConfigから必要な属性のみを抽出し、オプションで上書き可能。

Usage:
    from config import ResidualConfig
    from src.experiments.config import DataConfig, Phase1Config, Phase2Config

    base = ResidualConfig()
    device = torch.device("cuda")

    # データ設定
    data_cfg = DataConfig.from_base(base, num_samples=100)

    # Phase 1設定（context_dimを上書き）
    p1_cfg = Phase1Config.from_base(base, device, context_dim=1000)

    # Phase 2設定
    p2_cfg = Phase2Config.from_base(base, device, context_dim=1000)
"""

from dataclasses import dataclass
from typing import Union, Optional

import torch

from config import ResidualConfig


@dataclass
class DataConfig:
    """MemoryDataProvider用の設定"""
    tokenizer_name: str
    dataset_name: str
    dataset_split: str
    cache_dir: str
    num_samples: int
    val_data_source: str
    val_text_file: str = "./data/example_val.txt"

    @classmethod
    def from_base(
        cls,
        base: ResidualConfig,
        num_samples: Optional[int] = None,
        val_text_file: str = "./data/example_val.txt"
    ) -> "DataConfig":
        """ResidualConfigから生成"""
        return cls(
            tokenizer_name=base.tokenizer_name,
            dataset_name=base.dataset_name,
            dataset_split=base.dataset_split,
            cache_dir=base.cache_dir,
            num_samples=num_samples if num_samples is not None else base.num_samples,
            val_data_source=base.val_data_source,
            val_text_file=val_text_file,
        )


@dataclass
class Phase1Config:
    """Phase 1 Trainer用の設定"""
    # アーキテクチャ
    context_dim: int
    embed_dim: int
    num_layers: int
    num_input_tokens: int

    # 学習パラメータ
    phase1_learning_rate: float
    phase1_max_iterations: int
    phase1_convergence_threshold: float
    phase1_context_noise: float
    phase1_batch_size: int
    phase1_gradient_clip: float
    dist_reg_weight: float

    # Validation Early Stopping
    phase1_val_early_stopping: bool
    phase1_val_frequency: int
    phase1_val_sample_size: int
    phase1_val_patience: int

    # デバイス
    device: Union[str, torch.device]

    @classmethod
    def from_base(
        cls,
        base: ResidualConfig,
        device: Union[str, torch.device],
        context_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_input_tokens: Optional[int] = None,
        phase1_learning_rate: Optional[float] = None,
        phase1_max_iterations: Optional[int] = None,
        dist_reg_weight: Optional[float] = None,
    ) -> "Phase1Config":
        """ResidualConfigから生成（オプションで上書き可能）"""
        return cls(
            context_dim=context_dim if context_dim is not None else base.context_dim,
            embed_dim=base.embed_dim,
            num_layers=num_layers if num_layers is not None else base.num_layers,
            num_input_tokens=num_input_tokens if num_input_tokens is not None else base.num_input_tokens,
            phase1_learning_rate=phase1_learning_rate if phase1_learning_rate is not None else base.phase1_learning_rate,
            phase1_max_iterations=phase1_max_iterations if phase1_max_iterations is not None else base.phase1_max_iterations,
            phase1_convergence_threshold=base.phase1_convergence_threshold,
            phase1_context_noise=base.phase1_context_noise,
            phase1_batch_size=base.phase1_batch_size,
            phase1_gradient_clip=base.phase1_gradient_clip,
            dist_reg_weight=dist_reg_weight if dist_reg_weight is not None else base.dist_reg_weight,
            phase1_val_early_stopping=getattr(base, 'phase1_val_early_stopping', True),
            phase1_val_frequency=getattr(base, 'phase1_val_frequency', 5),
            phase1_val_sample_size=getattr(base, 'phase1_val_sample_size', 10000),
            phase1_val_patience=getattr(base, 'phase1_val_patience', 2),
            device=device,
        )


@dataclass
class Phase2Config:
    """Phase 2 Trainer用の設定"""
    # アーキテクチャ
    context_dim: int
    embed_dim: int
    num_layers: int
    num_input_tokens: int

    # 学習パラメータ
    phase2_learning_rate: float
    phase2_epochs: int
    phase2_patience: int
    phase2_batch_size: Optional[int]
    phase2_gradient_clip: float
    phase2_freeze_embedding: bool

    # メモリ管理
    phase2_memory_safety_factor: float
    phase2_min_batch_size: int
    phase2_max_batch_size: int

    # デバイス
    device: Union[str, torch.device]

    @classmethod
    def from_base(
        cls,
        base: ResidualConfig,
        device: Union[str, torch.device],
        context_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_input_tokens: Optional[int] = None,
        phase2_learning_rate: Optional[float] = None,
        phase2_epochs: Optional[int] = None,
    ) -> "Phase2Config":
        """ResidualConfigから生成（オプションで上書き可能）"""
        return cls(
            context_dim=context_dim if context_dim is not None else base.context_dim,
            embed_dim=base.embed_dim,
            num_layers=num_layers if num_layers is not None else base.num_layers,
            num_input_tokens=num_input_tokens if num_input_tokens is not None else base.num_input_tokens,
            phase2_learning_rate=phase2_learning_rate if phase2_learning_rate is not None else base.phase2_learning_rate,
            phase2_epochs=phase2_epochs if phase2_epochs is not None else base.phase2_epochs,
            phase2_patience=base.phase2_patience,
            phase2_batch_size=base.phase2_batch_size,
            phase2_gradient_clip=base.phase2_gradient_clip,
            phase2_freeze_embedding=base.phase2_freeze_embedding,
            phase2_memory_safety_factor=base.phase2_memory_safety_factor,
            phase2_min_batch_size=base.phase2_min_batch_size,
            phase2_max_batch_size=base.phase2_max_batch_size,
            device=device,
        )
