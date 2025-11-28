"""
実験設定クラス
"""

from dataclasses import dataclass, field
from typing import Optional

from config import ResidualConfig


@dataclass
class ExperimentConfig:
    """
    単一実験の設定

    Attributes:
        name: 実験名（例: "A", "B"）
        num_layers: レイヤー数
        context_multiplier: context_dim = embed_dim × context_multiplier
        description: 実験の説明（オプション）
        num_input_tokens: 入力トークン数（Noneの場合はbase_configから取得）
    """

    name: str
    num_layers: int
    context_multiplier: int = 1
    description: str = ""
    num_input_tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.description:
            self.description = f"{self.num_layers}層, ×{self.context_multiplier}"

    def get_context_dim(self, embed_dim: int) -> int:
        """context_dimを計算"""
        return embed_dim * self.context_multiplier

    def to_dict(self) -> dict:
        """辞書に変換"""
        return {
            "name": self.name,
            "num_layers": self.num_layers,
            "context_multiplier": self.context_multiplier,
            "description": self.description,
            "num_input_tokens": self.num_input_tokens,
        }


@dataclass
class TrainingConfig:
    """
    訓練パラメータ（config.pyから取得）

    デフォルト値は使用せず、ResidualConfigから取得する。
    """

    embed_dim: int = field(init=False)
    vocab_size: int = field(init=False)
    num_input_tokens: int = field(init=False)

    # Phase 1
    phase1_max_iterations: int = field(init=False)
    phase1_learning_rate: float = field(init=False)
    dist_reg_weight: float = field(init=False)

    # Phase 2
    phase2_epochs: int = field(init=False)
    phase2_batch_size: int = field(init=False)
    phase2_patience: int = field(init=False)
    phase2_freeze_embedding: bool = field(init=False)

    def __post_init__(self) -> None:
        config = ResidualConfig()
        self.embed_dim = config.embed_dim
        self.vocab_size = config.vocab_size
        self.num_input_tokens = config.num_input_tokens
        self.phase1_max_iterations = config.phase1_max_iterations
        self.phase1_learning_rate = config.phase1_learning_rate
        self.dist_reg_weight = config.dist_reg_weight
        self.phase2_epochs = config.phase2_epochs
        self.phase2_batch_size = config.phase2_batch_size
        self.phase2_patience = config.phase2_patience
        self.phase2_freeze_embedding = config.phase2_freeze_embedding

    def to_dict(self) -> dict:
        """辞書に変換"""
        return {
            "embed_dim": self.embed_dim,
            "vocab_size": self.vocab_size,
            "num_input_tokens": self.num_input_tokens,
            "phase1_max_iterations": self.phase1_max_iterations,
            "phase1_learning_rate": self.phase1_learning_rate,
            "dist_reg_weight": self.dist_reg_weight,
            "phase2_epochs": self.phase2_epochs,
            "phase2_batch_size": self.phase2_batch_size,
            "phase2_patience": self.phase2_patience,
            "phase2_freeze_embedding": self.phase2_freeze_embedding,
        }
