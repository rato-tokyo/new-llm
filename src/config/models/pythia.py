"""
Pythia Model Configuration

Pythiaのみで構成されるベースラインモデルの設定。
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..constants import OPEN_CALM_VOCAB_SIZE
from ..layers import LayerConfigType, default_pythia_layers

if TYPE_CHECKING:
    from src.models import TransformerLM


@dataclass
class PythiaModelConfig:
    """Pythiaモデル設定

    全層Pythia（RoPE + Softmax Attention）で構成されるベースラインモデル。

    Examples:
        from src.config import PythiaModelConfig, default_pythia_layers

        # デフォルト6層
        config = PythiaModelConfig()
        model = config.create_model()

        # カスタム層数
        config = PythiaModelConfig(layers=default_pythia_layers(12))
        model = config.create_model()
    """

    vocab_size: int = OPEN_CALM_VOCAB_SIZE
    layers: list[LayerConfigType] = field(default_factory=lambda: default_pythia_layers(6))

    def create_model(self) -> "TransformerLM":
        """設定からモデルを構築"""
        from src.models import create_model

        return create_model(self.layers, vocab_size=self.vocab_size)
