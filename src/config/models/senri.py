"""
Senri Model Configuration

Senri: Japanese LLM with Compressive Memory
OpenCALMトークナイザーを使用した日本語LLM。
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..constants import OPEN_CALM_VOCAB_SIZE, OPEN_CALM_TOKENIZER
from ..layers import LayerConfigType, default_senri_layers

if TYPE_CHECKING:
    from src.models import TransformerLM


@dataclass
class SenriModelConfig:
    """Senriモデル設定

    Infini-Attention + Multi-Memory を組み合わせたモデル。
    OpenCALMトークナイザーを使用した日本語LLM。

    Examples:
        from src.config import SenriModelConfig, default_senri_layers

        # デフォルト構成（1 Senri + 5 Pythia）
        config = SenriModelConfig()
        model = config.create_model()

        # カスタム構成
        config = SenriModelConfig(
            layers=default_senri_layers(
                num_senri=2,
                num_pythia=4,
                use_multi_memory=True,
                num_memories=8,
            )
        )
        model = config.create_model()
    """

    vocab_size: int = OPEN_CALM_VOCAB_SIZE
    tokenizer_name: str = OPEN_CALM_TOKENIZER
    layers: list[LayerConfigType] = field(default_factory=default_senri_layers)

    def create_model(self) -> "TransformerLM":
        """設定からモデルを構築

        Returns:
            TransformerLM instance
        """
        from src.models import create_model

        return create_model(self.layers, vocab_size=self.vocab_size)
