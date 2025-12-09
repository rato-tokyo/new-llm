"""
Senri Model Configuration

Senri: Japanese LLM with Compressive Memory
OpenCALMトークナイザーを使用した日本語LLM。
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..constants import OPEN_CALM_VOCAB_SIZE, OPEN_CALM_TOKENIZER
from ..layers import LayerConfigType, default_senri_layers, default_pythia_layers

if TYPE_CHECKING:
    from src.models import TransformerLM


@dataclass
class SenriModelConfig:
    """Senriモデル設定

    Infini-Attention + Multi-Memory を組み合わせたモデル。
    OpenCALMトークナイザーを使用した日本語LLM。

    Examples:
        # デフォルト構成（1 Senri + 5 Pythia）
        config = SenriModelConfig()
        model = config.create_model()

        # Infini-Attention構成
        config = SenriModelConfig.with_infini(num_memory_banks=2)
        model = config.create_model()

        # Multi-Memory構成
        config = SenriModelConfig.with_multi_memory(num_memories=8)
        model = config.create_model()

        # 全層Pythia（ベースライン）
        config = SenriModelConfig.pythia_only(num_layers=6)
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

    @classmethod
    def with_multi_memory(
        cls,
        num_memories: int = 4,
        num_senri: int = 1,
        num_pythia: int = 5,
    ) -> "SenriModelConfig":
        """MultiMemory有効のSenri構成

        Args:
            num_memories: メモリ数
            num_senri: Senriレイヤー数
            num_pythia: Pythiaレイヤー数
        """
        layers = default_senri_layers(
            num_senri=num_senri,
            num_pythia=num_pythia,
            use_multi_memory=True,
            num_memories=num_memories,
        )
        return cls(layers=layers)

    @classmethod
    def with_infini(
        cls,
        num_memory_banks: int = 1,
        segments_per_bank: int = 4,
        num_senri: int = 1,
        num_pythia: int = 5,
    ) -> "SenriModelConfig":
        """Infini-Attention構成

        Args:
            num_memory_banks: メモリバンク数
            segments_per_bank: バンクあたりセグメント数
            num_senri: Senriレイヤー数
            num_pythia: Pythiaレイヤー数
        """
        layers = default_senri_layers(
            num_senri=num_senri,
            num_pythia=num_pythia,
            use_multi_memory=False,
            num_memory_banks=num_memory_banks,
            segments_per_bank=segments_per_bank,
        )
        return cls(layers=layers)

    @classmethod
    def pythia_only(cls, num_layers: int = 6) -> "SenriModelConfig":
        """Pythiaのみの構成（ベースライン）

        Args:
            num_layers: レイヤー数
        """
        return cls(layers=default_pythia_layers(num_layers))
