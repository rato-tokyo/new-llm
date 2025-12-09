"""
Senri Model Configuration

Senri: Japanese LLM with Compressive Memory
OpenCALMトークナイザーを使用した日本語LLM。

このモジュールはSenriモデル構築の中心。
- SenriModelConfig: モデル全体の設定
- レイヤー構成のプリセット
- モデル作成の便利関数

Usage:
    from src.config import SenriModelConfig, default_senri_layers
    from src.models import create_model

    # 方法1: SenriModelConfigを使用
    config = SenriModelConfig()
    model = config.create_model()

    # 方法2: LayerConfigリストを使用
    layers = default_senri_layers()
    model = create_model(layers)

    # 方法3: カスタム構成
    layers = [
        SenriLayerConfig(use_multi_memory=True, num_memories=8),
        PythiaLayerConfig(),
        PythiaLayerConfig(),
    ]
    model = create_model(layers)
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .open_calm import OPEN_CALM_VOCAB_SIZE, OPEN_CALM_TOKENIZER
from .layers import (
    SenriLayerConfig,
    PythiaLayerConfig,
    LayerConfigType,
    default_senri_layers,
    default_pythia_layers,
)

if TYPE_CHECKING:
    from src.models import TransformerLM


@dataclass
class SenriModelConfig:
    """Senriモデル全体の設定

    モデル構造とトークナイザー設定を一元管理。
    LayerConfigリストとvocab_sizeをまとめて保持。

    Attributes:
        vocab_size: 語彙サイズ（OpenCALM: 52,000）
        tokenizer_name: トークナイザー名
        layers: レイヤー設定のリスト

    Examples:
        # デフォルト構成（1 Senri + 5 Pythia）
        config = SenriModelConfig()
        model = config.create_model()

        # MultiMemory有効化
        config = SenriModelConfig.with_multi_memory(num_memories=8)
        model = config.create_model()

        # 全層Pythia（ベースライン）
        config = SenriModelConfig.pythia_only(num_layers=6)
        model = config.create_model()

        # カスタム構成
        config = SenriModelConfig(
            layers=[
                SenriLayerConfig(use_multi_memory=True),
                PythiaLayerConfig(),
                PythiaLayerConfig(),
            ]
        )
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


# Re-export for convenience
__all__ = [
    "SenriModelConfig",
    "SenriLayerConfig",
    "PythiaLayerConfig",
    "default_senri_layers",
    "default_pythia_layers",
    "OPEN_CALM_VOCAB_SIZE",
    "OPEN_CALM_TOKENIZER",
]
