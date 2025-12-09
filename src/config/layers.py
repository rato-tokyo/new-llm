"""
Layer Configuration

レイヤーごとの設定をdataclassで定義。
これらのConfigリストからモデルを構築する。

Usage:
    from src.config import SenriLayerConfig, PythiaLayerConfig
    from src.models import create_model

    layers = [
        SenriLayerConfig(num_memories=4, num_memory_banks=2),
        PythiaLayerConfig(),
        PythiaLayerConfig(),
    ]
    model = create_model(layers, vocab_size=52000)
"""

from dataclasses import dataclass
from typing import Union


@dataclass
class BaseLayerConfig:
    """レイヤー設定の基底クラス

    全レイヤー共通の設定を定義。
    """
    hidden_size: int = 512
    num_attention_heads: int = 8
    intermediate_size: int = 2048


@dataclass
class PythiaLayerConfig(BaseLayerConfig):
    """Pythiaレイヤー設定

    標準的なTransformerレイヤー（RoPE + Softmax Attention）。

    Attributes:
        rotary_pct: Rotary Embeddingの割合（0.0でNoPE）
        max_position_embeddings: 最大シーケンス長
    """
    rotary_pct: float = 0.25
    max_position_embeddings: int = 2048


@dataclass
class SenriLayerConfig(BaseLayerConfig):
    """Senriレイヤー設定

    Infini-Attention + Multi-Memory を組み合わせた特殊レイヤー。
    圧縮メモリによる長期記憶とLinear Attentionを持つ。

    Attributes:
        num_memory_banks: Infiniメモリバンク数
        segments_per_bank: バンクあたりのセグメント数
        num_memories: MultiMemoryのメモリ数（Detail Memory）
        use_delta_rule: Delta Rule更新を使用
        use_multi_memory: MultiMemoryを使用するか（FalseならInfiniのみ）
    """
    # Infini-Attention設定
    num_memory_banks: int = 1
    segments_per_bank: int = 4

    # Multi-Memory設定
    num_memories: int = 4
    use_multi_memory: bool = False  # デフォルトはInfiniのみ

    # 共通メモリ設定
    use_delta_rule: bool = True


# Type alias
LayerConfigType = Union[PythiaLayerConfig, SenriLayerConfig]


def default_senri_layers(
    num_senri: int = 1,
    num_pythia: int = 5,
    **senri_kwargs,
) -> list[LayerConfigType]:
    """デフォルトのSenriレイヤー構成を生成

    Args:
        num_senri: Senriレイヤー数（先頭に配置）
        num_pythia: Pythiaレイヤー数
        **senri_kwargs: SenriLayerConfigに渡す追加引数

    Returns:
        レイヤー設定のリスト

    Examples:
        # デフォルト: 1 Senri + 5 Pythia
        layers = default_senri_layers()

        # カスタム: 2 Senri + 4 Pythia
        layers = default_senri_layers(num_senri=2, num_pythia=4)

        # MultiMemory有効化
        layers = default_senri_layers(use_multi_memory=True, num_memories=8)
    """
    return [
        SenriLayerConfig(**senri_kwargs) for _ in range(num_senri)
    ] + [
        PythiaLayerConfig() for _ in range(num_pythia)
    ]


def default_pythia_layers(num_layers: int = 6) -> list[LayerConfigType]:
    """Pythiaのみのレイヤー構成を生成（ベースライン用）

    Args:
        num_layers: レイヤー数

    Returns:
        PythiaLayerConfigのリスト
    """
    return [PythiaLayerConfig() for _ in range(num_layers)]
