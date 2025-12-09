"""
Experimental Model Configurations

実験対象モデル（Infini、MultiMemory）の構造設定を一元管理。

Usage:
    from src.config import InfiniConfig, MultiMemoryConfig
    from src.models import create_model

    # Infiniモデル
    config = InfiniConfig()
    model = create_model("infini", model_config=config)

    # MultiMemoryモデル（カスタム設定）
    config = MultiMemoryConfig(num_memories=8, use_delta_rule=False)
    model = create_model("multi_memory", model_config=config)
"""

from dataclasses import dataclass
from typing import Literal, Optional, Union


# ========== 基本設定（Pythia-70M準拠）==========
# PythiaConfigから継承される値（参照用）
# hidden_size = 512
# num_attention_heads = 8
# intermediate_size = 2048
# num_layers = 6
# vocab_size = 50304
# max_position_embeddings = 2048
# rotary_pct = 0.25


@dataclass
class InfiniConfig:
    """Infini-Attention モデル設定

    Infini-Attention: 長期メモリを圧縮して格納するアーキテクチャ。
    1層目がInfiniLayer、残りがPythiaLayer。

    Attributes:
        num_memory_banks: メモリバンク数（デフォルト: 1）
        segments_per_bank: バンクあたりのセグメント数（デフォルト: 4）
        use_delta_rule: Delta Rule更新を使用（デフォルト: True）
            - True: 差分のみを書き込む（上書き防止）
            - False: 単純加算（高速だが情報が混ざる）
    """

    # メモリバンク設定
    num_memory_banks: int = 1
    segments_per_bank: int = 4

    # メモリ更新方式
    use_delta_rule: bool = True


@dataclass
class MultiMemoryConfig:
    """Multi-Memory Attention モデル設定

    複数の独立したメモリを持ち、クエリに応じて動的に選択するアーキテクチャ。
    memory_norm方式（Landmark = Σσ(k)）でメモリを選択。

    Attributes:
        num_memories: メモリ数（デフォルト: 4）
            - 各メモリは独立した知識ドメインを担当可能
            - Detail Memory として機能
        use_delta_rule: Delta Rule更新を使用（デフォルト: True）
        top_k: 選択するメモリ数（デフォルト: 2）
            - 関連度の高いTop-Kメモリを混合
    """

    # メモリ数
    num_memories: int = 4

    # メモリ更新方式
    use_delta_rule: bool = True

    # メモリ選択設定
    top_k: int = 2


# Type alias for model config types
ModelConfigType = Optional[Union[InfiniConfig, MultiMemoryConfig]]
ModelTypeLiteral = Literal["pythia", "infini", "multi_memory"]
