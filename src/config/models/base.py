"""
Base Model Configuration

全モデル共通の設定を定義。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.models import TransformerLM


@dataclass
class BaseModelConfig:
    """モデル設定の基底クラス

    全モデル共通の設定を定義。
    """

    vocab_size: int = 52000

    def create_model(self) -> "TransformerLM":
        """設定からモデルを構築

        Returns:
            TransformerLM instance
        """
        raise NotImplementedError("Subclasses must implement create_model()")
