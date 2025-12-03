"""
Block components for New-LLM architecture.

ContextBlock: 文脈処理ブロック（Phase 1で学習、Phase 2でfreeze）

⚠️ このファイルの実装は動作確認済みです。削除・変更しないでください。
"""

import torch
import torch.nn as nn

from .layers import ContextLayer
from src.utils.initialization import count_parameters


class ContextBlock(nn.Module):
    """
    Context Block - 文脈処理ブロック（1層固定）

    Phase 1で学習、Phase 2でfreeze

    ⚠️ 重要: この実装は動作確認済みです。
    - 初期化: normal_(std=0.1), bias: normal_(std=0.01)
    - 構造: FFN(Linear + GELU) + LayerNorm + 残差接続

    Args:
        context_dim: Context vector dimension
        embed_dim: Token embedding dimension
    """

    def __init__(
        self,
        context_dim: int,
        embed_dim: int,
    ) -> None:
        super().__init__()

        self.context_dim = context_dim
        self.embed_dim = embed_dim

        # 単一レイヤー
        self.layer = ContextLayer(
            context_input_dim=context_dim,
            context_output_dim=context_dim,
            token_input_dim=embed_dim,
        )

    def forward(
        self,
        context: torch.Tensor,
        token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute context layer

        Args:
            context: [batch, context_dim]
            token_embeds: [batch, embed_dim]

        Returns:
            context: [batch, context_dim]
        """
        return self.layer(context, token_embeds)

    def forward_batch(
        self,
        context: torch.Tensor,
        token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batch forward pass（キャッシュ収集用）

        forward() と同一だが、明示的なバッチ処理用メソッド。

        Args:
            context: [batch, context_dim]
            token_embeds: [batch, embed_dim]

        Returns:
            context: [batch, context_dim]
        """
        return self.layer(context, token_embeds)

    def num_params(self) -> int:
        """パラメータ数を返す"""
        return count_parameters(self)
