"""
Layer components for New-LLM architecture.

ContextLayer: 文脈処理専用レイヤー

⚠️ このファイルの実装は動作確認済みです。削除・変更しないでください。
特に初期化方法（normal_ std=0.1）は絶対に変更しないこと。
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.ffn import FFN
from src.utils.initialization import init_linear_weights, count_parameters


class ContextLayer(nn.Module):
    """
    Context Layer - 文脈処理専用レイヤー

    入力: [context, token_embeds]
    出力: context_out

    ⚠️ 重要: この実装は動作確認済みです。
    - 初期化: normal_(std=0.1), bias: normal_(std=0.01)
    - 構造: FFN(Linear + GELU) + LayerNorm + 残差接続

    Args:
        context_input_dim: Input context dimension
        context_output_dim: Output context dimension
        token_input_dim: Token input dimension
    """

    def __init__(
        self,
        context_input_dim: int,
        context_output_dim: int,
        token_input_dim: int = 0
    ) -> None:
        super().__init__()

        self.context_input_dim = context_input_dim
        self.context_output_dim = context_output_dim
        self.token_input_dim = token_input_dim

        # FNN: [context (+ token_embeds)] -> context_output_dim
        input_dim = context_input_dim + token_input_dim
        self.fnn = FFN(input_dim, context_output_dim)

        # LayerNorm（必須：数値安定性のため）
        self.context_norm = nn.LayerNorm(context_output_dim)

        # 残差接続用の射影レイヤー（次元が異なる場合のみ）
        self.residual_proj: Optional[nn.Linear] = None
        if context_input_dim != context_output_dim:
            self.residual_proj = nn.Linear(context_input_dim, context_output_dim)

        # ⚠️ 重要: normal_(std=0.1)で初期化（Xavier禁止）
        init_linear_weights(self)

    def forward(self, context: torch.Tensor, token_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass: Update context only

        Args:
            context: Current context [batch, context_input_dim]
            token_embeds: Token embeddings [batch, token_input_dim] (optional)

        Returns:
            new_context: Updated context [batch, context_output_dim]
        """
        # Concatenate inputs
        if token_embeds is not None and self.token_input_dim > 0:
            fnn_input = torch.cat([context, token_embeds], dim=-1)
        else:
            fnn_input = context

        # FNN forward -> delta_context
        delta_context = self.fnn(fnn_input)

        # 残差接続（次元が異なる場合は射影）
        if self.residual_proj is not None:
            residual = self.residual_proj(context)
        else:
            residual = context

        # Residual connection + LayerNorm
        new_context: torch.Tensor = self.context_norm(residual + delta_context)

        return new_context

    def num_params(self) -> int:
        """このレイヤーのパラメータ数を返す"""
        return count_parameters(self)
