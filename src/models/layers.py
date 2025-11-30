"""
Layer components for CVFP architecture.

ContextLayer: 文脈処理専用レイヤー
TokenLayer: トークン処理専用レイヤー
"""

import torch
import torch.nn as nn

from src.utils.initialization import init_linear_weights


class ContextLayer(nn.Module):
    """
    Context Layer - 文脈処理専用レイヤー

    入力: [context, token_embeds]（全レイヤーでtoken継ぎ足し）
    出力: context_out

    Args:
        context_input_dim: Input context dimension
        context_output_dim: Output context dimension
        token_input_dim: Token input dimension
    """

    def __init__(self, context_input_dim, context_output_dim, token_input_dim=0):
        super().__init__()

        self.context_input_dim = context_input_dim
        self.context_output_dim = context_output_dim
        self.token_input_dim = token_input_dim

        # FNN: [context (+ token_embeds)] -> context_output_dim
        input_dim = context_input_dim + token_input_dim
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, context_output_dim),
            nn.ReLU()
        )

        # LayerNorm（必須：数値安定性のため）
        self.context_norm = nn.LayerNorm(context_output_dim)

        # 残差接続用の射影レイヤー（次元が異なる場合のみ）
        if context_input_dim != context_output_dim:
            self.residual_proj = nn.Linear(context_input_dim, context_output_dim)
        else:
            self.residual_proj = None

        init_linear_weights(self)

    def forward(self, context, token_embeds=None):
        """
        Forward pass: Update context only

        Args:
            context: Current context [batch, context_input_dim]
            token_embeds: Token embeddings [batch, token_input_dim] (optional)
                          最初のレイヤーのみ使用、それ以外はNone

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
        new_context = self.context_norm(residual + delta_context)

        return new_context

    def num_params(self) -> int:
        """このレイヤーのパラメータ数を返す"""
        return sum(p.numel() for p in self.parameters())


class TokenLayer(nn.Module):
    """
    Token Layer - トークン処理専用レイヤー

    入力: [context, token_embeds]
    出力: token_out（tokenのみ更新、contextは参照のみ）

    Args:
        context_dim: Context vector dimension
        token_input_dim: Input token dimension
        token_output_dim: Output token dimension
    """

    def __init__(self, context_dim, token_input_dim, token_output_dim):
        super().__init__()

        self.context_dim = context_dim
        self.token_input_dim = token_input_dim
        self.token_output_dim = token_output_dim

        # FNN: [context + token_embeds] -> token_output_dim
        input_dim = context_dim + token_input_dim
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, token_output_dim),
            nn.ReLU()
        )

        # LayerNorm（必須：数値安定性のため）
        self.token_norm = nn.LayerNorm(token_output_dim)

        # 残差接続用の射影レイヤー（次元が異なる場合のみ）
        if token_input_dim != token_output_dim:
            self.residual_proj = nn.Linear(token_input_dim, token_output_dim)
        else:
            self.residual_proj = None

        init_linear_weights(self)

    def forward(self, context, token_embeds):
        """
        Forward pass: Update token only

        Args:
            context: Context vector [batch, context_dim] (参照のみ)
            token_embeds: Token embeddings [batch, token_input_dim]

        Returns:
            new_token: Updated token [batch, token_output_dim]
        """
        # Concatenate inputs
        fnn_input = torch.cat([context, token_embeds], dim=-1)

        # FNN forward -> delta_token
        delta_token = self.fnn(fnn_input)

        # 残差接続（次元が異なる場合は射影）
        if self.residual_proj is not None:
            residual = self.residual_proj(token_embeds)
        else:
            residual = token_embeds

        # Residual connection + LayerNorm
        new_token = self.token_norm(residual + delta_token)

        return new_token

    def num_params(self) -> int:
        """このレイヤーのパラメータ数を返す"""
        return sum(p.numel() for p in self.parameters())
