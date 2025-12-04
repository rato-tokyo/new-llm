"""
Diverse Projection (DProj) - Rank-Preserving Dimensionality Reduction

高次元から低次元への射影を行いながら、出力ベクトルの多様性（高rank）を保持する。
OACDアルゴリズムで事前学習し、射影後のベクトルが多様性を持つようにする。

⚠️ このファイルの実装は動作確認済みです。
特に初期化方法（normal_ std=0.1）は絶対に変更しないこと。
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.ffn import FFN
from src.utils.initialization import init_linear_weights, count_parameters


class DiverseProjectionLayer(nn.Module):
    """
    Diverse Projection Layer - 多様性を保持する射影レイヤー

    入力: [prev_proj, token_embeds]
    出力: proj_out

    ⚠️ 重要: この実装は動作確認済みです。
    - 初期化: normal_(std=0.1), bias: normal_(std=0.01)
    - 構造: FFN(Linear + GELU) + LayerNorm + 残差接続

    Args:
        proj_input_dim: Input projection dimension
        proj_output_dim: Output projection dimension
        token_input_dim: Token input dimension
    """

    def __init__(
        self,
        proj_input_dim: int,
        proj_output_dim: int,
        token_input_dim: int = 0
    ) -> None:
        super().__init__()

        self.proj_input_dim = proj_input_dim
        self.proj_output_dim = proj_output_dim
        self.token_input_dim = token_input_dim

        # FFN: [prev_proj (+ token_embeds)] -> proj_output_dim
        input_dim = proj_input_dim + token_input_dim
        self.ffn = FFN(input_dim, proj_output_dim)

        # LayerNorm（必須：数値安定性のため）
        self.proj_norm = nn.LayerNorm(proj_output_dim)

        # 残差接続用の射影レイヤー（次元が異なる場合のみ）
        self.residual_proj: Optional[nn.Linear] = None
        if proj_input_dim != proj_output_dim:
            self.residual_proj = nn.Linear(proj_input_dim, proj_output_dim)

        # ⚠️ 重要: normal_(std=0.1)で初期化（Xavier禁止）
        init_linear_weights(self)

    def forward(
        self,
        prev_proj: torch.Tensor,
        token_embeds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            prev_proj: Previous projection [batch, proj_input_dim]
            token_embeds: Token embeddings [batch, token_input_dim] (optional)

        Returns:
            new_proj: New projection [batch, proj_output_dim]
        """
        # Concatenate inputs
        if token_embeds is not None and self.token_input_dim > 0:
            ffn_input = torch.cat([prev_proj, token_embeds], dim=-1)
        else:
            ffn_input = prev_proj

        # FFN forward -> delta
        delta = self.ffn(ffn_input)

        # 残差接続（次元が異なる場合は射影）
        if self.residual_proj is not None:
            residual = self.residual_proj(prev_proj)
        else:
            residual = prev_proj

        # Residual connection + LayerNorm
        new_proj: torch.Tensor = self.proj_norm(residual + delta)

        return new_proj

    def num_params(self) -> int:
        """パラメータ数を返す"""
        return count_parameters(self)


class DiverseProjection(nn.Module):
    """
    Diverse Projection (DProj) - 多様性を保持する次元圧縮

    高次元のtoken embeddingを低次元に射影しながら、
    出力ベクトルの多様性（高rank）を保持する。

    OACDアルゴリズムで事前学習後、メインモデルの学習時はfreeze。

    ⚠️ 重要: この実装は動作確認済みです。
    - 初期化: normal_(std=0.1), bias: normal_(std=0.01)
    - 構造: FFN(Linear + GELU) + LayerNorm + 残差接続

    Args:
        proj_dim: Output projection dimension
        embed_dim: Token embedding dimension
    """

    def __init__(
        self,
        proj_dim: int,
        embed_dim: int,
    ) -> None:
        super().__init__()

        self.proj_dim = proj_dim
        self.embed_dim = embed_dim

        # 単一レイヤー
        self.layer = DiverseProjectionLayer(
            proj_input_dim=proj_dim,
            proj_output_dim=proj_dim,
            token_input_dim=embed_dim,
        )

    def forward(
        self,
        prev_proj: torch.Tensor,
        token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute projection layer

        Args:
            prev_proj: [batch, proj_dim]
            token_embeds: [batch, embed_dim]

        Returns:
            new_proj: [batch, proj_dim]
        """
        return self.layer(prev_proj, token_embeds)

    def forward_batch(
        self,
        prev_proj: torch.Tensor,
        token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Batch forward pass

        Args:
            prev_proj: [batch, proj_dim]
            token_embeds: [batch, embed_dim]

        Returns:
            new_proj: [batch, proj_dim]
        """
        return self.layer(prev_proj, token_embeds)

    def num_params(self) -> int:
        """パラメータ数を返す"""
        return count_parameters(self)


# Backward compatibility aliases (will be removed)
ContextLayer = DiverseProjectionLayer
ContextBlock = DiverseProjection
