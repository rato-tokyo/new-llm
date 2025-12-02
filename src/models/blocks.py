"""
Block components for New-LLM architecture.

ContextBlock: 文脈処理ブロック（Phase 1で学習、Phase 2でfreeze）
TokenBlock: トークン処理ブロック（Phase 2で学習）

1層固定アーキテクチャ（2025-12-02 簡素化）:
- カスケード連結方式により複数レイヤーは不要
- ContextBlock: 1層、TokenBlock: 1層で固定
"""

import torch
import torch.nn as nn

from .layers import ContextLayer, TokenLayer


class ContextBlock(nn.Module):
    """
    Context Block - 文脈処理ブロック（1層固定）

    Phase 1で学習、Phase 2でfreeze

    Args:
        context_dim: Context vector dimension
        embed_dim: Token embedding dimension
        num_input_tokens: Number of input tokens (1 = current only)
    """

    def __init__(
        self,
        context_dim: int,
        embed_dim: int,
        num_input_tokens: int = 1
    ) -> None:
        super().__init__()

        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.num_input_tokens = num_input_tokens

        token_input_dim = embed_dim * num_input_tokens

        # 単一レイヤー
        self.layer = ContextLayer(
            context_input_dim=context_dim,
            context_output_dim=context_dim,
            token_input_dim=token_input_dim,
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
            token_embeds: [batch, embed_dim * num_input_tokens]

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
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            context: [batch, context_dim]
        """
        return self.layer(context, token_embeds)

    def num_params(self) -> int:
        """パラメータ数を返す"""
        return sum(p.numel() for p in self.parameters())


class TokenBlock(nn.Module):
    """
    Token Block - トークン処理ブロック（1層固定）

    Phase 2で学習

    カスケード連結方式（2025-12-02）:
    - 入力: concat(context_a, context_b) = combined_context
    - 1層のみ

    Args:
        context_dim: Context vector dimension (連結後の次元)
        embed_dim: Token embedding dimension
        num_input_tokens: Number of input tokens (1 = current only)
    """

    def __init__(
        self,
        context_dim: int,
        embed_dim: int,
        num_input_tokens: int = 1,
    ) -> None:
        super().__init__()

        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.num_input_tokens = num_input_tokens

        token_input_dim = embed_dim * num_input_tokens

        # 単一レイヤー
        self.layer = TokenLayer(
            context_dim=context_dim,
            token_input_dim=token_input_dim,
            token_output_dim=embed_dim,
        )

    def forward(
        self,
        context: torch.Tensor,
        token_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Execute token layer

        Args:
            context: [batch, context_dim] - 連結されたcontext
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            token: [batch, embed_dim]
        """
        return self.layer(context, token_embeds)

    def num_params(self) -> int:
        """パラメータ数を返す"""
        return sum(p.numel() for p in self.parameters())
