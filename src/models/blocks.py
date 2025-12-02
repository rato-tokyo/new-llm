"""
Block components for New-LLM architecture.

ContextBlock: 文脈処理ブロック（Phase 1で学習、Phase 2でfreeze）
TokenBlock: トークン処理ブロック（Phase 2で学習）

G案 Context Mode (2025-12-02採用):
- Phase 2では最終レイヤー出力のみ使用
- キャッシュ形式: [num_tokens, context_dim]（固定、レイヤー数に依存しない）
"""

import torch
import torch.nn as nn

from .layers import ContextLayer, TokenLayer


class ContextBlock(nn.Module):
    """
    Context Block - 文脈処理ブロック（複数レイヤー）

    Phase 1で学習、Phase 2でfreeze

    token継ぎ足し方式（2025-11-29に一本化）:
    - 全レイヤーでtoken入力
    - 次元: context_dim → context_dim（全レイヤー同じ）

    Args:
        num_layers: Number of context layers
        context_dim: Final context vector dimension
        embed_dim: Token embedding dimension (単一トークンの次元)
        num_input_tokens: Number of input tokens (1 = current only, 2+ = with history)
    """

    def __init__(
        self,
        num_layers: int,
        context_dim: int,
        embed_dim: int,
        num_input_tokens: int = 1
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.num_input_tokens = num_input_tokens
        self.context_dim = context_dim
        self.embed_dim = embed_dim

        token_input_dim = embed_dim * num_input_tokens

        # token継ぎ足し方式: 全レイヤーでtoken入力、次元は固定
        self.context_dims = [context_dim] * (num_layers + 1)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                ContextLayer(
                    context_input_dim=context_dim,
                    context_output_dim=context_dim,
                    token_input_dim=token_input_dim,  # 全レイヤーでtoken入力
                )
            )

    def forward(
        self,
        context: torch.Tensor,
        token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute all context layers sequentially

        Args:
            context: [batch, context_dim]
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            context: [batch, context_dim] - 最終レイヤー出力
        """
        for layer in self.layers:
            context = layer(context, token_embeds)
        return context

    def forward_batch(self, contexts: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """
        バッチ並列で最終レイヤー出力を計算（G案用）

        Args:
            contexts: [num_tokens, context_dim] - 入力context
            token_embeds: [num_tokens, embed_dim * num_input_tokens]

        Returns:
            outputs: [num_tokens, context_dim] - 最終レイヤー出力
        """
        current_context = contexts
        for layer in self.layers:
            current_context = layer(current_context, token_embeds)
        return current_context

    def num_params(self) -> int:
        """このブロックのパラメータ数を返す"""
        return sum(p.numel() for p in self.parameters())


class TokenBlock(nn.Module):
    """
    Token Block - トークン処理ブロック（複数レイヤー）

    Phase 2で学習

    G案 Context Mode (2025-12-02採用):
    - 1層目に前トークン時点のcontext (prev_context)
    - 最終層に現在トークン時点のcontext (current_context)
    - 中間層(3層以上の場合)はcontextなし

    token継ぎ足し方式（num_input_tokens=1の場合）:
    - 入力: embed_dim
    - 出力: embed_dim
    - 全レイヤー同じ次元

    複数トークン入力時（num_input_tokens > 1）:
    - 入力: embed_dim * num_input_tokens
    - 出力: embed_dim
    - 各レイヤーで次元を減少

    Args:
        num_layers: Number of token layers (2以上推奨)
        context_dim: Context vector dimension
        embed_dim: Token embedding dimension (最終出力次元)
        num_input_tokens: Number of input tokens (1 = current only, 2+ = with history)
    """

    def __init__(
        self,
        num_layers: int,
        context_dim: int,
        embed_dim: int,
        num_input_tokens: int = 1,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.num_input_tokens = num_input_tokens
        self.embed_dim = embed_dim
        self.context_dim = context_dim

        # 次元計算
        # 入力次元: embed_dim * num_input_tokens
        # 出力次元: embed_dim
        input_token_dim = embed_dim * num_input_tokens
        output_token_dim = embed_dim
        total_reduction = input_token_dim - output_token_dim

        # 各レイヤーの入出力次元を計算
        self.token_dims = []
        for i in range(num_layers + 1):
            # 線形補間: input_dim から output_dim へ
            dim = input_token_dim - (total_reduction * i) // num_layers
            self.token_dims.append(dim)

        # G案: 1層目と最終層のみcontext入力、中間層はcontext_dim=0
        # 1レイヤー: [context_dim]（1層目=最終層）
        # 2レイヤー: [context_dim, context_dim]
        # 3レイヤー以上: [context_dim, 0, ..., 0, context_dim]
        if num_layers == 1:
            self.context_dims_list = [context_dim]
        elif num_layers == 2:
            self.context_dims_list = [context_dim, context_dim]
        else:
            self.context_dims_list = [context_dim] + [0] * (num_layers - 2) + [context_dim]

        # Stack of Token layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TokenLayer(
                    context_dim=self.context_dims_list[i],
                    token_input_dim=self.token_dims[i],
                    token_output_dim=self.token_dims[i + 1],
                )
            )

    def forward(
        self,
        prev_context: torch.Tensor,
        current_context: torch.Tensor,
        token_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Execute all token layers sequentially (G案)

        1層目に前のcontext、最終層に現在のcontextを使用。

        Args:
            prev_context: [batch, context_dim] - 前のトークン時点のcontext
            current_context: [batch, context_dim] - 現在のトークン時点のcontext
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            token: Updated token [batch, embed_dim]
        """
        for i, layer in enumerate(self.layers):
            if i == 0:
                # 1層目: 前のcontext
                token_embeds = layer(prev_context, token_embeds)
            elif i == self.num_layers - 1:
                # 最終層: 現在のcontext
                token_embeds = layer(current_context, token_embeds)
            else:
                # 中間層: contextなし
                token_embeds = layer(None, token_embeds)

        return token_embeds

    def num_params(self) -> int:
        """このブロックのパラメータ数を返す"""
        return sum(p.numel() for p in self.parameters())
