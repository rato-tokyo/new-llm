"""
Block components for New-LLM architecture.

ContextBlock: 文脈処理ブロック（Phase 1で学習、Phase 2でfreeze）
TokenBlock: トークン処理ブロック（Phase 2で学習）
"""

from typing import List, Optional, Union

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
        return_intermediates: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Execute all context layers sequentially

        Args:
            context: [batch, context_dim]
            token_embeds: [batch, embed_dim * num_input_tokens]
            return_intermediates: If True, return all layer outputs (E案用)

        Returns:
            return_intermediates=False: context [batch, context_dim]
            return_intermediates=True: List of outputs [context_1, ..., context_N]
        """
        if return_intermediates:
            outputs = []
            for layer in self.layers:
                context = layer(context, token_embeds)
                outputs.append(context)
            return outputs
        else:
            for layer in self.layers:
                context = layer(context, token_embeds)
            return context

    def forward_with_intermediates(self, context: torch.Tensor, token_embeds: torch.Tensor) -> List[torch.Tensor]:
        """各レイヤーの中間出力を返す"""
        result = self.forward(context, token_embeds, return_intermediates=True)
        assert isinstance(result, list)
        return result

    def num_params(self) -> int:
        """このブロックのパラメータ数を返す"""
        return sum(p.numel() for p in self.parameters())

    def forward_with_intermediates_batch(self, contexts: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """
        バッチ並列で全レイヤーの中間出力を計算

        Phase 1で確定したcontextsを入力として、各レイヤーの出力を並列計算。
        シーケンシャル処理と異なり、全トークンを同時に処理できる。

        Args:
            contexts: [num_tokens, context_dim] - 確定済みのcontext（Phase 1の出力）
            token_embeds: [num_tokens, embed_dim * num_input_tokens]

        Returns:
            outputs: [num_layers, num_tokens, context_dim] - 各レイヤーの出力
        """
        num_tokens = contexts.shape[0]
        device = contexts.device

        # 結果を格納するテンソル
        outputs = torch.zeros(
            self.num_layers, num_tokens, self.context_dim,
            device=device, dtype=contexts.dtype
        )

        # token継ぎ足し方式: 全レイヤーでtoken入力
        current_context = contexts
        for layer_idx, layer in enumerate(self.layers):
            current_context = layer(current_context, token_embeds)
            outputs[layer_idx] = current_context

        return outputs


class TokenBlock(nn.Module):
    """
    Token Block - トークン処理ブロック（複数レイヤー）

    Phase 2で学習

    token継ぎ足し方式（num_input_tokens=1の場合）:
    - 入力: embed_dim
    - 出力: embed_dim
    - 全レイヤー同じ次元

    複数トークン入力時（num_input_tokens > 1）:
    - 入力: embed_dim * num_input_tokens
    - 出力: embed_dim
    - 各レイヤーで次元を減少

    E案対応:
        各レイヤーはContextBlockの対応するレイヤーの出力を参照

    Args:
        num_layers: Number of token layers
        context_dim: Final context vector dimension
        embed_dim: Token embedding dimension (最終出力次元)
        num_input_tokens: Number of input tokens (1 = current only, 2+ = with history)
        context_dims_list: List of context dimensions from ContextBlock (for E案)
    """

    def __init__(
        self,
        num_layers: int,
        context_dim: int,
        embed_dim: int,
        num_input_tokens: int = 1,
        context_dims_list: Optional[List[int]] = None
    ) -> None:
        super().__init__()

        self.num_layers = num_layers
        self.num_input_tokens = num_input_tokens
        self.embed_dim = embed_dim

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

        # ContextBlockからの次元リスト（E案用）
        # context_dims_listはContextBlockのcontext_dims[1:]に相当
        if context_dims_list is None:
            self.context_dims_list = [context_dim] * num_layers
        else:
            self.context_dims_list = context_dims_list

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

    def forward(self, context: Optional[torch.Tensor], token_embeds: torch.Tensor, context_list: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        Execute all token layers sequentially

        Args:
            context: [batch, context_dim] - 単一context（A案用、context_list=Noneの場合）
            token_embeds: [batch, embed_dim * num_input_tokens]
            context_list: List of context outputs（E案用）
                          [context_1, ..., context_N] len == num_layers
                          指定時はcontextは無視される

        Returns:
            token: Updated token [batch, embed_dim]

        Raises:
            ValueError: if context_list provided but len != num_layers
        """
        if context_list is not None:
            # E案: レイヤーごとに異なるcontextを使用
            if len(context_list) != self.num_layers:
                raise ValueError(
                    f"context_list length ({len(context_list)}) must equal "
                    f"num_layers ({self.num_layers})"
                )
            for i, layer in enumerate(self.layers):
                token_embeds = layer(context_list[i], token_embeds)
        else:
            # A案: 全レイヤーで同じcontextを使用
            for layer in self.layers:
                token_embeds = layer(context, token_embeds)

        return token_embeds

    def forward_with_contexts(self, context_list: List[torch.Tensor], token_embeds: torch.Tensor) -> torch.Tensor:
        """各レイヤーが対応するcontextを使用"""
        return self.forward(None, token_embeds, context_list=context_list)

    def num_params(self) -> int:
        """このブロックのパラメータ数を返す"""
        return sum(p.numel() for p in self.parameters())
