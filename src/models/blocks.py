"""
Block components for CVFP architecture.

ContextBlock: 文脈処理ブロック（Phase 1で学習、Phase 2でfreeze）
TokenBlock: トークン処理ブロック（Phase 2で学習）
SplitContextBlock: 分割されたContextBlockのコンテナ
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

    def __init__(self, num_layers, context_dim, embed_dim, num_input_tokens=1):
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
                    token_input_dim=token_input_dim  # 全レイヤーでtoken入力
                )
            )

    def forward(self, context, token_embeds, return_intermediates: bool = False):
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

    def forward_with_intermediates(self, context, token_embeds):
        """後方互換性のためのエイリアス"""
        return self.forward(context, token_embeds, return_intermediates=True)

    def num_params(self) -> int:
        """このブロックのパラメータ数を返す"""
        return sum(p.numel() for p in self.parameters())

    def forward_with_intermediates_batch(self, contexts, token_embeds):
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


class SplitContextBlock(nn.Module):
    """
    Split Context Block - 分割されたContextBlockのコンテナ

    N分割されたContextBlockを管理し、推論時に出力を連結する。
    各ブロックは異なるサンプルで訓練され、推論時は全ブロックを実行して
    出力を連結することで、元のcontext_dimと同じ次元の出力を生成。

    効果:
        - 計算量: 約 1/N に削減 (context_dim² → (context_dim/N)² × N)
        - パラメータ: 約 1/N に削減

    Args:
        num_splits: Number of splits
        num_layers: Number of layers per split block
        context_dim: Total context dimension (will be split into context_dim/N per block)
        embed_dim: Token embedding dimension (not split, full size to each block)
        num_input_tokens: Number of input tokens
    """

    def __init__(self, num_splits, num_layers, context_dim, embed_dim, num_input_tokens=1):
        super().__init__()

        self.num_splits = num_splits
        self.num_layers = num_layers
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.num_input_tokens = num_input_tokens

        # 分割後の各ブロックのcontext_dim
        if context_dim % num_splits != 0:
            raise ValueError(
                f"context_dim ({context_dim}) must be divisible by "
                f"num_splits ({num_splits})"
            )
        self.split_context_dim = context_dim // num_splits

        # 各分割ブロックを作成
        self.blocks = nn.ModuleList([
            ContextBlock(
                num_layers=num_layers,
                context_dim=self.split_context_dim,
                embed_dim=embed_dim,
                num_input_tokens=num_input_tokens
            )
            for _ in range(num_splits)
        ])

        # E案用: 各レイヤーの出力次元（結合後）
        # 各ブロックのcontext_dims[1:]を結合
        self.context_dims = self._compute_merged_context_dims()

    def _compute_merged_context_dims(self):
        """各レイヤーの結合後の出力次元を計算"""
        # 全ブロックの最初のブロックから次元情報を取得
        # 各ブロックは同じ構造なので、最初のブロックの次元 × num_splits
        base_dims = self.blocks[0].context_dims
        merged_dims = [dim * self.num_splits for dim in base_dims]
        return merged_dims

    def forward(self, context, token_embeds, split_id=None):
        """
        Forward pass

        Args:
            context: [batch, context_dim] or [batch, split_context_dim] (split_id指定時)
            token_embeds: [batch, embed_dim * num_input_tokens]
            split_id: None = 全ブロック実行して連結（推論用）
                      int = 指定ブロックのみ実行（訓練用）

        Returns:
            context: [batch, context_dim] (split_id=None)
                     [batch, split_context_dim] (split_id指定時)
        """
        if split_id is not None:
            # 訓練: 特定の分割のみ実行
            return self.blocks[split_id](context, token_embeds)
        else:
            # 推論: 全分割を実行して連結
            outputs = []
            for i, block in enumerate(self.blocks):
                start = i * self.split_context_dim
                end = (i + 1) * self.split_context_dim
                split_context = context[:, start:end]
                outputs.append(block(split_context, token_embeds))
            return torch.cat(outputs, dim=-1)

    def forward_with_intermediates(self, context, token_embeds, split_id=None):
        """
        Forward pass with intermediate outputs (E案用)

        Args:
            context: [batch, context_dim] or [batch, split_context_dim]
            token_embeds: [batch, embed_dim * num_input_tokens]
            split_id: None = 全ブロックの出力を連結
                      int = 指定ブロックのみ

        Returns:
            outputs: List of context outputs [context_1, ..., context_N]
                     split_id=None: 各要素は結合された次元
                     split_id指定: 各要素は分割された次元
        """
        if split_id is not None:
            # 訓練: 特定の分割のみ
            return self.blocks[split_id].forward_with_intermediates(context, token_embeds)
        else:
            # 推論: 全分割の出力を連結
            all_intermediates = []
            for i, block in enumerate(self.blocks):
                start = i * self.split_context_dim
                end = (i + 1) * self.split_context_dim
                split_context = context[:, start:end]
                intermediates = block.forward_with_intermediates(split_context, token_embeds)
                all_intermediates.append(intermediates)

            # レイヤーごとに連結
            num_layers = len(all_intermediates[0])
            merged = []
            for layer_idx in range(num_layers):
                layer_outputs = [all_intermediates[s][layer_idx] for s in range(self.num_splits)]
                merged.append(torch.cat(layer_outputs, dim=-1))
            return merged

    def num_params(self) -> int:
        """このブロックのパラメータ数を返す"""
        return sum(p.numel() for p in self.parameters())


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

    def __init__(self, num_layers, context_dim, embed_dim, num_input_tokens=1,
                 context_dims_list=None):
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
                    token_output_dim=self.token_dims[i + 1]
                )
            )

    def forward(self, context, token_embeds, context_list=None):
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

    def forward_with_contexts(self, context_list, token_embeds):
        """後方互換性のためのエイリアス"""
        return self.forward(None, token_embeds, context_list=context_list)

    def num_params(self) -> int:
        """このブロックのパラメータ数を返す"""
        return sum(p.numel() for p in self.parameters())
