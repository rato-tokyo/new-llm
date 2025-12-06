"""
Base Components for Transformer Models

共通のビルディングブロックを提供:
- Embedding層
- MLP層
- 重み初期化
- Transformer Layer基底クラス
"""

from typing import Optional

import torch
import torch.nn as nn


class PythiaMLP(nn.Module):
    """
    Pythia-style MLP (Feed-Forward Network)

    構造: Linear -> GELU -> Linear
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(hidden_size, intermediate_size)
        self.dense_4h_to_h = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class EmbeddingLayer(nn.Module):
    """
    Token Embedding Layer

    入力トークンIDを隠れ表現に変換。
    """

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)

    @property
    def weight(self) -> torch.Tensor:
        return self.embed.weight


class OutputHead(nn.Module):
    """
    Language Model Output Head

    隠れ表現をlogitsに変換。
    """

    def __init__(self, hidden_size: int, vocab_size: int, tie_weights: bool = False):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)
        self.tie_weights = tie_weights

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.layer_norm(hidden_states)
        logits = self.linear(hidden_states)
        return logits

    def tie_weights_with_embedding(self, embedding_weight: torch.Tensor) -> None:
        """埋め込み層と重みを共有"""
        if self.tie_weights:
            self.linear.weight = embedding_weight


class BaseTransformerLayer(nn.Module):
    """
    Base Transformer Layer (Pythia-style: Pre-LayerNorm + Parallel Attention/MLP)

    構造:
        x' = x + Attention(LayerNorm(x)) + MLP(LayerNorm(x))

    サブクラスで使用:
        attention = MyAttention(...)
        layer = BaseTransformerLayer(hidden_size, intermediate_size, attention)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        attention: nn.Module,
    ):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.attention = attention
        self.mlp = PythiaMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Parallel Attention and MLP (Pythia style)
        attn_output = self.attention(hidden_states, attention_mask=attention_mask, **kwargs)
        mlp_output = self.mlp(hidden_states)

        return residual + attn_output + mlp_output


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    """
    標準的な重み初期化

    Args:
        module: 初期化するモジュール
        std: 正規分布の標準偏差
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module) -> int:
    """モデルの総パラメータ数を返す"""
    return sum(p.numel() for p in model.parameters())
