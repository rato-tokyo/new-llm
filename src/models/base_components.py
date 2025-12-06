"""
Base Components for Transformer Models

共通のビルディングブロック:
- PythiaMLP (Feed-Forward Network)
- 重み初期化
- パラメータカウント
"""

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
