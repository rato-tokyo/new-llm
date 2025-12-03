"""
FFN (Feed-Forward Network) モジュール

シンプルな1層FFN: Linear → GELU
"""

import torch
import torch.nn as nn


class FFN(nn.Module):
    """
    シンプルなFFN: Linear → GELU

    Args:
        input_dim: 入力次元
        output_dim: 出力次元
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fnn(x)
