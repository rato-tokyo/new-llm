"""
FFN (Feed-Forward Network) モジュール

疎結合設計により、将来的にSwiGLU等への拡張が容易。

使い方:
    from src.models.ffn import create_ffn

    ffn = create_ffn(
        fnn_type="standard",
        input_dim=768,
        output_dim=768,
        expand_factor=4,
        num_layers=2,
        activation="gelu"
    )
"""

from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn


class BaseFFN(ABC, nn.Module):
    """
    FFN抽象基底クラス

    将来のSwiGLU等への拡張ポイント。
    すべてのFFN実装はこのクラスを継承する。
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播

        Args:
            x: 入力テンソル [batch, input_dim]

        Returns:
            出力テンソル [batch, output_dim]
        """
        pass


class StandardFFN(BaseFFN):
    """
    標準FFN: Linear → Activation (→ Linear → Activation)

    設定例:
        - num_layers=1, expand_factor=1: 現状維持（Linear → ReLU）
        - num_layers=2, expand_factor=4: Transformer標準（expand → contract）
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        expand_factor: int = 1,
        num_layers: int = 1,
        activation: str = "relu"
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.expand_factor = expand_factor
        self.num_layers = num_layers
        self.activation = activation

        # 活性化関数の選択
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}. Use 'relu' or 'gelu'.")

        # FFN構築
        if num_layers == 1:
            # 1層: Linear → Activation（現状維持）
            self.fnn = nn.Sequential(
                nn.Linear(input_dim, output_dim),
                act_fn()
            )
        elif num_layers == 2:
            # 2層: expand → contract（Transformer標準）
            hidden_dim = input_dim * expand_factor
            self.fnn = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                act_fn(),
                nn.Linear(hidden_dim, output_dim),
                act_fn()
            )
        else:
            # N層: 一般化
            hidden_dim = input_dim * expand_factor
            layers: List[nn.Module] = []
            dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
            for i in range(num_layers):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(act_fn())
            self.fnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fnn(x)


def create_ffn(
    fnn_type: str,
    input_dim: int,
    output_dim: int,
    expand_factor: int = 1,
    num_layers: int = 1,
    activation: str = "relu"
) -> BaseFFN:
    """
    FFNファクトリ関数

    設定に基づいてFFNインスタンスを生成する。
    将来的にSwiGLU等を追加する場合はここに分岐を追加。

    Args:
        fnn_type: FFNタイプ ("standard", 将来 "swiglu")
        input_dim: 入力次元
        output_dim: 出力次元
        expand_factor: 中間層の拡張率 (1=拡張なし, 4=Transformer標準)
        num_layers: FFN内の層数 (1=現状, 2=Transformer標準)
        activation: 活性化関数 ("relu", "gelu")

    Returns:
        FFNインスタンス
    """
    if fnn_type == "standard":
        return StandardFFN(
            input_dim=input_dim,
            output_dim=output_dim,
            expand_factor=expand_factor,
            num_layers=num_layers,
            activation=activation
        )
    elif fnn_type == "swiglu":
        # 将来の拡張ポイント
        raise NotImplementedError(
            "SwiGLU is not yet implemented. "
            "To add SwiGLU, create a SwiGLU class inheriting from BaseFFN."
        )
    else:
        raise ValueError(f"Unknown FFN type: {fnn_type}. Use 'standard'.")
