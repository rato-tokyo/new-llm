"""
Unified Transformer Language Model

レイヤーリストを受け取る汎用モデル。
レイヤーを組み合わせることで様々なアーキテクチャを実現。

使用例:
    from src.models.model import TransformerLM
    from src.models.layers import InfiniLayer, PythiaLayer

    # Infini-Pythia: 1層目Infini + 5層Pythia
    layers = [
        InfiniLayer(hidden_size=512, num_heads=8, intermediate_size=2048),
        *[PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048) for _ in range(5)]
    ]
    model = TransformerLM(layers=layers)

    # 全層Infini
    layers = [InfiniLayer(...) for _ in range(6)]
    model = TransformerLM(layers=layers)
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.base_components import init_weights
from src.models.layers import BaseLayer


class TransformerLM(nn.Module):
    """
    Unified Transformer Language Model

    レイヤーリストを受け取り、共通の構造で言語モデルを構築。

    構造:
        Token Embedding
            ↓
        Layer 0, 1, ..., N-1 (任意のレイヤータイプ)
            ↓
        Final LayerNorm
            ↓
        LM Head (vocab projection)
    """

    def __init__(
        self,
        layers: list[BaseLayer],
        vocab_size: int = 50304,
        hidden_size: int = 512,
    ):
        """
        Args:
            layers: レイヤーのリスト（任意のBaseLayerサブクラス）
            vocab_size: 語彙サイズ
            hidden_size: 隠れ層次元
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = len(layers)

        # Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList(layers)

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size)

        # LM Head
        self.embed_out = nn.Linear(hidden_size, vocab_size, bias=False)

        # Initialize weights
        self.apply(init_weights)

    def reset_memory(self) -> None:
        """全メモリ系レイヤーのメモリをリセット"""
        device = self.embed_in.weight.device
        for layer in self.layers:
            layer.reset_memory(device)

    def get_memory_state(self) -> dict:
        """
        全レイヤーのメモリ状態を取得

        Returns:
            dict: {layer_idx: memory_state} 形式
        """
        states = {}
        for i, layer in enumerate(self.layers):
            state = layer.get_memory_state()
            if state is not None:
                states[i] = state
        return states

    def set_memory_state(self, states: dict) -> None:
        """
        全レイヤーのメモリ状態を設定

        Args:
            states: get_memory_state()で取得した状態
        """
        device = self.embed_in.weight.device
        for i, state in states.items():
            self.layers[i].set_memory_state(state, device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            update_memory: メモリを更新するか（メモリ系レイヤーのみ）

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        hidden_states = self.embed_in(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                update_memory=update_memory,
            )

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.embed_out(hidden_states)

        return logits

    def num_parameters(self) -> dict[str, int]:
        """Count parameters"""
        total = sum(p.numel() for p in self.parameters())
        embedding = self.embed_in.weight.numel()
        lm_head = self.embed_out.weight.numel()

        layer_params = [
            sum(p.numel() for p in layer.parameters())
            for layer in self.layers
        ]

        return {
            "total": total,
            "embedding": embedding,
            "lm_head": lm_head,
            "transformer": sum(layer_params),
            "per_layer": layer_params,
        }
