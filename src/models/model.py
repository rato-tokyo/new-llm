"""
Unified Transformer Language Model

レイヤーリストを受け取る汎用モデル。
レイヤーを組み合わせることで様々なアーキテクチャを実現。

使用例:
    from src.models.model import TransformerLM
    from src.models.layers import SenriLayer, PythiaLayer

    # Senri-Pythia: 1層目Senri + 5層Pythia
    layers = [
        SenriLayer(hidden_size=512, num_heads=8, intermediate_size=2048),
        *[PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048) for _ in range(5)]
    ]
    model = TransformerLM(layers=layers)

    # 複数メモリ構成
    layers = [SenriLayer(num_memories=4, ...), ...]
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

    def reset_memory(self, keep_frozen: bool = True) -> None:
        """
        全メモリ系レイヤーのメモリをリセット

        Args:
            keep_frozen: If True, only reset unfrozen banks
        """
        device = self.embed_in.weight.device
        for layer in self.layers:
            if hasattr(layer, 'reset_memory'):
                layer.reset_memory(device, keep_frozen)

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

    # =========================================================================
    # Freeze / Unfreeze Methods
    # =========================================================================

    def freeze_memory(
        self,
        memory_indices: Optional[list[int]] = None,
        layer_indices: Optional[list[int]] = None,
    ) -> None:
        """
        Freeze memories in specified layers.

        Args:
            memory_indices: Memory indices to freeze. If None, freeze all.
            layer_indices: Layer indices. If None, apply to all memory layers.
        """
        for i, layer in enumerate(self.layers):
            if layer_indices is not None and i not in layer_indices:
                continue
            if hasattr(layer, 'freeze_memory'):
                layer.freeze_memory(memory_indices)

    def unfreeze_memory(
        self,
        memory_indices: Optional[list[int]] = None,
        layer_indices: Optional[list[int]] = None,
    ) -> None:
        """
        Unfreeze memories in specified layers.

        Args:
            memory_indices: Memory indices to unfreeze. If None, unfreeze all.
            layer_indices: Layer indices. If None, apply to all memory layers.
        """
        for i, layer in enumerate(self.layers):
            if layer_indices is not None and i not in layer_indices:
                continue
            if hasattr(layer, 'unfreeze_memory'):
                layer.unfreeze_memory(memory_indices)

    # =========================================================================
    # Export / Import Methods for Memory Sharing
    # =========================================================================

    def export_memory(
        self,
        memory_indices: Optional[list[int]] = None,
        layer_indices: Optional[list[int]] = None,
    ) -> dict:
        """
        Export memory from specified layers for sharing.

        Args:
            memory_indices: Memory indices to export. If None, export all.
            layer_indices: Layer indices. If None, export from all memory layers.

        Returns:
            dict: {layer_idx: memory_data} that can be saved with torch.save()
        """
        exported = {}
        for i, layer in enumerate(self.layers):
            if layer_indices is not None and i not in layer_indices:
                continue
            if hasattr(layer, 'export_memory'):
                exported[i] = layer.export_memory(memory_indices)
        return exported

    def import_memory(
        self,
        memory_data: dict,
        memory_indices: Optional[list[int]] = None,
        freeze: bool = True,
    ) -> None:
        """
        Import memory from another model or saved state.

        Args:
            memory_data: Dictionary from export_memory()
            memory_indices: Target memory indices. If None, use source indices.
            freeze: Whether to freeze imported memories
        """
        for layer_idx, layer_memory in memory_data.items():
            if layer_idx < len(self.layers) and hasattr(self.layers[layer_idx], 'import_memory'):
                self.layers[layer_idx].import_memory(layer_memory, memory_indices, freeze)

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
