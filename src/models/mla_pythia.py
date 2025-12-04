"""
MLA-Pythia: Pythia with Multi-head Latent Attention (ALiBi版)

Pythia-70MをベースにMLAでKVキャッシュを削減。
RoPEの代わりにALiBiを使用し、吸収モードを実現。

アーキテクチャ:
  Token Embedding (512-dim)
         ↓
  MLALayer × 6 (kv_dim=128, ALiBi)
         ↓
  Output Head (512 → vocab)

KVキャッシュ:
  - 標準: K(512) + V(512) = 1024 per layer
  - MLA: c_kv(128) = 128 per layer
  - 削減率: 87.5%
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from src.models.mla import MLALayer


class MLAPythiaModel(nn.Module):
    """
    MLA-Pythia: Pythia with Multi-head Latent Attention

    Features:
    - KV共通圧縮 (hidden_size → kv_dim)
    - ALiBi位置エンコーディング
    - Parallel Attention + MLP
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        kv_dim: int = 128,
        q_compressed: bool = False,
        alibi_slope: float = 0.0625,
    ):
        """
        Args:
            vocab_size: 語彙サイズ
            hidden_size: 隠れ層次元
            num_layers: レイヤー数
            num_heads: アテンションヘッド数
            intermediate_size: FFN中間層次元
            kv_dim: KV圧縮後の次元
            q_compressed: Qも圧縮するか（フルMLAモード）
            alibi_slope: ALiBiスロープ（統一値）
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.kv_dim = kv_dim
        self.q_compressed = q_compressed

        # Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # MLA Layers
        self.layers = nn.ModuleList([
            MLALayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                kv_dim=kv_dim,
                q_compressed=q_compressed,
                alibi_slope=alibi_slope,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size)

        # LM Head
        self.embed_out = nn.Linear(hidden_size, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Embedding
        hidden_states = self.embed_in(input_ids)

        # MLA Layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        # LM Head
        logits = self.embed_out(hidden_states)

        return logits

    def num_parameters(self) -> Dict[str, int]:
        """Count parameters"""
        total = sum(p.numel() for p in self.parameters())
        embedding = self.embed_in.weight.numel()
        lm_head = self.embed_out.weight.numel()

        # MLA compression parameters
        mla_params = 0
        for layer in self.layers:
            mla_params += sum(
                p.numel() for p in layer.attention.parameters()
            )

        return {
            "total": total,
            "embedding": embedding,
            "lm_head": lm_head,
            "mla_attention": mla_params,
            "transformer": total - embedding - lm_head,
        }

    def kv_cache_size(self, seq_len: int, batch_size: int = 1) -> Dict[str, Union[int, float]]:
        """
        Calculate KV cache size

        Args:
            seq_len: Sequence length
            batch_size: Batch size

        Returns:
            Dictionary with cache sizes in bytes (float32)
        """
        # 標準MHA: K + V = 2 * hidden_size per layer
        standard_per_layer = batch_size * seq_len * self.hidden_size * 2 * 4
        standard_total = standard_per_layer * self.num_layers

        # MLA: c_kv のみ per layer
        mla_per_layer = batch_size * seq_len * self.kv_dim * 4
        mla_total = mla_per_layer * self.num_layers

        reduction = (standard_total - mla_total) / standard_total * 100

        return {
            "standard_bytes": standard_total,
            "mla_bytes": mla_total,
            "reduction_percent": reduction,
            "standard_per_layer": standard_per_layer,
            "mla_per_layer": mla_per_layer,
        }
