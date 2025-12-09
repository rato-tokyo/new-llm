"""
Standard Pythia Layer (RoPE + Softmax Attention)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_components import PythiaMLP
from src.models.position_encoding import RotaryEmbedding, apply_rotary_pos_emb
from .base import BaseLayer

# デフォルト値
DEFAULT_HIDDEN_SIZE = 512
DEFAULT_NUM_HEADS = 8
DEFAULT_INTERMEDIATE_SIZE = 2048


class PythiaAttention(nn.Module):
    """Pythia Multi-Head Attention with Rotary Embedding"""

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_heads: int = DEFAULT_NUM_HEADS,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_dim = int(self.head_dim * rotary_pct)

        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

        # NoPE: rotary_dim=0の場合はRoPEを使わない
        self.rotary_emb: Optional[RotaryEmbedding] = None
        if self.rotary_dim > 0:
            self.rotary_emb = RotaryEmbedding(
                self.rotary_dim,
                max_position_embeddings=max_position_embeddings,
            )
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        # RoPE適用（NoPEの場合はスキップ）
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(query, seq_len)
            cos = cos.squeeze(0).squeeze(0).unsqueeze(0).unsqueeze(0)
            sin = sin.squeeze(0).squeeze(0).unsqueeze(0).unsqueeze(0)

            query_rot, key_rot = apply_rotary_pos_emb(
                query[..., :self.rotary_dim],
                key[..., :self.rotary_dim],
                cos, sin
            )
            query = torch.cat([query_rot, query[..., self.rotary_dim:]], dim=-1)
            key = torch.cat([key_rot, key[..., self.rotary_dim:]], dim=-1)

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device) * float("-inf"),
                diagonal=1
            )
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        return self.dense(attn_output)


class PythiaLayer(BaseLayer):
    """
    Standard Pythia Transformer Layer

    Features:
    - RoPE (Rotary Position Embedding)
    - Softmax Attention
    - Parallel Attention + MLP

    Args:
        hidden_size: Hidden dimension (default: 512)
        num_heads: Number of attention heads (default: 8)
        intermediate_size: MLP intermediate dimension (default: 2048)
        rotary_pct: Rotary embedding percentage (default: 0.25)
        max_position_embeddings: Maximum sequence length (default: 2048)

    Example:
        # デフォルト設定
        layer = PythiaLayer()

        # カスタム設定
        layer = PythiaLayer(hidden_size=768, num_heads=12)
    """

    def __init__(
        self,
        hidden_size: int = DEFAULT_HIDDEN_SIZE,
        num_heads: int = DEFAULT_NUM_HEADS,
        intermediate_size: int = DEFAULT_INTERMEDIATE_SIZE,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.attention = PythiaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            rotary_pct=rotary_pct,
            max_position_embeddings=max_position_embeddings,
        )
        self.mlp = PythiaMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.attention(hidden_states, attention_mask)
        mlp_output = self.mlp(hidden_states)
        return residual + attn_output + mlp_output
