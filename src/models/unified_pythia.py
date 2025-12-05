"""
Unified Pythia Model

位置エンコーディングを設定で切り替え可能な統一Pythiaモデル。
RoPE, ALiBi, NoPEを簡単に入れ替え可能。

Usage:
    from src.models.unified_pythia import UnifiedPythiaModel
    from src.models.position_encoding import PositionEncodingConfig

    # RoPE (default Pythia)
    model = UnifiedPythiaModel(
        pos_encoding=PositionEncodingConfig(type="rope", rotary_pct=0.25)
    )

    # ALiBi
    model = UnifiedPythiaModel(
        pos_encoding=PositionEncodingConfig(type="alibi", alibi_slope=0.0625)
    )

    # No position encoding
    model = UnifiedPythiaModel(
        pos_encoding=PositionEncodingConfig(type="none")
    )
"""

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.position_encoding import (
    PositionEncodingConfig,
    PositionEncoding,
    create_position_encoding,
)


class UnifiedAttention(nn.Module):
    """
    Unified Multi-Head Attention

    位置エンコーディングをモジュールとして注入可能。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        pos_encoding: PositionEncoding,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.pos_encoding = pos_encoding

        # Query, Key, Value projections
        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)

        # Output projection
        self.dense = nn.Linear(hidden_size, hidden_size)

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Apply position encoding to Q/K (if applicable)
        query, key = self.pos_encoding.apply_to_qk(query, key, seq_len)

        # Attention scores
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        # Apply position bias to scores (includes causal mask)
        attn_weights = self.pos_encoding.apply_to_scores(attn_weights, seq_len)

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Attention output
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        output = self.dense(attn_output)
        return output


class UnifiedMLP(nn.Module):
    """Feed-Forward Network"""

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


class UnifiedLayer(nn.Module):
    """
    Unified Transformer Layer

    Parallel Attention + MLP (Pythia style)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        pos_encoding: PositionEncoding,
    ):
        super().__init__()

        # Layer norms
        self.input_layernorm = nn.LayerNorm(hidden_size)

        # Attention
        self.attention = UnifiedAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_encoding=pos_encoding,
        )

        # MLP
        self.mlp = UnifiedMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LayerNorm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Parallel attention and MLP
        attn_output = self.attention(hidden_states, attention_mask)
        mlp_output = self.mlp(hidden_states)

        # Combine and add residual
        hidden_states = residual + attn_output + mlp_output

        return hidden_states


class UnifiedPythiaModel(nn.Module):
    """
    Unified Pythia Language Model

    位置エンコーディングを設定で切り替え可能。

    Architecture:
    - Embedding: vocab_size -> hidden_size
    - N Transformer layers with configurable position encoding
    - Final LayerNorm
    - LM Head: hidden_size -> vocab_size
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        pos_encoding: Optional[PositionEncodingConfig] = None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Default to RoPE if not specified
        if pos_encoding is None:
            pos_encoding = PositionEncodingConfig(type="rope")

        self.pos_encoding_config = pos_encoding

        # Create position encoding module
        pos_enc_module = create_position_encoding(pos_encoding, self.head_dim)

        # Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # Transformer layers (share the same pos_encoding config)
        self.layers = nn.ModuleList([
            UnifiedLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                pos_encoding=pos_enc_module,
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

        # Transformer layers
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

        return {
            "total": total,
            "embedding": embedding,
            "lm_head": lm_head,
            "transformer": total - embedding - lm_head,
        }

    @property
    def position_encoding_type(self) -> str:
        """現在の位置エンコーディングタイプを返す"""
        return self.pos_encoding_config.type
