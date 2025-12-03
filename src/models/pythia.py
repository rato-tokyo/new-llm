"""
Pythia-70M アーキテクチャの再現実装

オリジナルPythia-70Mと同等のアーキテクチャをPyTorchで実装。
HuggingFaceのGPTNeoXモデルと互換性のある構造。

Specs (Pythia-70M):
- hidden_size: 512
- num_layers: 6
- num_heads: 8
- intermediate_size: 2048
- vocab_size: 50304
- max_position_embeddings: 2048
- rotary_pct: 0.25
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: int = 10000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build cos/sin cache
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_position_embeddings:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len],
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors."""
    # cos, sin: [seq_len, dim]
    # q, k: [batch, heads, seq_len, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PythiaAttention(nn.Module):
    """
    Pythia/GPTNeoX style attention with rotary embeddings.

    Uses fused QKV projection and rotary position embeddings.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_ndims = int(self.head_dim * rotary_pct)

        # Fused QKV projection
        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            max_position_embeddings=max_position_embeddings,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Fused QKV
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [batch, heads, seq, 3*head_dim]

        q, k, v = qkv.chunk(3, dim=-1)

        # Apply rotary embeddings to rotary_ndims portion
        cos, sin = self.rotary_emb(hidden_states, seq_len)

        # Split into rotary and non-rotary parts
        q_rot, q_pass = q[..., : self.rotary_ndims], q[..., self.rotary_ndims :]
        k_rot, k_pass = k[..., : self.rotary_ndims], k[..., self.rotary_ndims :]

        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)

        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)

        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=hidden_states.device),
                diagonal=1,
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))
        else:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.dense(attn_output)

        return attn_output


class PythiaMLP(nn.Module):
    """Pythia/GPTNeoX MLP with GELU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.dense_h_to_4h = nn.Linear(hidden_size, intermediate_size)
        self.dense_4h_to_h = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class PythiaLayer(nn.Module):
    """
    Single Pythia/GPTNeoX transformer layer.

    Uses parallel attention and MLP (both receive the same input).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.attention = PythiaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            rotary_pct=rotary_pct,
            max_position_embeddings=max_position_embeddings,
        )
        self.mlp = PythiaMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Parallel attention and MLP (GPTNeoX style)
        residual = hidden_states

        # Pre-norm
        ln_out = self.input_layernorm(hidden_states)

        # Attention
        attn_output = self.attention(ln_out, attention_mask)

        # MLP (parallel - uses same ln_out)
        mlp_output = self.mlp(ln_out)

        # Residual
        hidden_states = residual + attn_output + mlp_output

        return hidden_states


class PythiaModel(nn.Module):
    """
    Pythia-70M equivalent model.

    Architecture:
    - Token embeddings (no position embeddings - uses rotary)
    - 6 transformer layers with parallel attention/MLP
    - Final layer norm
    - LM head (output projection)
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 2048,
        rotary_pct: float = 0.25,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Embeddings
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            PythiaLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                rotary_pct=rotary_pct,
                max_position_embeddings=max_position_embeddings,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        # LM head (not tied with embeddings in Pythia)
        self.embed_out = nn.Linear(hidden_size, vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
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
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Token embeddings
        hidden_states = self.embed_in(input_ids)

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final norm
        hidden_states = self.final_layer_norm(hidden_states)

        # LM head
        logits = self.embed_out(hidden_states)

        return logits

    def num_parameters(self) -> dict:
        """Count parameters by component."""
        embed_in = self.embed_in.weight.numel()
        embed_out = self.embed_out.weight.numel()

        layer_params = 0
        for layer in self.layers:
            layer_params += sum(p.numel() for p in layer.parameters())

        final_norm = sum(p.numel() for p in self.final_layer_norm.parameters())

        total = sum(p.numel() for p in self.parameters())

        return {
            "embed_in": embed_in,
            "embed_out": embed_out,
            "layers": layer_params,
            "final_norm": final_norm,
            "total": total,
        }
