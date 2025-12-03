"""
Context-Pythia: Pythia with Context-based KV Compression

全LayerのAttention入力をcontext (256-dim) に置き換え、
KVキャッシュを50%削減する。

Architecture:
1. Token Embedding (512-dim)
2. ContextBlock: token_embed → context (256-dim)
3. 6 Layers: 全て context を入力として使用
4. Output Head: 512-dim → vocab_size

Training:
- Phase 1: ContextBlock のみ学習 (OACD)
- Phase 2: 全体をファインチューニング
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pythia import RotaryEmbedding, apply_rotary_pos_emb


class ContextBlock(nn.Module):
    """
    Context Block: token embeddings を compressed context に変換

    入力: prev_context (context_dim) + token_embed (embed_dim)
    出力: context (context_dim)
    """

    def __init__(
        self,
        context_dim: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.embed_dim = embed_dim

        # FFN: [context + token_embed] → context
        input_dim = context_dim + embed_dim
        hidden_dim = input_dim * 2

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, context_dim),
        )
        self.norm = nn.LayerNorm(context_dim)

    def forward(
        self,
        prev_context: torch.Tensor,
        token_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prev_context: [batch, seq, context_dim] or [batch, context_dim]
            token_embeds: [batch, seq, embed_dim] or [batch, embed_dim]

        Returns:
            context: [batch, seq, context_dim] or [batch, context_dim]
        """
        # Concatenate
        combined = torch.cat([prev_context, token_embeds], dim=-1)

        # FFN
        delta = self.ffn(combined)

        # Residual + Norm
        context = self.norm(prev_context + delta)

        return context


class ContextPythiaAttention(nn.Module):
    """
    Context-based Pythia Attention

    入力: context (context_dim) - token_embed (hidden_size) ではなく
    KV Cache: context_dim × seq_len (50%削減)
    """

    def __init__(
        self,
        context_dim: int,
        hidden_size: int,
        num_heads: int,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_ndims = int(self.head_dim * rotary_pct)

        # QKV from context (not hidden_size)
        # Q: context_dim → hidden_size
        # K: context_dim → hidden_size
        # V: context_dim → hidden_size
        self.query_key_value = nn.Linear(context_dim, 3 * hidden_size)

        # Output projection
        self.dense = nn.Linear(hidden_size, hidden_size)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(
            self.rotary_ndims,
            max_position_embeddings=max_position_embeddings,
        )

    def forward(
        self,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            context: [batch, seq, context_dim]

        Returns:
            output: [batch, seq, hidden_size]
        """
        batch_size, seq_len, _ = context.shape

        # Fused QKV from context
        qkv = self.query_key_value(context)
        qkv = qkv.view(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [batch, heads, seq, 3*head_dim]

        q, k, v = qkv.chunk(3, dim=-1)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(context, seq_len)

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
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=context.device),
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


class ContextPythiaLayer(nn.Module):
    """
    Context-Pythia Layer

    入力: context (context_dim)
    出力: hidden_states (hidden_size)
    """

    def __init__(
        self,
        context_dim: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.hidden_size = hidden_size

        # Input layer norm (on context)
        self.input_layernorm = nn.LayerNorm(context_dim, eps=layer_norm_eps)

        # Attention
        self.attention = ContextPythiaAttention(
            context_dim=context_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            rotary_pct=rotary_pct,
            max_position_embeddings=max_position_embeddings,
        )

        # MLP (from context, output hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )

        # Projection for residual (context_dim → hidden_size)
        self.residual_proj = nn.Linear(context_dim, hidden_size)

    def forward(
        self,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            context: [batch, seq, context_dim]

        Returns:
            hidden_states: [batch, seq, hidden_size]
        """
        # Pre-norm
        ln_out = self.input_layernorm(context)

        # Attention
        attn_output = self.attention(ln_out, attention_mask)

        # MLP (parallel)
        mlp_output = self.mlp(ln_out)

        # Residual (project context to hidden_size)
        residual = self.residual_proj(context)

        # Combine
        hidden_states = residual + attn_output + mlp_output

        return hidden_states


class ContextPythiaModel(nn.Module):
    """
    Context-Pythia Model

    全LayerでcontextをKVとして使用し、KVキャッシュを50%削減。

    Architecture:
    1. Token Embedding (hidden_size=512)
    2. ContextBlock: 512 → context_dim (256)
    3. 6 Layers: context を入力、hidden_size を出力
    4. Final Layer Norm
    5. LM Head: hidden_size → vocab_size
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 512,
        context_dim: int = 256,
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
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        # Token Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # ContextBlock: token_embed → context
        self.context_block = ContextBlock(
            context_dim=context_dim,
            embed_dim=hidden_size,
        )

        # Transformer Layers (all use context as input)
        self.layers = nn.ModuleList([
            ContextPythiaLayer(
                context_dim=context_dim,
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

        # LM head
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
        prev_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            prev_context: Optional previous context for sequential processing

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embeds = self.embed_in(input_ids)  # [batch, seq, hidden_size]

        # Initialize context if not provided
        if prev_context is None:
            prev_context = torch.zeros(
                batch_size, seq_len, self.context_dim,
                device=input_ids.device, dtype=token_embeds.dtype
            )

        # Generate context (shifted for causal)
        # For training: use shifted context
        context = self._generate_context_causal(token_embeds, prev_context)

        # Transformer layers (all use context)
        # Sum outputs from all layers
        hidden_states = torch.zeros(
            batch_size, seq_len, self.hidden_size,
            device=input_ids.device, dtype=token_embeds.dtype
        )

        for layer in self.layers:
            layer_output = layer(context, attention_mask)
            hidden_states = hidden_states + layer_output

        # Average over layers
        hidden_states = hidden_states / self.num_layers

        # Final norm
        hidden_states = self.final_layer_norm(hidden_states)

        # LM head
        logits = self.embed_out(hidden_states)

        return logits

    def _generate_context_causal(
        self,
        token_embeds: torch.Tensor,
        prev_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate context causally (position i uses context from position i-1).

        For parallel training, use shifted context.
        """
        batch_size, seq_len, _ = token_embeds.shape

        # Shift prev_context: position i gets context from position i-1
        # Position 0 gets zero context
        zero_context = torch.zeros(
            batch_size, 1, self.context_dim,
            device=token_embeds.device, dtype=token_embeds.dtype
        )

        if seq_len > 1:
            shifted_context = torch.cat([zero_context, prev_context[:, :-1, :]], dim=1)
        else:
            shifted_context = zero_context

        # Generate new context
        context = self.context_block(shifted_context, token_embeds)

        return context

    def forward_with_context_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prev_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with context output (for Phase 1 training).

        Returns:
            logits: [batch, seq_len, vocab_size]
            context: [batch, seq_len, context_dim]
        """
        batch_size, seq_len = input_ids.shape

        # Token embeddings
        token_embeds = self.embed_in(input_ids)

        # Initialize context if not provided
        if prev_context is None:
            prev_context = torch.zeros(
                batch_size, seq_len, self.context_dim,
                device=input_ids.device, dtype=token_embeds.dtype
            )

        # Generate context
        context = self._generate_context_causal(token_embeds, prev_context)

        # Transformer layers
        hidden_states = torch.zeros(
            batch_size, seq_len, self.hidden_size,
            device=input_ids.device, dtype=token_embeds.dtype
        )

        for layer in self.layers:
            layer_output = layer(context, attention_mask)
            hidden_states = hidden_states + layer_output

        hidden_states = hidden_states / self.num_layers
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.embed_out(hidden_states)

        return logits, context

    def freeze_context_block(self) -> None:
        """Freeze ContextBlock for Phase 2 training."""
        for param in self.context_block.parameters():
            param.requires_grad = False
        print("✓ ContextBlock frozen")

    def num_parameters(self) -> dict:
        """Count parameters by component."""
        embed_in = self.embed_in.weight.numel()
        embed_out = self.embed_out.weight.numel()
        context_block = sum(p.numel() for p in self.context_block.parameters())

        layer_params = 0
        for layer in self.layers:
            layer_params += sum(p.numel() for p in layer.parameters())

        final_norm = sum(p.numel() for p in self.final_layer_norm.parameters())

        total = sum(p.numel() for p in self.parameters())

        return {
            "embed_in": embed_in,
            "embed_out": embed_out,
            "context_block": context_block,
            "layers": layer_params,
            "final_norm": final_norm,
            "total": total,
        }

    def kv_cache_size_comparison(self, seq_len: int) -> dict:
        """
        Compare KV cache size with original Pythia.

        Returns memory in bytes (assuming float16).
        """
        # Original Pythia: 6 layers × seq_len × hidden_size × 2 (K+V)
        original_kv = self.num_layers * seq_len * self.hidden_size * 2 * 2  # float16

        # Context-Pythia: 6 layers × seq_len × context_dim × 2 (K+V)
        context_kv = self.num_layers * seq_len * self.context_dim * 2 * 2  # float16

        reduction = 1 - (context_kv / original_kv)

        return {
            "original_bytes": original_kv,
            "context_bytes": context_kv,
            "reduction_pct": reduction * 100,
            "original_mb": original_kv / (1024 * 1024),
            "context_mb": context_kv / (1024 * 1024),
        }
