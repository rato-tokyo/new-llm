"""
KA Cache (Key-Attention Output Cache) for Inference

推論時にKVキャッシュの代わりにKAキャッシュを使用する実装。
学習は不要で、既存のモデルをそのまま使用可能。

KAキャッシュの仕組み:
  - 標準: K, V をキャッシュ
  - KA: K, A (Attention Output) をキャッシュ

推論時の計算:
  Token i の Attention Output A[i] を計算する際:
  - 過去トークン (1 to i-1): キャッシュされた A[1:i-1] を使用
  - 現在トークン (i): V[i] を使用
  - A[i] = weights @ [A[1:i-1], V[i]]
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KACache:
    """KAキャッシュのデータ構造"""
    keys: List[Optional[torch.Tensor]]  # [num_layers] each: [batch, heads, cached_len, head_dim]
    attention_outputs: List[Optional[torch.Tensor]]  # [num_layers] each: [batch, heads, cached_len, head_dim]

    @classmethod
    def empty(cls, num_layers: int) -> "KACache":
        return cls(keys=[None] * num_layers, attention_outputs=[None] * num_layers)

    def get_seq_len(self) -> int:
        if self.keys[0] is None:
            return 0
        return self.keys[0].shape[2]


@dataclass
class KVCache:
    """標準KVキャッシュのデータ構造"""
    keys: List[Optional[torch.Tensor]]  # [num_layers] each: [batch, heads, cached_len, head_dim]
    values: List[Optional[torch.Tensor]]  # [num_layers] each: [batch, heads, cached_len, head_dim]

    @classmethod
    def empty(cls, num_layers: int) -> "KVCache":
        return cls(keys=[None] * num_layers, values=[None] * num_layers)

    def get_seq_len(self) -> int:
        if self.keys[0] is None:
            return 0
        return self.keys[0].shape[2]


class KACacheAttention(nn.Module):
    """
    KAキャッシュを使用するAttention

    推論時にKVキャッシュの代わりにKAキャッシュを使用。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
        base: int = 10000,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_dim = int(self.head_dim * rotary_pct)
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._init_rope_cache(max_position_embeddings)

    def _init_rope_cache(self, max_len: int) -> None:
        t = torch.arange(max_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
        self.max_seq_len_cached = max_len

    def _apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE to query and key"""
        seq_len = int(position_ids.max().item()) + 1
        if seq_len > self.max_seq_len_cached:
            self._init_rope_cache(seq_len)

        cos = self.cos_cached[position_ids].unsqueeze(1)  # [batch, 1, seq, dim]
        sin = self.sin_cached[position_ids].unsqueeze(1)

        # Split rotary and pass-through
        q_rot, q_pass = q[..., :self.rotary_dim], q[..., self.rotary_dim:]
        k_rot, k_pass = k[..., :self.rotary_dim], k[..., self.rotary_dim:]

        # Rotate
        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = q_rot * cos + rotate_half(q_rot) * sin
        k_rot = k_rot * cos + rotate_half(k_rot) * sin

        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

    def forward_with_kv_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """標準KVキャッシュを使用したforward"""
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        query, key = self._apply_rotary(query, key, position_ids)

        # Update KV cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            key = torch.cat([cached_k, key], dim=2)
            value = torch.cat([cached_v, value], dim=2)

        new_kv_cache = (key, value)

        # Attention
        total_len = key.shape[2]
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, total_len, device=hidden_states.device) * float("-inf"),
            diagonal=total_len - seq_len + 1
        )
        attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        # Output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.dense(attn_output)

        return output, new_kv_cache

    def forward_with_ka_cache(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        ka_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """KAキャッシュを使用したforward"""
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Apply RoPE
        query, key = self._apply_rotary(query, key, position_ids)

        # Handle KA cache
        if ka_cache is not None:
            cached_k, cached_a = ka_cache
            # Concatenate keys
            full_key = torch.cat([cached_k, key], dim=2)
            cached_len = cached_k.shape[2]
        else:
            full_key = key
            cached_a = None
            cached_len = 0

        total_len = full_key.shape[2]

        # Attention scores
        attn_weights = torch.matmul(query, full_key.transpose(-1, -2)) * self.scale

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, total_len, device=hidden_states.device) * float("-inf"),
            diagonal=total_len - seq_len + 1
        )
        attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        # KA-Attention: use cached A for past tokens, V for current token
        # attn_output[i] = sum(weights[i,j] * (A[j] if j < cached_len else V[j]))
        if cached_a is not None:
            # Split weights for cached (A) and current (V)
            weights_for_cached = attn_weights[..., :cached_len]  # [batch, heads, seq, cached_len]
            weights_for_current = attn_weights[..., cached_len:]  # [batch, heads, seq, seq_len]

            # Compute attention output
            # Past tokens: use cached Attention Output
            cached_contribution = torch.matmul(weights_for_cached, cached_a)
            # Current tokens: use Value
            current_contribution = torch.matmul(weights_for_current, value)

            attn_output = cached_contribution + current_contribution
        else:
            # No cache, use V directly (first token)
            attn_output = torch.matmul(attn_weights, value)

        # Update KA cache: store K and A (not V)
        # A for current position = attn_output (per-head)
        new_a = attn_output  # [batch, heads, seq_len, head_dim]
        if cached_a is not None:
            new_cached_a = torch.cat([cached_a, new_a], dim=2)
        else:
            new_cached_a = new_a

        new_ka_cache = (full_key, new_cached_a)

        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.dense(attn_output)

        return output, new_ka_cache


class KACachePythiaModel(nn.Module):
    """
    KAキャッシュ推論対応のPythiaモデル

    標準のPythiaModelと同じ構造だが、KVキャッシュとKAキャッシュの両方に対応。
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
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # Attention layers
        self.attentions = nn.ModuleList([
            KACacheAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                rotary_pct=rotary_pct,
                max_position_embeddings=max_position_embeddings,
            )
            for _ in range(num_layers)
        ])

        # MLPs
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, intermediate_size),
                nn.GELU(),
                nn.Linear(intermediate_size, hidden_size),
            )
            for _ in range(num_layers)
        ])

        # Layer norms
        self.input_layernorms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size)

        # LM Head
        self.embed_out = nn.Linear(hidden_size, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
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
        """Standard forward (no cache)"""
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_in(input_ids)

        for i in range(self.num_layers):
            residual = hidden_states
            hidden_states = self.input_layernorms[i](hidden_states)

            # Parallel attention and MLP
            attn_output, _ = self.attentions[i].forward_with_kv_cache(hidden_states, position_ids, None)
            mlp_output = self.mlps[i](hidden_states)

            hidden_states = residual + attn_output + mlp_output

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.embed_out(hidden_states)

        return logits

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        cache: Optional[List] = None,
        use_ka_cache: bool = False,
    ) -> Tuple[torch.Tensor, List]:
        """Forward with cache (KV or KA)"""
        batch_size, seq_len = input_ids.shape

        if cache is None:
            cache = [None] * self.num_layers
            past_len = 0
        else:
            past_len = cache[0][0].shape[2] if cache[0] is not None else 0

        position_ids = torch.arange(
            past_len, past_len + seq_len, device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)

        hidden_states = self.embed_in(input_ids)
        new_cache = []

        for i in range(self.num_layers):
            residual = hidden_states
            hidden_states = self.input_layernorms[i](hidden_states)

            # Use KA or KV cache
            if use_ka_cache:
                attn_output, layer_cache = self.attentions[i].forward_with_ka_cache(
                    hidden_states, position_ids, cache[i]
                )
            else:
                attn_output, layer_cache = self.attentions[i].forward_with_kv_cache(
                    hidden_states, position_ids, cache[i]
                )

            mlp_output = self.mlps[i](hidden_states)
            hidden_states = residual + attn_output + mlp_output

            new_cache.append(layer_cache)

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.embed_out(hidden_states)

        return logits, new_cache

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        use_ka_cache: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate tokens with KV or KA cache"""
        cache = None

        for _ in range(max_new_tokens):
            if cache is None:
                # First forward: process all input tokens
                logits, cache = self.forward_with_cache(input_ids, None, use_ka_cache)
            else:
                # Subsequent forwards: only process last token
                logits, cache = self.forward_with_cache(
                    input_ids[:, -1:], cache, use_ka_cache
                )

            # Sample next token
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def num_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        return {"total": total}
