"""
V-DProj Pythia: Value Compression for KV Cache Reduction

Vを圧縮してそのままAttentionに使用する方式。

アーキテクチャ:
- Q, K: 通常通り (512-dim, head_dim=64)
- V: 512 → v_compress → 320 (head_dim=40) でAttention計算
- Output: 320 → dense_v → 512

KVキャッシュ削減:
- K: 512-dim (変更なし)
- V: 320-dim (圧縮)
- 削減率: (512+512 - 512+320) / (512+512) = 18.8%
"""

from typing import Optional, Dict, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pythia import RotaryEmbedding, apply_rotary_pos_emb


class VDProjAttention(nn.Module):
    """
    V-DProj Attention

    Vを圧縮してそのままAttention計算に使用。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        v_proj_dim: int = 320,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads  # 64
        self.rotary_dim = int(self.head_dim * rotary_pct)
        self.v_proj_dim = v_proj_dim
        self.v_head_dim = v_proj_dim // num_heads  # 40

        # Query, Key projections (standard)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)

        # Value: compress to v_proj_dim
        self.value = nn.Linear(hidden_size, v_proj_dim)

        # Output projection: from v_proj_dim back to hidden_size
        self.dense = nn.Linear(v_proj_dim, hidden_size)

        # Rotary embedding
        self.rotary_emb = RotaryEmbedding(
            self.rotary_dim,
            max_position_embeddings=max_position_embeddings,
        )

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Q, K projection (standard 512-dim)
        query = self.query(hidden_states)
        key = self.key(hidden_states)

        # V projection (compressed 320-dim)
        value = self.value(hidden_states)

        # Reshape Q, K to heads: [batch, heads, seq, head_dim=64]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        query = query.transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.transpose(1, 2)

        # Reshape V to heads: [batch, heads, seq, v_head_dim=40]
        value = value.view(batch_size, seq_len, self.num_heads, self.v_head_dim)
        value = value.transpose(1, 2)

        # Apply rotary embedding to Q, K
        cos, sin = self.rotary_emb(query, seq_len)

        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]

        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)

        query = torch.cat([query_rot, query_pass], dim=-1)
        key = torch.cat([key_rot, key_pass], dim=-1)

        # Attention scores: Q @ K^T
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        # Causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device)
                * float("-inf"),
                diagonal=1,
            )
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Attention output: attn @ V (compressed)
        # [batch, heads, seq, seq] @ [batch, heads, seq, v_head_dim=40]
        # = [batch, heads, seq, v_head_dim=40]
        attn_output = torch.matmul(attn_weights, value)

        # Reshape back: [batch, seq, v_proj_dim=320]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.v_proj_dim)

        # Output projection: 320 → 512
        output = self.dense(attn_output)

        return output


class VDProjLayer(nn.Module):
    """
    V-DProj Transformer Layer (Pythia style)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        v_proj_dim: int = 320,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()

        # Layer norms
        self.input_layernorm = nn.LayerNorm(hidden_size)

        # V-DProj Attention
        self.attention = VDProjAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            v_proj_dim=v_proj_dim,
            rotary_pct=rotary_pct,
            max_position_embeddings=max_position_embeddings,
        )

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-LayerNorm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Parallel attention and MLP (Pythia specific)
        attn_output = self.attention(hidden_states, attention_mask)
        mlp_output = self.mlp(hidden_states)

        # Combine and add residual
        hidden_states = residual + attn_output + mlp_output

        return hidden_states


class VDProjPythiaModel(nn.Module):
    """
    V-DProj Pythia: Pythia with Value Compression

    Vを圧縮してそのままAttentionに使用し、KVキャッシュを削減。
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        v_proj_dim: int = 320,
        max_position_embeddings: int = 2048,
        rotary_pct: float = 0.25,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.v_proj_dim = v_proj_dim

        # Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # V-DProj Layers
        self.layers = nn.ModuleList(
            [
                VDProjLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    intermediate_size=intermediate_size,
                    v_proj_dim=v_proj_dim,
                    rotary_pct=rotary_pct,
                    max_position_embeddings=max_position_embeddings,
                )
                for _ in range(num_layers)
            ]
        )

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

        # V-DProj Layers
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

        # V projection parameters (value linear + dense)
        v_proj_params = 0
        for layer in self.layers:
            # value: hidden_size → v_proj_dim
            v_proj_params += layer.attention.value.weight.numel()
            v_proj_params += layer.attention.value.bias.numel()
            # dense: v_proj_dim → hidden_size
            v_proj_params += layer.attention.dense.weight.numel()
            v_proj_params += layer.attention.dense.bias.numel()

        return {
            "total": total,
            "embedding": embedding,
            "lm_head": lm_head,
            "v_projection": v_proj_params,
            "transformer": total - embedding - lm_head,
        }

    def kv_cache_size(self, seq_len: int, batch_size: int = 1) -> Dict[str, Union[int, float]]:
        """
        Calculate KV cache size

        Args:
            seq_len: Sequence length
            batch_size: Batch size

        Returns:
            Dictionary with cache sizes in bytes (assuming float32)
        """
        # Standard: K (512) + V (512) per layer
        standard_k = batch_size * seq_len * self.hidden_size * self.num_layers * 4
        standard_v = batch_size * seq_len * self.hidden_size * self.num_layers * 4
        standard_total = standard_k + standard_v

        # V-DProj: K (512) + V_compressed (320) per layer
        vdproj_k = batch_size * seq_len * self.hidden_size * self.num_layers * 4
        vdproj_v = batch_size * seq_len * self.v_proj_dim * self.num_layers * 4
        vdproj_total = vdproj_k + vdproj_v

        reduction = (standard_total - vdproj_total) / standard_total * 100

        return {
            "standard_k_bytes": standard_k,
            "standard_v_bytes": standard_v,
            "standard_total_bytes": standard_total,
            "vdproj_k_bytes": vdproj_k,
            "vdproj_v_bytes": vdproj_v,
            "vdproj_total_bytes": vdproj_total,
            "reduction_percent": reduction,
        }
