"""
V-DProj Pythia: Value Compression with Invertible Projection

Vに対してDProj（次元圧縮）を適用し、逆射影で復元する方式。

アーキテクチャ:
- Q, K: 通常通り (512-dim)
- V: 512 → v_proj → 320 → v_inv_proj → 512

学習目標:
1. Reconstruction Loss: ||V - V_restored||^2
2. LM Loss: Cross-entropy

KVキャッシュ削減:
- 推論時はV_compressed (320-dim) のみ保存
- 512 → 320 = 37.5%削減
"""

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pythia import RotaryEmbedding, apply_rotary_pos_emb


class VDProjAttention(nn.Module):
    """
    V-DProj Attention

    VをDProjで圧縮し、逆射影で復元してAttentionを計算。
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
        self.head_dim = hidden_size // num_heads
        self.rotary_dim = int(self.head_dim * rotary_pct)
        self.v_proj_dim = v_proj_dim
        self.v_head_dim = v_proj_dim // num_heads

        # Query, Key, Value projections
        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)

        # V compression: hidden_size → v_proj_dim
        self.v_compress = nn.Linear(hidden_size, v_proj_dim)

        # V restoration: v_proj_dim → hidden_size
        self.v_restore = nn.Linear(v_proj_dim, hidden_size)

        # Output projection
        self.dense = nn.Linear(hidden_size, hidden_size)

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
        return_reconstruction_loss: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            return_reconstruction_loss: Whether to return V reconstruction loss

        Returns:
            output: [batch, seq_len, hidden_size]
            reconstruction_loss: Optional[torch.Tensor] if return_reconstruction_loss
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Apply rotary embedding to Q, K
        cos, sin = self.rotary_emb(query, seq_len)

        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]

        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)

        query = torch.cat([query_rot, query_pass], dim=-1)
        key = torch.cat([key_rot, key_pass], dim=-1)

        # ===== V Compression & Restoration =====
        # value: [batch, heads, seq, head_dim] → hidden_states format
        value_hidden = value.transpose(1, 2).contiguous()
        value_hidden = value_hidden.view(batch_size, seq_len, self.hidden_size)

        # Compress V
        v_compressed = self.v_compress(value_hidden)  # [batch, seq, v_proj_dim]

        # Restore V
        v_restored = self.v_restore(v_compressed)  # [batch, seq, hidden_size]

        # Compute reconstruction loss if needed
        reconstruction_loss = None
        if return_reconstruction_loss:
            reconstruction_loss = F.mse_loss(v_restored, value_hidden)

        # Convert back to heads format for attention
        v_restored_heads = v_restored.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        v_restored_heads = v_restored_heads.transpose(1, 2)

        # Attention scores
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

        # Attention output with restored V
        attn_output = torch.matmul(attn_weights, v_restored_heads)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        output = self.dense(attn_output)

        if return_reconstruction_loss:
            return output, reconstruction_loss
        return output, None


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
        return_reconstruction_loss: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Pre-LayerNorm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Parallel attention and MLP (Pythia specific)
        attn_output, recon_loss = self.attention(
            hidden_states,
            attention_mask,
            return_reconstruction_loss=return_reconstruction_loss,
        )
        mlp_output = self.mlp(hidden_states)

        # Combine and add residual
        hidden_states = residual + attn_output + mlp_output

        return hidden_states, recon_loss


class VDProjPythiaModel(nn.Module):
    """
    V-DProj Pythia: Pythia with Value Compression

    Vを圧縮して復元する方式でKVキャッシュを削減。
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
        return_reconstruction_loss: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            return_reconstruction_loss: Whether to return V reconstruction loss

        Returns:
            logits: [batch, seq_len, vocab_size]
            reconstruction_loss: Optional[torch.Tensor] average across all layers
        """
        # Embedding
        hidden_states = self.embed_in(input_ids)

        # Collect reconstruction losses
        total_recon_loss = 0.0
        num_layers_with_loss = 0

        # V-DProj Layers
        for layer in self.layers:
            hidden_states, recon_loss = layer(
                hidden_states,
                attention_mask,
                return_reconstruction_loss=return_reconstruction_loss,
            )
            if recon_loss is not None:
                total_recon_loss = total_recon_loss + recon_loss
                num_layers_with_loss += 1

        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        # LM Head
        logits = self.embed_out(hidden_states)

        # Average reconstruction loss across layers
        avg_recon_loss = None
        if return_reconstruction_loss and num_layers_with_loss > 0:
            avg_recon_loss = total_recon_loss / num_layers_with_loss

        return logits, avg_recon_loss

    def num_parameters(self) -> Dict[str, int]:
        """Count parameters"""
        total = sum(p.numel() for p in self.parameters())
        embedding = self.embed_in.weight.numel()
        lm_head = self.embed_out.weight.numel()

        # V projection parameters
        v_proj_params = 0
        for layer in self.layers:
            v_proj_params += layer.attention.v_compress.weight.numel()
            v_proj_params += layer.attention.v_compress.bias.numel()
            v_proj_params += layer.attention.v_restore.weight.numel()
            v_proj_params += layer.attention.v_restore.bias.numel()

        return {
            "total": total,
            "embedding": embedding,
            "lm_head": lm_head,
            "v_projection": v_proj_params,
            "transformer": total - embedding - lm_head,
        }

    def kv_cache_size(self, seq_len: int, batch_size: int = 1) -> Dict[str, int]:
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
