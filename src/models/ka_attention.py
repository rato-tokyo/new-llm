"""
KA-Attention: Key-Attention Cache Method

通常のKVキャッシュの代わりに、KAキャッシュを使用する方式。
- K: Key vectors (通常通り)
- A: Attention output (Vの代わりに、過去のattention出力を使用)

仮説:
Aには「Vの情報 + Attention構造の情報」が含まれている。
過去トークンについては、既に集約処理された結果を再利用することで、
同等の表現力を維持しながら異なる情報構造を持つ。

動作:
- トークン1: Q[1], K[1], V[1] → attention計算 → A[1]
- トークン2: Q[2], K[1:2] → attention_probs → weighted sum of [A[1], V[2]] → A[2]
- トークンn: Q[n], K[1:n] → attention_probs → weighted sum of [A[1:n-1], V[n]] → A[n]
"""

from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pythia import RotaryEmbedding, apply_rotary_pos_emb


class KAAttention(nn.Module):
    """
    KA-Attention: Key-Attention Cache Method

    過去トークンにはV[i]の代わりにA[i]（attention output）を使用。
    現在のトークンにはV[n]を使用。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_dim = int(self.head_dim * rotary_pct)

        # Query, Key, Value projections (通常通り)
        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)

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
    ) -> torch.Tensor:
        """
        Forward pass with KA-cache style computation

        学習時: 全シーケンスを一度に処理（効率的な並列計算）

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        query, key, value = qkv[0], qkv[1], qkv[2]

        # Apply rotary embedding
        cos, sin = self.rotary_emb(query, seq_len)

        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]

        query_rot, key_rot = apply_rotary_pos_emb(query_rot, key_rot, cos, sin)

        query = torch.cat([query_rot, query_pass], dim=-1)
        key = torch.cat([key_rot, key_pass], dim=-1)

        # Attention scores
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        # Causal mask
        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device) * float("-inf"),
                diagonal=1
            )
            attn_weights = attn_weights + causal_mask

        attn_probs = F.softmax(attn_weights, dim=-1)

        # ===== KA-cache style computation =====
        # 各位置iの出力A[i]を順次計算
        # A[i] = sum_j(attn_probs[i,j] * (A[j] if j < i else V[j]))

        # value: [batch, heads, seq, head_dim]
        # attn_probs: [batch, heads, seq, seq]

        # 位置ごとに計算（学習時も推論時と同じロジック）
        A_outputs: List[torch.Tensor] = []

        for i in range(seq_len):
            # 位置iでのattention weights: [batch, heads, 1, i+1]
            attn_i = attn_probs[:, :, i:i+1, :i+1]  # [batch, heads, 1, i+1]

            if i == 0:
                # 最初のトークン: V[0]のみ使用
                # attn_i: [batch, heads, 1, 1], value[:,:,0:1]: [batch, heads, 1, head_dim]
                A_i = torch.matmul(attn_i, value[:, :, 0:1, :])  # [batch, heads, 1, head_dim]
            else:
                # 過去トークン: A[0:i-1]を使用
                # 現在トークン: V[i]を使用
                past_A = torch.cat(A_outputs, dim=2)  # [batch, heads, i, head_dim]
                current_V = value[:, :, i:i+1, :]     # [batch, heads, 1, head_dim]

                # 結合: [A[0], A[1], ..., A[i-1], V[i]]
                values_for_i = torch.cat([past_A, current_V], dim=2)  # [batch, heads, i+1, head_dim]

                # Attention出力
                A_i = torch.matmul(attn_i, values_for_i)  # [batch, heads, 1, head_dim]

            A_outputs.append(A_i)

        # 全てのA出力を結合
        attn_output = torch.cat(A_outputs, dim=2)  # [batch, heads, seq, head_dim]

        # Reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        output = self.dense(attn_output)
        return output


class KALayer(nn.Module):
    """
    KA-Attention Layer (Pythia style with parallel attention)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()

        # Layer norms
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

        # KA-Attention
        self.attention = KAAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            rotary_pct=rotary_pct,
            max_position_embeddings=max_position_embeddings,
        )

        # MLP (same as Pythia)
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


class KAPythiaModel(nn.Module):
    """
    KA-Pythia: Pythia with KA-Attention

    通常のPythiaと同じ構造だが、AttentionでKA方式を使用。
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

        # Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # KA-Attention Layers
        self.layers = nn.ModuleList([
            KALayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                rotary_pct=rotary_pct,
                max_position_embeddings=max_position_embeddings,
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

        # KA-Attention Layers
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
