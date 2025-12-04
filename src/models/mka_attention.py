"""
Mini-Context KA-Attention: Two-Stage Attention with Pure KV and KA Caches

提案手法:
1. Stage 1 (Local KV): 直近mini_context_lengthトークンのV[n-w:n]から純粋にA[n]を計算
2. Stage 2 (Global KA): 過去の全てのA[1:n-1]とA[n]から最終出力を計算

これにより:
- KVキャッシュ: 直近wトークンのVのみ（ローカル文脈）
- KAキャッシュ: 全過去のA（グローバル文脈）

前回のKA-Attention（VとAの混在）との違いを検証。
"""

from typing import Optional, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pythia import RotaryEmbedding, apply_rotary_pos_emb


class MKAAttention(nn.Module):
    """
    Mini-Context KA-Attention

    Stage 1: 直近mini_context_lengthトークンのKVから純粋にAを計算
    Stage 2: 過去の全てのAから最終出力を計算
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mini_context_length: int = 16,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_dim = int(self.head_dim * rotary_pct)
        self.mini_context_length = mini_context_length

        # Query, Key, Value projections
        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)

        # Stage 2用の追加projection（オプション）
        # A同士のattentionには別のQ/Kを使用
        self.query_a = nn.Linear(hidden_size, hidden_size)
        self.key_a = nn.Linear(hidden_size, hidden_size)

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
        Forward pass with two-stage attention

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape
        w = self.mini_context_length

        # ===== Stage 1: Local KV Attention =====
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

        # 各位置でのlocal A[i]を計算（直近wトークンのKVのみ使用）
        A_local: List[torch.Tensor] = []

        for i in range(seq_len):
            # Local window: [max(0, i-w+1), i+1)
            start = max(0, i - w + 1)
            end = i + 1

            # Local query, key, value
            q_i = query[:, :, i:i+1, :]  # [batch, heads, 1, head_dim]
            k_local = key[:, :, start:end, :]  # [batch, heads, local_len, head_dim]
            v_local = value[:, :, start:end, :]  # [batch, heads, local_len, head_dim]

            # Local attention
            attn_weights = torch.matmul(q_i, k_local.transpose(-1, -2)) * self.scale
            # [batch, heads, 1, local_len]

            attn_probs = F.softmax(attn_weights, dim=-1)

            # 純粋なKV attentionでA[i]を計算
            A_i = torch.matmul(attn_probs, v_local)  # [batch, heads, 1, head_dim]
            A_local.append(A_i)

        # 全local Aを結合
        A_all = torch.cat(A_local, dim=2)  # [batch, heads, seq, head_dim]

        # ===== Stage 2: Global KA Attention =====
        # A同士のattentionを計算
        # A[i]をhidden_statesに戻してからQ_a, K_aを計算

        # Reshape A to hidden_states format
        A_hidden = A_all.transpose(1, 2).contiguous()
        A_hidden = A_hidden.view(batch_size, seq_len, self.hidden_size)

        # Stage 2用のQ, K
        query_a = self.query_a(A_hidden)  # [batch, seq, hidden]
        key_a = self.key_a(A_hidden)  # [batch, seq, hidden]

        query_a = query_a.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key_a = key_a.view(batch_size, seq_len, self.num_heads, self.head_dim)
        query_a = query_a.transpose(1, 2)  # [batch, heads, seq, head_dim]
        key_a = key_a.transpose(1, 2)  # [batch, heads, seq, head_dim]

        # Global attention scores (全過去のAを参照)
        attn_weights_global = torch.matmul(query_a, key_a.transpose(-1, -2)) * self.scale
        # [batch, heads, seq, seq]

        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=hidden_states.device) * float("-inf"),
            diagonal=1
        )
        attn_weights_global = attn_weights_global + causal_mask

        attn_probs_global = F.softmax(attn_weights_global, dim=-1)

        # Global attention output (過去の全Aの重み付き和)
        # A_all: [batch, heads, seq, head_dim]
        attn_output = torch.matmul(attn_probs_global, A_all)

        # Reshape
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        output = self.dense(attn_output)
        return output


class MKALayer(nn.Module):
    """
    Mini-Context KA-Attention Layer (Pythia style)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        mini_context_length: int = 16,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()

        # Layer norms
        self.input_layernorm = nn.LayerNorm(hidden_size)

        # MKA-Attention
        self.attention = MKAAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mini_context_length=mini_context_length,
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


class MKAPythiaModel(nn.Module):
    """
    MKA-Pythia: Pythia with Mini-Context KA-Attention

    - Stage 1: 直近wトークンの純粋KV attention → local A
    - Stage 2: 全過去の純粋KA attention → global output
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        mini_context_length: int = 16,
        max_position_embeddings: int = 2048,
        rotary_pct: float = 0.25,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mini_context_length = mini_context_length

        # Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # MKA-Attention Layers
        self.layers = nn.ModuleList([
            MKALayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                mini_context_length=mini_context_length,
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

        # MKA-Attention Layers
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
