"""
Context-Pythia: Pythia with Context-based KV Compression

全LayerのAttention入力をcontext (256-dim) に置き換え、
KVキャッシュを50%削減する。

Architecture:
1. Token Embedding (512-dim) - Pythia pretrained
2. ContextBlock: token_embed → context (256-dim)
   ⚠️ 既存の動作確認済みContextBlockを再利用
3. 6 Layers: 全て context を入力として使用
4. Output Head: 512-dim → vocab_size

Training:
- Phase 1: ContextBlock のみ学習 (OACD)
- Phase 2: 全体をファインチューニング
"""

import math
from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pythia import RotaryEmbedding, apply_rotary_pos_emb
from src.models.blocks import ContextBlock  # 既存の動作確認済みContextBlockを再利用
from src.utils.io import print_flush


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
    ):
        super().__init__()
        self.context_dim = context_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_dim = int(self.head_dim * rotary_pct)

        # Query: from context (context_dim -> hidden_size)
        self.query_proj = nn.Linear(context_dim, hidden_size)

        # Key, Value: from context (context_dim -> hidden_size each)
        # これにより KV cache は context_dim ベースになる
        self.key_proj = nn.Linear(context_dim, hidden_size)
        self.value_proj = nn.Linear(context_dim, hidden_size)

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
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            context: [batch, seq_len, context_dim]

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = context.shape

        # Project from context_dim to hidden_size
        query = self.query_proj(context)
        key = self.key_proj(context)
        value = self.value_proj(context)

        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

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
                torch.ones(seq_len, seq_len, device=context.device) * float("-inf"),
                diagonal=1
            )
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Attention output
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        output = self.dense(attn_output)
        return output


class ContextPythiaMLP(nn.Module):
    """Context-Pythia Feed-Forward Network"""

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


class ContextPythiaLayer(nn.Module):
    """
    Context-Pythia Transformer Layer

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
    ):
        super().__init__()

        self.context_dim = context_dim
        self.hidden_size = hidden_size

        # Context -> Hidden projection (最初のレイヤーで必要)
        self.context_proj = nn.Linear(context_dim, hidden_size)

        # Layer norms
        self.input_layernorm = nn.LayerNorm(hidden_size)

        # Attention (context-based)
        self.attention = ContextPythiaAttention(
            context_dim=context_dim,
            hidden_size=hidden_size,
            num_heads=num_heads,
            rotary_pct=rotary_pct,
            max_position_embeddings=max_position_embeddings,
        )

        # MLP
        self.mlp = ContextPythiaMLP(hidden_size, intermediate_size)

    def forward(
        self,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            context: [batch, seq_len, context_dim]

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
        """
        # Project context to hidden size for residual
        residual = self.context_proj(context)

        # Pre-LayerNorm
        hidden_states = self.input_layernorm(residual)

        # Parallel attention and MLP
        attn_output = self.attention(context, attention_mask)
        mlp_output = self.mlp(hidden_states)

        # Combine and add residual
        hidden_states = residual + attn_output + mlp_output

        return hidden_states


class ContextPythiaModel(nn.Module):
    """
    Context-Pythia Language Model

    Architecture:
    1. Embedding: vocab_size -> hidden_size (512)
    2. ContextBlock: hidden_size -> context_dim (256)
       ⚠️ 既存の動作確認済みContextBlockを使用
    3. 6 Transformer layers (context-based attention)
    4. Final LayerNorm
    5. LM Head: hidden_size -> vocab_size

    KV Cache削減: 50% (context_dim / hidden_size)
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
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.context_dim = context_dim
        self.num_layers = num_layers

        # Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # ContextBlock: hidden_size -> context_dim
        # ⚠️ 既存の動作確認済みContextBlockを使用
        # 初期化は normal_(std=0.1) で行われる（削除禁止）
        self.context_block = ContextBlock(
            context_dim=context_dim,
            embed_dim=hidden_size,
        )

        # Context-based Transformer layers
        self.layers = nn.ModuleList([
            ContextPythiaLayer(
                context_dim=context_dim,
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

        # Initialize weights (ContextBlock以外)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights (ContextBlock以外)"""
        for name, module in self.named_modules():
            # ContextBlockは既に初期化済み（normal_ std=0.1）なのでスキップ
            if "context_block" in name:
                continue

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
        Forward pass

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            prev_context: [batch, context_dim] 前回のcontext（オプション）

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Embedding
        token_embeds = self.embed_in(input_ids)  # [batch, seq, hidden_size]

        # ContextBlock: token_embed -> context
        # shifted_prev_context方式で並列処理
        if prev_context is None:
            prev_context = torch.zeros(batch_size, self.context_dim, device=input_ids.device)

        # 各位置のcontextを計算
        contexts = []
        current_context = prev_context
        for i in range(seq_len):
            current_context = self.context_block(
                current_context,
                token_embeds[:, i, :]
            )
            contexts.append(current_context)

        context = torch.stack(contexts, dim=1)  # [batch, seq, context_dim]

        # Context-based Transformer layers
        hidden_states = None
        for layer in self.layers:
            hidden_states = layer(context, attention_mask)

        # Final layer norm
        assert hidden_states is not None
        hidden_states = self.final_layer_norm(hidden_states)

        # LM Head
        logits = self.embed_out(hidden_states)

        return logits

    def forward_with_context(
        self,
        input_ids: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Context already computed (Phase 2用)

        Args:
            input_ids: [batch, seq_len] (使用しない、互換性のため)
            context: [batch, seq_len, context_dim]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Context-based Transformer layers
        hidden_states = None
        for layer in self.layers:
            hidden_states = layer(context, attention_mask)

        # Final layer norm
        assert hidden_states is not None
        hidden_states = self.final_layer_norm(hidden_states)

        # LM Head
        logits = self.embed_out(hidden_states)

        return logits

    def freeze_context_block(self) -> None:
        """ContextBlockをfreeze（Phase 2用）"""
        for param in self.context_block.parameters():
            param.requires_grad = False
        print_flush("✓ ContextBlock frozen")

    def unfreeze_context_block(self) -> None:
        """ContextBlockをunfreeze"""
        for param in self.context_block.parameters():
            param.requires_grad = True

    def num_parameters(self) -> Dict[str, int]:
        """Count parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        context_block = sum(p.numel() for p in self.context_block.parameters())
        embedding = self.embed_in.weight.numel()
        lm_head = self.embed_out.weight.numel()

        return {
            "total": total,
            "trainable": trainable,
            "context_block": context_block,
            "embedding": embedding,
            "lm_head": lm_head,
            "transformer": total - context_block - embedding - lm_head,
        }

    def kv_cache_size_comparison(self, seq_len: int = 1024) -> Dict[str, float]:
        """KV cacheサイズの比較"""
        # 元のPythia: hidden_size × seq_len × num_layers × 2 (K,V)
        original_kv = self.hidden_size * seq_len * self.num_layers * 2 * 4  # float32

        # Context-Pythia: context_dim × seq_len × num_layers × 2 (K,V)
        context_kv = self.context_dim * seq_len * self.num_layers * 2 * 4

        reduction = 1 - (context_kv / original_kv)

        return {
            "original_bytes": original_kv,
            "context_bytes": context_kv,
            "reduction_pct": reduction * 100,
            "original_mb": original_kv / (1024 * 1024),
            "context_mb": context_kv / (1024 * 1024),
        }
