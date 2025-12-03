"""
Context-KV Attention LLM - ContextをKVキャッシュとして使用するモデル

コンセプト:
- 等間隔（interval）でContextを取得してKVキャッシュとして使用
- 常に「現在位置のcontext」を含める
- KVキャッシュサイズを大幅に削減（~99%削減可能）

動作イメージ (interval=100):
  Position 350の場合:
    KV = [context[350], context[250], context[150], context[50]]
         ↑現在          ↑100前        ↑200前        ↑300前
"""

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.blocks import ContextBlock
from src.utils.io import print_flush
from src.utils.initialization import count_parameters
from src.utils.embedding import load_pretrained_gpt2_embeddings


class ContextToKV(nn.Module):
    """Context を K, V に変換するモジュール"""

    def __init__(self, context_dim: int, hidden_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Context -> K, V
        self.to_k = nn.Linear(context_dim, hidden_dim)
        self.to_v = nn.Linear(context_dim, hidden_dim)

    def forward(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Context を K, V に変換

        Args:
            context: [batch, context_dim] または [batch, num_chunks, context_dim]

        Returns:
            K: [batch, num_heads, seq_len, head_dim]
            V: [batch, num_heads, seq_len, head_dim]
        """
        # 2次元の場合は3次元に拡張
        if context.dim() == 2:
            context = context.unsqueeze(1)  # [batch, 1, context_dim]

        batch_size, seq_len, _ = context.shape

        k = self.to_k(context)  # [batch, seq_len, hidden_dim]
        v = self.to_v(context)  # [batch, seq_len, hidden_dim]

        # Multi-head形式に変換
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        return k, v


class ContextKVAttention(nn.Module):
    """Context-based KV Attention Layer"""

    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query projection (from token embeddings)
        self.to_q = nn.Linear(hidden_dim, hidden_dim)

        # Context to K, V
        self.context_to_kv = ContextToKV(context_dim, hidden_dim, num_heads)

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token_embeds: torch.Tensor,
        context_chunks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Context-based KV Attention

        Args:
            token_embeds: [batch, hidden_dim] - 現在のトークン埋め込み
            context_chunks: [batch, num_chunks, context_dim] - チャンクごとのcontext

        Returns:
            output: [batch, hidden_dim]
        """
        batch_size = token_embeds.shape[0]

        # Query from token embeddings
        q = self.to_q(token_embeds)  # [batch, hidden_dim]
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch, num_heads, 1, head_dim]

        # K, V from context chunks
        k, v = self.context_to_kv(context_chunks)
        # k, v: [batch, num_heads, num_chunks, head_dim]

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # [batch, num_heads, 1, num_chunks]

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention output
        attn_output = torch.matmul(attn_weights, v)
        # [batch, num_heads, 1, head_dim]

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, self.hidden_dim)

        output = self.out_proj(attn_output)

        # Residual + LayerNorm
        output = self.norm(token_embeds + output)

        return output


class ContextKVAttentionLLM(nn.Module):
    """
    Context-KV Attention LLM

    チャンク単位のcontextをKVキャッシュとして使用するモデル。

    Args:
        vocab_size: 語彙サイズ
        embed_dim: トークン埋め込み次元
        context_dims: 各ContextBlockの出力次元
        num_heads: Attention head数
        chunk_size: チャンクサイズ（何トークンごとにcontextを保存するか）
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context_dims: List[int],
        num_heads: int = 8,
        chunk_size: int = 100,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dims = context_dims
        self.num_context_blocks = len(context_dims)
        self.combined_context_dim = sum(context_dims)
        self.num_heads = num_heads
        self.chunk_size = chunk_size

        # Token Embeddings (GPT-2 pretrained)
        self._load_pretrained_embeddings()
        self.embed_norm = nn.LayerNorm(embed_dim)

        # N個のContextBlock
        self.context_blocks = nn.ModuleList([
            ContextBlock(context_dim=dim, embed_dim=embed_dim)
            for dim in context_dims
        ])

        # Context-KV Attention Layer
        self.context_kv_attention = ContextKVAttention(
            context_dim=self.combined_context_dim,
            hidden_dim=embed_dim,
            num_heads=num_heads,
        )

        # Output projection (FFN)
        self.output_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Output Head (Weight Tying)
        self.token_output = nn.Linear(embed_dim, vocab_size, bias=False)
        self.token_output.weight = self.token_embedding.weight
        print_flush("✓ Weight Tying: token_output shares weights with token_embedding")

    def _load_pretrained_embeddings(self) -> None:
        """GPT-2 embeddings をロード"""
        self.token_embedding = load_pretrained_gpt2_embeddings(
            self.vocab_size, self.embed_dim, freeze=True
        )

    def forward_attention(
        self,
        token_embeds: torch.Tensor,
        context_chunks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Context-KV Attention + FFN

        Args:
            token_embeds: [batch, embed_dim]
            context_chunks: [batch, num_chunks, combined_context_dim]

        Returns:
            output: [batch, embed_dim]
        """
        # Context-KV Attention
        hidden = self.context_kv_attention(token_embeds, context_chunks)

        # FFN
        hidden = hidden + self.output_ffn(hidden)

        return hidden

    def forward_output(self, hidden: torch.Tensor) -> torch.Tensor:
        """Output logits"""
        return self.token_output(hidden)

    def freeze_context_block(self, block_idx: int) -> None:
        """指定されたContextBlockをfreeze"""
        for param in self.context_blocks[block_idx].parameters():
            param.requires_grad = False

    def freeze_all_context_blocks(self) -> None:
        """全ContextBlockをfreeze"""
        for i in range(self.num_context_blocks):
            self.freeze_context_block(i)
        print_flush(f"✓ All {self.num_context_blocks} ContextBlocks frozen")

    def num_params(self) -> Dict[str, int]:
        """パラメータ数を返す"""
        embedding_params = self.token_embedding.weight.numel()
        embed_norm_params = count_parameters(self.embed_norm)

        context_block_params = {}
        total_context_params = 0
        for i, block in enumerate(self.context_blocks):
            params = count_parameters(block)
            context_block_params[f'context_block_{i}'] = params
            total_context_params += params

        attention_params = count_parameters(self.context_kv_attention)
        ffn_params = count_parameters(self.output_ffn)

        total = (
            embedding_params + embed_norm_params + total_context_params
            + attention_params + ffn_params
        )

        result = {
            'embedding': embedding_params,
            'embed_norm': embed_norm_params,
            'context_kv_attention': attention_params,
            'output_ffn': ffn_params,
            'total': total,
            'total_context_blocks': total_context_params,
        }
        result.update(context_block_params)
        return result


class ContextKVWrapper(nn.Module):
    """
    Phase 1 用: ContextBlock のラッパー（ContextKVAttentionLLM用）
    """

    def __init__(self, model: ContextKVAttentionLLM, block_idx: int = 0):
        super().__init__()
        self.model = model
        self.block_idx = block_idx

        # Phase1Trainerが期待するプロパティ
        self.token_embedding = model.token_embedding
        self.embed_norm = model.embed_norm
        self.context_dim = model.context_dims[block_idx]
        self.embed_dim = model.embed_dim
        self.vocab_size = model.vocab_size

        self.context_block = model.context_blocks[block_idx]

    def forward_context(
        self, context: torch.Tensor, token_embeds: torch.Tensor
    ) -> torch.Tensor:
        """ContextBlock の順伝搬"""
        return self.context_block(context, token_embeds)
