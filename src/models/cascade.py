"""
Cascade Context LLM - 複数ContextBlockをカスケード連結するモデル

N分割方式:
- Phase 1[i]: ContextBlock[i] を i 番目のデータ区間で学習（入力: ゼロベクトル）
- Phase 2: concat(context[0], ..., context[N-1]) で TokenBlock を学習
"""

from typing import Dict

import torch
import torch.nn as nn

from src.models.blocks import ContextBlock, TokenBlock
from src.utils.io import print_flush
from src.utils.initialization import count_parameters
from src.utils.embedding import load_pretrained_gpt2_embeddings


class CascadeContextLLM(nn.Module):
    """
    Cascade Context LLM - N個のContextBlockをカスケード連結（1層固定）

    N分割方式:
    - Phase 1[i]: ContextBlock[i] を i 番目のデータ区間で学習（入力: ゼロベクトル）
    - Phase 2: concat(context[0], ..., context[N-1]) で TokenBlock を学習

    Args:
        vocab_size: 語彙サイズ
        embed_dim: トークン埋め込み次元
        context_dim: 各ContextBlockの出力次元
        num_context_blocks: ContextBlockの数（デフォルト: 2）
        prev_context_steps: 前のトークン時のcontextも連結する数（0で無効）
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context_dim: int,
        num_context_blocks: int = 2,
        prev_context_steps: int = 0,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.num_context_blocks = num_context_blocks
        self.prev_context_steps = prev_context_steps
        # TokenBlockへの入力次元: (現在 + 履歴) × ブロック数
        self.combined_context_dim = context_dim * num_context_blocks * (1 + prev_context_steps)

        # Token Embeddings (GPT-2 pretrained)
        self._load_pretrained_embeddings()
        self.embed_norm = nn.LayerNorm(embed_dim)

        # N個のContextBlock（各1層）
        self.context_blocks = nn.ModuleList([
            ContextBlock(context_dim=context_dim, embed_dim=embed_dim)
            for _ in range(num_context_blocks)
        ])

        # TokenBlock（連結されたcontext用、1層）
        self.token_block = TokenBlock(
            context_dim=self.combined_context_dim,
            embed_dim=embed_dim,
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

    def forward_context(
        self, block_idx: int, context: torch.Tensor, token_embeds: torch.Tensor
    ) -> torch.Tensor:
        """指定されたContextBlockの順伝搬"""
        return self.context_blocks[block_idx](context, token_embeds)

    def forward_token(
        self, context: torch.Tensor, token_embeds: torch.Tensor
    ) -> torch.Tensor:
        """TokenBlock の順伝搬（contextは連結済み）"""
        return self.token_block(context, token_embeds)

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

        token_block_params = count_parameters(self.token_block)

        total = embedding_params + embed_norm_params + total_context_params + token_block_params

        result = {
            'embedding': embedding_params,
            'embed_norm': embed_norm_params,
            'token_block': token_block_params,
            'total': total,
            'total_context_blocks': total_context_params,
        }
        result.update(context_block_params)
        return result


class SingleContextWrapper(nn.Module):
    """
    Phase 1 用: ContextBlock のラッパー

    MemoryPhase1Trainer と互換性を持たせる。
    """

    def __init__(self, cascade_model: CascadeContextLLM, block_idx: int = 0):
        super().__init__()
        self.cascade_model = cascade_model
        self.block_idx = block_idx

        # Phase1Trainerが期待するプロパティ
        self.token_embedding = cascade_model.token_embedding
        self.embed_norm = cascade_model.embed_norm
        self.context_dim = cascade_model.context_dim
        self.embed_dim = cascade_model.embed_dim
        self.vocab_size = cascade_model.vocab_size

        self.context_block = cascade_model.context_blocks[block_idx]

    def forward_context(
        self, context: torch.Tensor, token_embeds: torch.Tensor
    ) -> torch.Tensor:
        """ContextBlock の順伝搬"""
        return self.context_block(context, token_embeds)
