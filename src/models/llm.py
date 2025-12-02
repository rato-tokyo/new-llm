"""
New-LLM: Separated Context/Token Architecture

分離アーキテクチャ（1層固定、2025-12-02簡素化）:
1. ContextBlock: 文脈処理専用（Phase 1で学習、Phase 2でfreeze）
2. TokenBlock: トークン処理専用（Phase 2で学習）

Main features:
1. Context vector updates with residual connections (ContextBlock)
2. Token prediction with residual connections (TokenBlock)
3. LayerNorm for stable training
4. GPT-2 pretrained embeddings support

カスケード連結方式（2025-12-02）:
- 複数レイヤーは不要
- context_a → context_b のカスケード連結で十分な表現力
"""

from typing import Dict

import torch
import torch.nn as nn

from .blocks import ContextBlock, TokenBlock
from src.utils.io import print_flush
from src.utils.initialization import count_parameters


class LLM(nn.Module):
    """
    New-LLM with Separated Context/Token Architecture (1層固定)

    分離アーキテクチャ:
    - ContextBlock: 文脈処理専用（Phase 1で学習、Phase 2でfreeze）
    - TokenBlock: トークン処理専用（Phase 2で学習）

    1層固定アーキテクチャ（2025-12-02）:
    - カスケード連結方式により複数レイヤーは不要
    - ContextBlock: 1層、TokenBlock: 1層で固定

    Args:
        vocab_size: Vocabulary size
        embed_dim: Token embedding dimension
        context_dim: Context vector dimension
        num_input_tokens: Number of input tokens (1 = current only)
        use_pretrained_embeddings: Whether to use GPT-2 pretrained embeddings
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context_dim: int,
        num_input_tokens: int = 1,
        use_pretrained_embeddings: bool = True,
    ) -> None:
        super().__init__()

        # Save configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.num_input_tokens = num_input_tokens

        # ========== Token Embeddings ==========
        if use_pretrained_embeddings:
            self._load_pretrained_embeddings()
        else:
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.embed_norm = nn.LayerNorm(embed_dim)

        # ========== Separated Architecture (1層固定) ==========
        print_flush("Architecture: ContextBlock(1L) + TokenBlock(1L)")
        print_flush(f"  context_dim: {context_dim}")
        print_flush(f"  num_input_tokens: {num_input_tokens}")

        # ContextBlock: 文脈処理専用（1層）
        self.context_block = ContextBlock(
            context_dim=context_dim,
            embed_dim=embed_dim,
            num_input_tokens=num_input_tokens,
        )

        # TokenBlock: トークン処理専用（1層）
        self.token_block = TokenBlock(
            context_dim=context_dim,
            embed_dim=embed_dim,
            num_input_tokens=num_input_tokens,
        )

        # ========== Output Head (Weight Tying) ==========
        self.token_output = nn.Linear(embed_dim, vocab_size, bias=False)
        self.token_output.weight = self.token_embedding.weight
        saved_params = vocab_size * embed_dim
        print_flush("✓ Weight Tying: token_output shares weights with token_embedding")
        print_flush(f"  → Saved ~{saved_params / 1e6:.2f}M parameters")

        # Initialize embeddings (if not pretrained)
        if not use_pretrained_embeddings:
            self._init_weights()

    def _load_pretrained_embeddings(self) -> None:
        """Load GPT-2 pretrained embeddings"""
        try:
            from transformers import GPT2Model
            print_flush("Loading GPT-2 pretrained embeddings...")

            gpt2 = GPT2Model.from_pretrained('gpt2')
            pretrained_embeddings = gpt2.wte.weight.data

            self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            self.token_embedding.weight.data.copy_(pretrained_embeddings)
            self.token_embedding.weight.requires_grad = False

            print_flush(f"✓ Loaded GPT-2 embeddings: {pretrained_embeddings.shape}")

        except Exception as e:
            print_flush(f"Warning: Failed to load GPT-2 embeddings: {e}")
            print_flush("Falling back to random initialization...")
            self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    def _init_weights(self) -> None:
        """Initialize embedding weights"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    def forward_context(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """
        ContextBlock forward pass (Phase 1用)

        Args:
            context: [batch, context_dim]
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            new_context: [batch, context_dim]
        """
        return self.context_block(context, token_embeds)

    def forward_token(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """
        TokenBlock forward pass (Phase 2用)

        Args:
            context: [batch, context_dim]
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            token_out: [batch, embed_dim]
        """
        return self.token_block(context, token_embeds)

    def freeze_context_block(self) -> None:
        """ContextBlockのパラメータをfreezeする（Phase 2用）"""
        for param in self.context_block.parameters():
            param.requires_grad = False
        print_flush("✓ ContextBlock frozen")

    def num_params(self) -> Dict[str, int]:
        """
        モデル全体のパラメータ数を返す

        Returns:
            パラメータ数の詳細辞書
        """
        embedding_params = self.token_embedding.weight.numel()
        embed_norm_params = count_parameters(self.embed_norm)
        context_block_params = self.context_block.num_params()
        token_block_params = self.token_block.num_params()

        total = embedding_params + embed_norm_params + context_block_params + token_block_params

        return {
            'embedding': embedding_params,
            'embed_norm': embed_norm_params,
            'context_block': context_block_params,
            'token_block': token_block_params,
            'total': total,
            'trainable_phase1': context_block_params,
            'trainable_phase2': token_block_params,
        }
