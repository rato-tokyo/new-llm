"""Token Embedding Module for New-LLM

Simple token embedding layer with normalization.
"""

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """
    Token embedding layer with LayerNorm

    Converts token IDs to dense vectors and normalizes them.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embeddings
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Normalization (standard in modern LLMs)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings

        Args:
            input_ids: Token IDs [batch, seq_len]

        Returns:
            embeddings: Token embeddings [batch, seq_len, embed_dim]
        """
        # Get embeddings
        embeds = self.embedding(input_ids)

        # Normalize (standard in modern LLMs)
        embeds = self.norm(embeds)

        return embeds

    def _init_weights(self):
        """Initialize weights"""
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
