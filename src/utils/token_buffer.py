"""
Token history buffer for new-llm.

Provides a reusable sliding window buffer for token history management.
"""

from typing import Union

import torch
from torch import Tensor


class TokenHistoryBuffer:
    """
    Sliding window buffer for managing token history.

    Used for num_input_tokens > 1 scenarios where multiple previous
    tokens need to be combined with the current token.

    Args:
        num_input_tokens: Number of tokens to combine (window size)
        embed_dim: Embedding dimension per token
        device: Device for tensor storage
    """

    def __init__(
        self,
        num_input_tokens: int,
        embed_dim: int,
        device: Union[str, torch.device]
    ):
        self.num_input_tokens = num_input_tokens
        self.embed_dim = embed_dim
        self.device = device

        # Initialize with zeros for history slots
        self.buffer = [
            torch.zeros(embed_dim, device=device)
            for _ in range(num_input_tokens - 1)
        ]

    def append_and_combine(self, token_embed: Tensor) -> Tensor:
        """
        Append a token to the buffer and return combined tokens.

        Args:
            token_embed: Token embedding [embed_dim]

        Returns:
            Combined tokens [embed_dim * num_input_tokens]
        """
        self.buffer.append(token_embed)
        combined = torch.cat(self.buffer[-self.num_input_tokens:], dim=-1)
        return combined

    def reset(self) -> None:
        """Reset buffer to initial state (all zeros)."""
        self.buffer = [
            torch.zeros(self.embed_dim, device=self.device)
            for _ in range(self.num_input_tokens - 1)
        ]

    def get_current_combined(self) -> Tensor:
        """
        Get current combined tokens without appending.

        Returns:
            Combined tokens [embed_dim * num_input_tokens]
        """
        return torch.cat(self.buffer[-self.num_input_tokens:], dim=-1)
