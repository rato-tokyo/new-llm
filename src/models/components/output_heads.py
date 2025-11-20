"""Output Heads for New-LLM

Contains prediction heads for:
1. Token prediction (language modeling)
2. Context reconstruction (optional, for reconstruction learning)
"""

import torch
import torch.nn as nn


class TokenPredictionHead(nn.Module):
    """
    Token prediction head

    Projects hidden state to vocabulary logits for next token prediction.
    """

    def __init__(self, hidden_dim: int, vocab_size: int):
        """
        Args:
            hidden_dim: Hidden state dimension
            vocab_size: Vocabulary size
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Predict next token

        Args:
            hidden: Hidden state [batch, hidden_dim]

        Returns:
            logits: Token logits [batch, vocab_size]
        """
        return self.output(hidden)


class ContextDecoder(nn.Module):
    """
    Context reconstruction decoder

    Reconstructs [prev_context + current_token] from context vector.
    Used for reconstruction learning (optional training objective).
    """

    def __init__(self, context_dim: int, target_dim: int):
        """
        Args:
            context_dim: Context vector dimension
            target_dim: Target dimension (embed_dim + context_dim)
        """
        super().__init__()
        self.context_dim = context_dim
        self.target_dim = target_dim

        self.decoder = nn.Sequential(
            nn.Linear(context_dim, target_dim),
            nn.ReLU(),
            nn.Linear(target_dim, target_dim),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct [prev_context + current_token] from context

        Args:
            context: Context vector [batch, context_dim]

        Returns:
            reconstruction: Reconstructed [prev_context + token] [batch, target_dim]
        """
        return self.decoder(context)
