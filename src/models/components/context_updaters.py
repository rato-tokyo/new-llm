"""Context Update Strategies for New-LLM

This module provides pluggable context update strategies that can be easily swapped.
Each strategy defines how the context vector is updated at each time step.
"""

import torch
import torch.nn as nn


class BaseContextUpdater(nn.Module):
    """Base class for context update strategies"""

    def __init__(self, config, hidden_dim, context_dim):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Update context vector based on hidden state

        Args:
            hidden: Hidden state from FNN [batch, hidden_dim]
            context: Current context vector [batch, context_dim]

        Returns:
            Updated context vector [batch, context_dim]
        """
        raise NotImplementedError("Subclasses must implement forward()")


class SimpleOverwriteUpdater(BaseContextUpdater):
    """
    Simple overwrite strategy: completely replace old context with new one.

    This is the most straightforward approach where the model generates a completely
    new context vector at each step, discarding the previous context.

    Formula:
        context_new = tanh(W_context @ hidden)
    """

    def __init__(self, config, hidden_dim, context_dim):
        super().__init__(config, hidden_dim, context_dim)

        # Single linear layer to produce new context
        self.context_update = nn.Linear(hidden_dim, context_dim)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Generate completely new context vector (overwrite old one)

        Args:
            hidden: [batch, hidden_dim]
            context: [batch, context_dim] - NOT USED (will be overwritten)

        Returns:
            new_context: [batch, context_dim]
        """
        # Generate new context (bounded to [-1, 1] by tanh)
        new_context = torch.tanh(self.context_update(hidden))

        return new_context


class GatedAdditiveUpdater(BaseContextUpdater):
    """
    Gated additive strategy: combine old context with new information using LSTM-style gates.

    This approach uses forget and input gates to control how much of the old context
    to retain and how much new information to incorporate.

    Formula:
        context_delta = tanh(W_delta @ hidden)
        forget_gate = sigmoid(W_forget @ hidden)
        input_gate = sigmoid(W_input @ hidden)
        context_new = forget_gate * context_old + input_gate * context_delta

    Inspired by LSTM cell update mechanism.
    """

    def __init__(self, config, hidden_dim, context_dim):
        super().__init__(config, hidden_dim, context_dim)

        # Context delta (new information to add)
        self.context_update = nn.Linear(hidden_dim, context_dim)

        # LSTM-style gates
        self.forget_gate = nn.Linear(hidden_dim, context_dim)
        self.input_gate = nn.Linear(hidden_dim, context_dim)

    def forward(self, hidden: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Update context using gated addition (LSTM-style)

        Args:
            hidden: [batch, hidden_dim]
            context: [batch, context_dim] - old context to be updated

        Returns:
            new_context: [batch, context_dim]
        """
        # Generate context delta (bounded to [-1, 1])
        context_delta = torch.tanh(self.context_update(hidden))

        # Generate gates (bounded to [0, 1])
        forget_g = torch.sigmoid(self.forget_gate(hidden))
        input_g = torch.sigmoid(self.input_gate(hidden))

        # Gated update: forget old + accept new
        new_context = forget_g * context + input_g * context_delta

        return new_context


# Registry for easy lookup
CONTEXT_UPDATERS = {
    "simple": SimpleOverwriteUpdater,
    "gated": GatedAdditiveUpdater,
}


def get_context_updater(strategy: str, config, hidden_dim: int, context_dim: int) -> BaseContextUpdater:
    """
    Factory function to get context updater by name

    Args:
        strategy: Name of the strategy ("simple" or "gated")
        config: Model configuration
        hidden_dim: Hidden dimension of FNN
        context_dim: Context vector dimension

    Returns:
        Context updater instance

    Raises:
        ValueError: If strategy name is not recognized
    """
    if strategy not in CONTEXT_UPDATERS:
        available = ", ".join(CONTEXT_UPDATERS.keys())
        raise ValueError(f"Unknown context update strategy: {strategy}. Available: {available}")

    updater_class = CONTEXT_UPDATERS[strategy]
    return updater_class(config, hidden_dim, context_dim)
