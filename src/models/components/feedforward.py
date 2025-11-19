"""Feedforward Network Module for New-LLM

Multi-layer feedforward network that processes concatenated [token, context] inputs.
"""

import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """
    Multi-layer feedforward network

    Processes concatenated input of [token_embedding, context_vector].
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        """
        Args:
            input_dim: Input dimension (typically embed_dim + context_dim)
            hidden_dim: Hidden layer dimension
            num_layers: Number of FNN layers
            dropout: Dropout rate
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Build layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                # First layer: input_dim -> hidden_dim
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])
            else:
                # Subsequent layers: hidden_dim -> hidden_dim
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ])

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FNN

        Args:
            x: Input tensor [batch, input_dim]

        Returns:
            hidden: Hidden representation [batch, hidden_dim]
        """
        return self.layers(x)

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
