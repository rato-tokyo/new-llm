"""Baseline FNN-based Language Model (no attention mechanism)"""

import torch
import torch.nn as nn


class BaselineLLM(nn.Module):
    """
    Simple feedforward-based language model without attention.
    Processes each token independently through FNN layers.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # Positional embedding (learned)
        self.position_embedding = nn.Embedding(config.max_seq_length, config.embed_dim)

        # Feedforward layers
        layers = []
        input_dim = config.embed_dim
        for _ in range(config.num_layers):
            layers.extend([
                nn.Linear(input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.embed_dim),
                nn.Dropout(config.dropout),
            ])
            input_dim = config.embed_dim

        self.fnn_layers = nn.Sequential(*layers)

        # Output projection to vocabulary
        self.output_projection = nn.Linear(config.embed_dim, config.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: Token indices [batch_size, seq_len]

        Returns:
            logits: Next token predictions [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Get embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch, seq, embed]

        # Add positional embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)  # [1, seq, embed]
        x = token_embeds + pos_embeds

        # Process through FNN layers (each token independently)
        x = self.fnn_layers(x)  # [batch, seq, embed]

        # Project to vocabulary
        logits = self.output_projection(x)  # [batch, seq, vocab_size]

        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 20, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate new tokens autoregressively

        Args:
            input_ids: Starting tokens [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature

        Returns:
            generated: Generated sequence [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                logits = self.forward(input_ids)

                # Get logits for last position
                next_token_logits = logits[:, -1, :] / temperature

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Truncate if exceeds max length
                if input_ids.size(1) > self.config.max_seq_length:
                    input_ids = input_ids[:, -self.config.max_seq_length:]

        return input_ids
