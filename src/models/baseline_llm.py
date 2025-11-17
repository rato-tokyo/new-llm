"""Baseline LSTM-based Language Model (no attention mechanism)"""

import torch
import torch.nn as nn


class BaselineLLM(nn.Module):
    """
    LSTM-based language model without attention.
    Uses LSTM hidden states to capture sequential context.
    This provides a fairer comparison to context vector propagation.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # LSTM layers (stacked)
        self.lstm = nn.LSTM(
            input_size=config.embed_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            batch_first=True
        )

        # Output projection to vocabulary
        self.output_projection = nn.Linear(config.hidden_dim, config.vocab_size)

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

        # Process through LSTM (captures sequential context in hidden states)
        lstm_output, _ = self.lstm(token_embeds)  # [batch, seq, hidden_dim]

        # Project to vocabulary
        logits = self.output_projection(lstm_output)  # [batch, seq, vocab_size]

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
        batch_size = input_ids.size(0)

        with torch.no_grad():
            # Initialize LSTM hidden state
            h = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_dim, device=input_ids.device)
            c = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_dim, device=input_ids.device)

            # Process initial sequence to get hidden state
            embeds = self.token_embedding(input_ids)
            _, (h, c) = self.lstm(embeds, (h, c))

            # Generate new tokens
            for _ in range(max_new_tokens):
                # Get embedding of last token
                last_token_embed = self.token_embedding(input_ids[:, -1:])

                # LSTM forward with hidden state
                lstm_out, (h, c) = self.lstm(last_token_embed, (h, c))

                # Get next token prediction
                next_token_logits = self.output_projection(lstm_out[:, -1, :]) / temperature

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
