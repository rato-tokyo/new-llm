"""New-LLM: Context Vector Propagation Language Model

This file defines the main processing flow for New-LLM.
All components are modular and can be easily swapped.

Architecture Overview:
    1. Token Embedding: input_ids → token_embeds
    2. Initialize Context: context = zeros
    3. For each position t:
        a. Concatenate: [token_embed[t], context]
        b. FNN processing: hidden = FNN([token, context])
        c. Token prediction: logits = OutputHead(hidden)
        d. Context update: context = ContextUpdater(hidden, context)
    4. Return logits and context trajectory

Key Innovation: NO attention mechanism, fixed O(1) memory regardless of sequence length.
"""

import torch
import torch.nn as nn

from .components.embeddings import TokenEmbedding
from .components.feedforward import FeedForwardNetwork
from .components.output_heads import TokenPredictionHead, ContextDecoder
from .components.context_updaters import get_context_updater


class NewLLM(nn.Module):
    """
    New-LLM: Context Vector Propagation Language Model

    Modular architecture with pluggable components:
    - Token embedding
    - Feedforward network
    - Context updater (simple/gated/custom)
    - Output heads (token prediction, context reconstruction)
    """

    def __init__(self, config):
        """
        Initialize New-LLM with config

        Args:
            config: Configuration object with attributes:
                - vocab_size: Vocabulary size
                - embed_dim: Token embedding dimension
                - context_vector_dim: Context vector dimension
                - hidden_dim: FNN hidden dimension
                - num_layers: Number of FNN layers
                - dropout: Dropout rate
                - context_update_strategy: "simple" or "gated"
                - max_seq_length: Maximum sequence length
        """
        super().__init__()
        self.config = config
        self.context_dim = config.context_vector_dim

        # ========== Component 1: Token Embedding ==========
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim
        )

        # ========== Component 2: Feedforward Network ==========
        # Input: [token_embed, context] concatenated
        fnn_input_dim = config.embed_dim + self.context_dim
        self.fnn = FeedForwardNetwork(
            input_dim=fnn_input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout
        )

        # ========== Component 3: Output Heads ==========
        # 3a. Token prediction head
        self.token_output = TokenPredictionHead(
            hidden_dim=config.hidden_dim,
            vocab_size=config.vocab_size
        )

        # 3b. Context decoder (for reconstruction learning)
        self.context_decoder = ContextDecoder(
            context_dim=self.context_dim,
            target_dim=config.embed_dim + self.context_dim
        )

        # ========== Component 4: Context Updater (Pluggable) ==========
        context_update_strategy = getattr(config, 'context_update_strategy', 'simple')
        self.context_updater = get_context_updater(
            strategy=context_update_strategy,
            config=config,
            hidden_dim=config.hidden_dim,
            context_dim=self.context_dim
        )

        # ========== Context Normalization (Critical for stability) ==========
        self.context_norm = nn.LayerNorm(self.context_dim)

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

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: Process sequence with context vector propagation

        Processing Flow (visible at a glance):
            1. Token embedding
            2. Initialize context vector
            3. For each position:
                a. Concatenate [token, context]
                b. FNN processing
                c. Token prediction
                d. Context update
            4. Return logits and context trajectory

        Args:
            input_ids: Token IDs [batch, seq_len]

        Returns:
            logits: Next token predictions [batch, seq_len, vocab_size]
            context_trajectory: Context vectors [batch, seq_len, context_dim]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # ========== Step 1: Token Embedding ==========
        token_embeds = self.token_embedding(input_ids)  # [batch, seq_len, embed_dim]

        # ========== Step 2: Initialize Context Vector ==========
        context = torch.zeros(batch_size, self.context_dim, device=device)

        # ========== Step 3: Sequential Processing ==========
        # Pre-allocate tensors for memory efficiency (avoids list append + torch.stack)
        logits = torch.zeros(batch_size, seq_len, self.token_output.vocab_size, device=device)
        context_trajectory = torch.zeros(batch_size, seq_len, self.context_dim, device=device)
        reconstruction_targets_tensor = torch.zeros(
            batch_size, seq_len, self.context_dim + self.token_embedding.embed_dim, device=device
        )

        for t in range(seq_len):
            # Get current token embedding
            current_token = token_embeds[:, t, :]  # [batch, embed_dim]

            # Store reconstruction target: [prev_context, current_token]
            reconstruction_targets_tensor[:, t, :] = torch.cat([context, current_token], dim=-1)

            # ========== Step 3a: Concatenate [token, context] ==========
            fnn_input = torch.cat([current_token, context], dim=-1)  # [batch, embed+context]

            # ========== Step 3b: FNN Processing ==========
            hidden = self.fnn(fnn_input)  # [batch, hidden_dim]

            # ========== Step 3c: Token Prediction ==========
            token_logits = self.token_output(hidden)  # [batch, vocab_size]
            logits[:, t, :] = token_logits

            # ========== Step 3d: Context Update (Pluggable Strategy) ==========
            context = self.context_updater(hidden, context)  # [batch, context_dim]

            # Normalize context (critical for stability)
            context = self.context_norm(context)

            # Clip to prevent extreme values
            context = torch.clamp(context, min=-10.0, max=10.0)

            # Store context
            context_trajectory[:, t, :] = context

        # ========== Step 4: Return Results ==========

        # Store for reconstruction learning (detach to avoid memory leak)
        self.context_history = context_trajectory.detach()
        self.reconstruction_targets = reconstruction_targets_tensor.detach()

        return logits, context_trajectory

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 20, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate new tokens autoregressively

        Args:
            input_ids: Starting tokens [batch, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature

        Returns:
            generated: Generated sequence [batch, seq_len + max_new_tokens]
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Initialize context from input sequence
        with torch.no_grad():
            _, context_traj = self.forward(input_ids)
            context = context_traj[:, -1, :]  # Last context vector

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get last token
                last_token_id = input_ids[:, -1:]  # [batch, 1]

                # Get token embedding
                token_embed = self.token_embedding.embedding(last_token_id).squeeze(1)
                token_embed = self.token_embedding.norm(token_embed)

                # Concatenate with context
                fnn_input = torch.cat([token_embed, context], dim=-1)

                # FNN forward
                hidden = self.fnn(fnn_input)

                # Get next token prediction
                next_token_logits = self.token_output(hidden) / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Update context
                context = self.context_updater(hidden, context)
                context = self.context_norm(context)
                context = torch.clamp(context, min=-10.0, max=10.0)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Truncate if needed
                if input_ids.size(1) > self.config.max_seq_length * 2:
                    input_ids = input_ids[:, -self.config.max_seq_length:]

        return input_ids
