"""New-LLM 2-Layer: Context Vector Propagation Language Model (2 FNN Layers)

This is a 2-layer variant of New-LLM.
For easy experimentation, this is a separate file from new_llm.py (1-layer).

Architecture Overview:
    1. Token Embedding: input_ids → token_embeds
    2. Initialize Context: context = zeros
    3. For each position t:
        a. Concatenate: [token_embed[t], context]
        b. FNN Layer 1: hidden1 = FNN1([token, context])
        c. FNN Layer 2: hidden2 = FNN2(hidden1)
        d. Token prediction: logits = OutputHead(hidden2)
        e. Context update: context = ContextUpdater(hidden2, context)
    4. Return logits and context trajectory

Key Difference from 1-layer:
    - 2 FNN layers instead of 1
    - Only final layer (hidden2) is used for prediction and context update
"""

import torch
import torch.nn as nn

from .components.embeddings import TokenEmbedding
from .components.output_heads import TokenPredictionHead, ContextDecoder
from .components.context_updaters import get_context_updater


class NewLLM2Layer(nn.Module):
    """
    New-LLM 2-Layer: Context Vector Propagation Language Model

    Fixed 2-layer FNN architecture for easy experimentation.
    """

    def __init__(self, config):
        """
        Initialize New-LLM 2-Layer with config

        Args:
            config: Configuration object with attributes:
                - vocab_size: Vocabulary size
                - embed_dim: Token embedding dimension
                - context_vector_dim: Context vector dimension
                - hidden_dim: FNN hidden dimension
                - dropout: Dropout rate
                - context_update_strategy: "simple" or "gated"
                - max_seq_length: Maximum sequence length
        """
        super().__init__()
        self.config = config
        self.context_dim = config.context_vector_dim
        self.hidden_dim = config.hidden_dim

        # ========== Component 1: Token Embedding ==========
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim
        )

        # ========== Component 2: 2-Layer Feedforward Network ==========
        # Input: [token_embed, context] concatenated
        fnn_input_dim = config.embed_dim + self.context_dim

        # Layer 1: input_dim -> hidden_dim
        self.fnn_layer1 = nn.Sequential(
            nn.Linear(fnn_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # Layer 2: hidden_dim -> hidden_dim
        self.fnn_layer2 = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # ========== Component 3: Output Heads ==========
        # 3a. Token prediction head (uses final layer output)
        self.token_output = TokenPredictionHead(
            hidden_dim=self.hidden_dim,
            vocab_size=config.vocab_size
        )

        # 3b. Context decoder (for reconstruction learning)
        self.context_decoder = ContextDecoder(
            context_dim=self.context_dim,
            target_dim=config.embed_dim + self.context_dim
        )

        # ========== Component 4: Context Updater (Pluggable) ==========
        context_update_strategy = getattr(config, 'context_update_strategy', 'gated')
        self.context_updater = get_context_updater(
            strategy=context_update_strategy,
            config=config,
            hidden_dim=self.hidden_dim,
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

        Processing Flow:
            1. Token embedding
            2. Initialize context vector
            3. For each position:
                a. Concatenate [token, context]
                b. FNN Layer 1 processing
                c. FNN Layer 2 processing
                d. Token prediction (from layer 2)
                e. Context update (from layer 2)
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
        # Pre-allocate tensors for memory efficiency
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

            # ========== Step 3b: FNN Layer 1 Processing ==========
            hidden1 = self.fnn_layer1(fnn_input)  # [batch, hidden_dim]

            # ========== Step 3c: FNN Layer 2 Processing ==========
            hidden2 = self.fnn_layer2(hidden1)  # [batch, hidden_dim]

            # ========== Step 3d: Token Prediction (from final layer) ==========
            token_logits = self.token_output(hidden2)  # [batch, vocab_size]
            logits[:, t, :] = token_logits

            # ========== Step 3e: Context Update (from final layer) ==========
            context = self.context_updater(hidden2, context)  # [batch, context_dim]

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

                # FNN forward (2 layers)
                hidden1 = self.fnn_layer1(fnn_input)
                hidden2 = self.fnn_layer2(hidden1)

                # Get next token prediction
                next_token_logits = self.token_output(hidden2) / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Update context
                context = self.context_updater(hidden2, context)
                context = self.context_norm(context)
                context = torch.clamp(context, min=-10.0, max=10.0)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Truncate if needed
                if input_ids.size(1) > self.config.max_seq_length * 2:
                    input_ids = input_ids[:, -self.config.max_seq_length:]

        return input_ids
