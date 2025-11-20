"""New-LLM: Simplified Flexible Architecture for Dialogue Training

This is a simplified implementation focused on:
- Variable number of layers (easy to change)
- Variable context vector dimensions (easy to change)
- Two-phase training: Phase 1 (context), Phase 2 (token prediction)
- UltraChat dialogue dataset support
- Efficient caching

Architecture:
    1. Token Embedding: input_ids → token_embeds
    2. Context Vector: Fixed-size state vector
    3. FNN Layers: Configurable depth (1, 2, 3, ... layers)
    4. Output Heads: Token prediction + Context update
"""

import torch
import torch.nn as nn


class NewLLMFlexible(nn.Module):
    """
    Simplified New-LLM with flexible architecture

    Key features:
    - Easy to change num_layers (just modify config)
    - Easy to change context_dim (just modify config)
    - Two-phase training support
    - Gated context updater (prevents global attractor)
    """

    def __init__(self, config):
        """
        Initialize model

        Args:
            config: Configuration with:
                - vocab_size: Vocabulary size
                - embed_dim: Token embedding dimension
                - context_dim: Context vector dimension
                - hidden_dim: Hidden dimension for FNN layers
                - num_layers: Number of FNN layers
                - dropout: Dropout rate
        """
        super().__init__()
        self.config = config

        # Extract config
        self.vocab_size = config.vocab_size
        self.embed_dim = config.embed_dim
        self.context_dim = config.context_dim
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers

        # ========== Token Embedding ==========
        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embed_norm = nn.LayerNorm(self.embed_dim)

        # ========== FNN Layers ==========
        # Input: [token_embed, context] concatenated
        fnn_input_dim = self.embed_dim + self.context_dim

        # Build layers dynamically based on num_layers
        layers = []

        # First layer: fnn_input_dim -> hidden_dim
        layers.append(nn.Linear(fnn_input_dim, self.hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(config.dropout))

        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(self.num_layers - 1):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))

        self.fnn = nn.Sequential(*layers)

        # ========== Output Heads ==========
        # Token prediction head
        self.token_output = nn.Linear(self.hidden_dim, self.vocab_size)

        # Context reconstruction decoder (autoencoder)
        # Reconstructs [prev_context + current_token_embed] from context
        target_dim = self.context_dim + self.embed_dim
        self.context_decoder = nn.Sequential(
            nn.Linear(self.context_dim, target_dim),
            nn.ReLU(),
            nn.Linear(target_dim, target_dim),
        )

        # ========== Context Updater (Gated) ==========
        # LSTM-style gated update to prevent global attractor problem
        self.context_delta_proj = nn.Linear(self.hidden_dim, self.context_dim)
        self.forget_gate = nn.Linear(self.hidden_dim, self.context_dim)
        self.input_gate = nn.Linear(self.hidden_dim, self.context_dim)
        self.context_norm = nn.LayerNorm(self.context_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, return_context_trajectory=False):
        """
        Forward pass

        Args:
            input_ids: Token IDs [batch, seq_len]
            return_context_trajectory: If True, return all context vectors

        Returns:
            logits: Next token predictions [batch, seq_len, vocab_size]
            context_trajectory: Context vectors [batch, seq_len, context_dim] (if requested)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embedding
        token_embeds = self.token_embedding(input_ids)  # [batch, seq_len, embed_dim]
        token_embeds = self.embed_norm(token_embeds)

        # Initialize context vector
        context = torch.zeros(batch_size, self.context_dim, device=device)

        # Pre-allocate output tensors
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)

        if return_context_trajectory:
            context_trajectory = torch.zeros(batch_size, seq_len, self.context_dim, device=device)

        # Sequential processing
        for t in range(seq_len):
            # Current token
            current_token = token_embeds[:, t, :]  # [batch, embed_dim]

            # Concatenate [token, context]
            fnn_input = torch.cat([current_token, context], dim=-1)

            # FNN processing
            hidden = self.fnn(fnn_input)  # [batch, hidden_dim]

            # Token prediction
            token_logits = self.token_output(hidden)
            logits[:, t, :] = token_logits

            # Context update (gated)
            context_delta = torch.tanh(self.context_delta_proj(hidden))
            forget = torch.sigmoid(self.forget_gate(hidden))
            input_g = torch.sigmoid(self.input_gate(hidden))

            context = forget * context + input_g * context_delta
            context = self.context_norm(context)
            context = torch.clamp(context, min=-10.0, max=10.0)

            if return_context_trajectory:
                context_trajectory[:, t, :] = context

        if return_context_trajectory:
            return logits, context_trajectory
        else:
            return logits

    def get_fixed_point_context(self, input_ids, max_iterations=100, tolerance=1e-4, warmup_iterations=10):
        """
        Compute fixed-point context vectors for each token

        This is used in Phase 1 training where we repeatedly process
        the same token until context converges.

        Args:
            input_ids: Token IDs [batch, seq_len]
            max_iterations: Maximum iterations for fixed-point search
            tolerance: Convergence threshold (L2 distance)
            warmup_iterations: Number of warmup iterations before checking convergence (n)

        Returns:
            fixed_contexts: Fixed-point contexts [batch, seq_len, context_dim]
            converged: Convergence flags [batch, seq_len]
            num_iters: Number of iterations needed [batch, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        token_embeds = self.embed_norm(token_embeds)

        # Storage
        fixed_contexts = torch.zeros(batch_size, seq_len, self.context_dim, device=device)
        converged = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        num_iters = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

        with torch.no_grad():
            for t in range(seq_len):
                current_token = token_embeds[:, t, :]

                # Initialize context
                context = torch.zeros(batch_size, self.context_dim, device=device)

                # Fixed-point iteration with warmup
                for iteration in range(max_iterations):
                    fnn_input = torch.cat([current_token, context], dim=-1)
                    hidden = self.fnn(fnn_input)

                    # Update context
                    context_delta = torch.tanh(self.context_delta_proj(hidden))
                    forget = torch.sigmoid(self.forget_gate(hidden))
                    input_g = torch.sigmoid(self.input_gate(hidden))

                    context_new = forget * context + input_g * context_delta
                    context_new = self.context_norm(context_new)
                    context_new = torch.clamp(context_new, min=-10.0, max=10.0)

                    # Only check convergence after warmup iterations (n)
                    if iteration >= warmup_iterations:
                        delta = torch.norm(context_new - context, dim=-1)  # [batch]

                        # Check if converged (element-wise for batch)
                        converged[:, t] = delta < tolerance

                        # If all in batch converged, break early
                        if converged[:, t].all():
                            num_iters[:, t] = iteration + 1
                            context = context_new
                            break

                    context = context_new
                    num_iters[:, t] = iteration + 1

                # Progress logging every 10 tokens
                if (t + 1) % 10 == 0 or t == seq_len - 1:
                    converged_count = converged[:, :t+1].sum().item()
                    total_count = batch_size * (t + 1)
                    progress_pct = (t + 1) / seq_len * 100
                    print(f"  Token {t+1}/{seq_len} ({progress_pct:.1f}%) | Converged: {converged_count}/{total_count} ({converged_count/total_count*100:.1f}%)", end='\r')

                fixed_contexts[:, t, :] = context

        return fixed_contexts, converged, num_iters

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
