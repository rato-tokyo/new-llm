"""New-LLM: Layer-wise Context Update Architecture (LLM-like)

This architecture updates context vector at each layer, similar to Transformer.

Architecture B (Layer-wise):
    For each layer:
        1. Concatenate [token_embed, context]
        2. FNN processing
        3. Context update (each layer updates context)
        4. Pass updated context to next layer

This is more similar to Transformer where each layer processes and updates the representation.
"""

import torch
import torch.nn as nn


class NewLLMLayerwise(nn.Module):
    """
    New-LLM with layer-wise context update (Transformer-like)

    Each layer:
    - Takes [token, context] as input
    - Processes through FNN
    - Updates context
    - Passes updated context to next layer
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
                - num_layers: Number of layers
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

        # ========== Layer-wise FNN and Context Updaters ==========
        # Each layer has its own FNN and context updater
        self.fnn_layers = nn.ModuleList()
        self.context_delta_projs = nn.ModuleList()
        self.forget_gates = nn.ModuleList()
        self.input_gates = nn.ModuleList()
        self.context_norms = nn.ModuleList()

        for layer_idx in range(self.num_layers):
            # FNN for this layer: [token_embed + context] -> hidden_dim
            fnn_input_dim = self.embed_dim + self.context_dim
            fnn = nn.Sequential(
                nn.Linear(fnn_input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            )
            self.fnn_layers.append(fnn)

            # Context updater for this layer (gated)
            self.context_delta_projs.append(nn.Linear(self.hidden_dim, self.context_dim))
            self.forget_gates.append(nn.Linear(self.hidden_dim, self.context_dim))
            self.input_gates.append(nn.Linear(self.hidden_dim, self.context_dim))
            self.context_norms.append(nn.LayerNorm(self.context_dim))

        # ========== Output Heads (from final layer) ==========
        self.token_output = nn.Linear(self.hidden_dim, self.vocab_size)

        # Context reconstruction decoder (autoencoder)
        target_dim = self.context_dim + self.embed_dim
        self.context_decoder = nn.Sequential(
            nn.Linear(self.context_dim, target_dim),
            nn.ReLU(),
            nn.Linear(target_dim, target_dim),
        )

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
        Forward pass with layer-wise context update

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

        # Pre-allocate output tensors
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)

        if return_context_trajectory:
            context_trajectory = torch.zeros(batch_size, seq_len, self.context_dim, device=device)

        # Sequential processing
        for t in range(seq_len):
            # Current token
            current_token = token_embeds[:, t, :]  # [batch, embed_dim]

            # Initialize context for this token
            context = torch.zeros(batch_size, self.context_dim, device=device)

            # Layer-wise processing
            for layer_idx in range(self.num_layers):
                # Concatenate [token, context]
                fnn_input = torch.cat([current_token, context], dim=-1)

                # FNN processing
                hidden = self.fnn_layers[layer_idx](fnn_input)  # [batch, hidden_dim]

                # Context update (gated)
                context_delta = torch.tanh(self.context_delta_projs[layer_idx](hidden))
                forget = torch.sigmoid(self.forget_gates[layer_idx](hidden))
                input_g = torch.sigmoid(self.input_gates[layer_idx](hidden))

                context = forget * context + input_g * context_delta
                context = self.context_norms[layer_idx](context)
                context = torch.clamp(context, min=-10.0, max=10.0)

            # Token prediction (from final layer's hidden state)
            token_logits = self.token_output(hidden)
            logits[:, t, :] = token_logits

            if return_context_trajectory:
                context_trajectory[:, t, :] = context

        if return_context_trajectory:
            return logits, context_trajectory
        else:
            return logits

    def get_fixed_point_context(self, input_ids, max_iterations=100, tolerance=1e-4, warmup_iterations=10):
        """
        Compute fixed-point context vectors for each token

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
                    context_old = context.clone()

                    # Layer-wise processing
                    for layer_idx in range(self.num_layers):
                        fnn_input = torch.cat([current_token, context], dim=-1)
                        hidden = self.fnn_layers[layer_idx](fnn_input)

                        # Context update
                        context_delta = torch.tanh(self.context_delta_projs[layer_idx](hidden))
                        forget = torch.sigmoid(self.forget_gates[layer_idx](hidden))
                        input_g = torch.sigmoid(self.input_gates[layer_idx](hidden))

                        context = forget * context + input_g * context_delta
                        context = self.context_norms[layer_idx](context)
                        context = torch.clamp(context, min=-10.0, max=10.0)

                    # Only check convergence after warmup iterations (n)
                    if iteration >= warmup_iterations:
                        delta = torch.norm(context - context_old, dim=-1)  # [batch]

                        # Check if converged (element-wise for batch)
                        converged[:, t] = delta < tolerance

                        # If all in batch converged, break early
                        if converged[:, t].all():
                            num_iters[:, t] = iteration + 1
                            break

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
