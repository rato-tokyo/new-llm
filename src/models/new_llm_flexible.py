"""New-LLM: Flexible Architecture (Unified Model)

This unified architecture allows mixing Sequential and Layer-wise approaches:
- Sequential: Deep FNN stack without intermediate context updates
- Layer-wise: Context updates at each layer (Transformer-like)
- Mixed: Combination of both (e.g., 2 layer-wise blocks, each with 2 sequential layers)

Configuration is specified as a simple list:
- [4]: Pure Sequential (4 layers, no intermediate context update)
- [1, 1, 1, 1]: Pure Layer-wise (4 layers, context update after each)
- [2, 2]: Mixed (2 blocks of 2-layer sequential)
- [3, 1, 2]: Mixed (3-layer sequential, then update, then 1-layer, update, then 2-layer)

Benefits:
- Single model class for all architectures
- No config classes needed
- Easy experimentation with different structures
- Parameters specified as simple lists
"""

import torch
import torch.nn as nn


class NewLLMFlexible(nn.Module):
    """
    Unified New-LLM architecture with flexible layer structure

    Args:
        vocab_size: Vocabulary size
        embed_dim: Token embedding dimension
        context_dim: Context vector dimension
        hidden_dim: Hidden dimension for FNN layers
        layer_structure: List specifying FNN layers between context updates
                        Examples:
                        - [4]: Sequential (4 layers, 1 context update at end)
                        - [1, 1, 1, 1]: Layer-wise (4 layers, 4 context updates)
                        - [2, 2]: Mixed (2 blocks of 2-layer sequential)
        dropout: Dropout rate
    """

    def __init__(self, vocab_size, embed_dim, context_dim, hidden_dim,
                 layer_structure, dropout=0.1):
        super().__init__()

        # Store configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.layer_structure = layer_structure
        self.dropout = dropout

        # Total number of FNN layers
        self.total_layers = sum(layer_structure)
        # Number of context update points
        self.num_blocks = len(layer_structure)

        # ========== Token Embedding ==========
        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.embed_norm = nn.LayerNorm(self.embed_dim)

        # ========== FNN Blocks ==========
        # Build FNN blocks based on layer_structure
        self.fnn_blocks = nn.ModuleList()

        for block_idx, num_layers in enumerate(layer_structure):
            # Each block is a sequential FNN
            layers = []

            # First layer of block: input is [token_embed + context]
            input_dim = self.embed_dim + self.context_dim
            layers.append(nn.Linear(input_dim, self.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            # Additional layers in block: hidden_dim -> hidden_dim
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

            self.fnn_blocks.append(nn.Sequential(*layers))

        # ========== Context Updaters (one per block) ==========
        self.context_delta_projs = nn.ModuleList()
        self.forget_gates = nn.ModuleList()
        self.input_gates = nn.ModuleList()
        self.context_norms = nn.ModuleList()

        for _ in range(self.num_blocks):
            self.context_delta_projs.append(nn.Linear(self.hidden_dim, self.context_dim))
            self.forget_gates.append(nn.Linear(self.hidden_dim, self.context_dim))
            self.input_gates.append(nn.Linear(self.hidden_dim, self.context_dim))
            self.context_norms.append(nn.LayerNorm(self.context_dim))

        # ========== Output Heads ==========
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

    def _update_context_one_step(self, token_embed, context, return_hidden=False):
        """
        Update context for one iteration (shared between training and inference)

        Args:
            token_embed: Token embedding [batch, embed_dim]
            context: Current context [batch, context_dim]
            return_hidden: If True, return final hidden state

        Returns:
            context_new: Updated context [batch, context_dim]
            hidden: Final hidden state [batch, hidden_dim] (if return_hidden=True)
        """
        context_temp = context

        # Process through each FNN block
        for block_idx in range(self.num_blocks):
            # FNN input: [token_embed, current context]
            fnn_input = torch.cat([token_embed, context_temp], dim=-1)

            # Process through FNN block
            hidden = self.fnn_blocks[block_idx](fnn_input)

            # Update context using gated mechanism
            context_delta = torch.tanh(self.context_delta_projs[block_idx](hidden))
            forget = torch.sigmoid(self.forget_gates[block_idx](hidden))
            input_g = torch.sigmoid(self.input_gates[block_idx](hidden))

            context_temp = forget * context_temp + input_g * context_delta
            context_temp = self.context_norms[block_idx](context_temp)

        if return_hidden:
            return context_temp, hidden
        return context_temp

    def forward(self, input_ids, return_context_trajectory=False):
        """
        Forward pass with flexible architecture

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
        token_embeds = self.token_embedding(input_ids)
        token_embeds = self.embed_norm(token_embeds)

        # Pre-allocate output tensors
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)

        if return_context_trajectory:
            context_trajectory = torch.zeros(batch_size, seq_len, self.context_dim, device=device)

        # Sequential processing
        for t in range(seq_len):
            # Current token
            current_token = token_embeds[:, t, :]

            # Initialize context for this token
            context = torch.zeros(batch_size, self.context_dim, device=device)

            # Update context and get hidden state
            context, hidden = self._update_context_one_step(current_token, context, return_hidden=True)

            # Token prediction (from final hidden state)
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
            tolerance: Convergence threshold (MSE)
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

                    # Update context using shared method
                    context = self._update_context_one_step(current_token, context)

                    # Only check convergence after warmup iterations (n)
                    if iteration >= warmup_iterations:
                        # Use MSE for consistency with training
                        delta = torch.mean((context - context_old) ** 2, dim=-1)

                        # Check if converged (element-wise for batch)
                        converged[:, t] = delta < tolerance

                        # If all in batch converged, break early
                        if converged[:, t].all():
                            num_iters[:, t] = iteration + 1
                            break

                    num_iters[:, t] = iteration + 1

                # Progress logging every 100 tokens
                if (t + 1) % 100 == 0 or t == seq_len - 1:
                    converged_count = converged[:, :t+1].sum().item()
                    total_count = batch_size * (t + 1)
                    progress_pct = (t + 1) / seq_len * 100
                    print(f"  Token {t+1}/{seq_len} ({progress_pct:.1f}%) | Converged: {converged_count}/{total_count} ({converged_count/total_count*100:.1f}%)", end='\r')

                fixed_contexts[:, t, :] = context

        return fixed_contexts, converged, num_iters

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_architecture_description(self):
        """Return human-readable architecture description"""
        total = sum(self.layer_structure)
        blocks = len(self.layer_structure)

        if self.layer_structure == [total]:
            return f"Sequential ({total} layers)"
        elif all(x == 1 for x in self.layer_structure):
            return f"Layer-wise ({total} layers)"
        else:
            structure_str = '-'.join(map(str, self.layer_structure))
            return f"Mixed [{structure_str}] ({total} layers, {blocks} blocks)"


# Convenience functions for creating common architectures
def create_sequential_model(vocab_size, embed_dim, context_dim, hidden_dim,
                           num_layers, dropout=0.1):
    """Create pure Sequential architecture (single block)"""
    return NewLLMFlexible(vocab_size, embed_dim, context_dim, hidden_dim,
                         layer_structure=[num_layers], dropout=dropout)


def create_layerwise_model(vocab_size, embed_dim, context_dim, hidden_dim,
                          num_layers, dropout=0.1):
    """Create pure Layer-wise architecture (one layer per block)"""
    return NewLLMFlexible(vocab_size, embed_dim, context_dim, hidden_dim,
                         layer_structure=[1] * num_layers, dropout=dropout)


def create_mixed_model(vocab_size, embed_dim, context_dim, hidden_dim,
                      layer_structure, dropout=0.1):
    """Create mixed architecture with custom structure"""
    return NewLLMFlexible(vocab_size, embed_dim, context_dim, hidden_dim,
                         layer_structure=layer_structure, dropout=dropout)
