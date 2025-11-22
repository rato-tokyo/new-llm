"""New-LLM: Residual Connection Architecture (Standard Version)

This architecture uses ResNet-style residual connections for both context and token vectors.

Key Features:
1. FNN output is split into _context and _token (256 + 256 = 512)
2. Residual connections (Addition): context += _context, token += _token
3. Each layer updates both context and token vectors
4. Final prediction from updated token vector

Architecture (Standard Version):
- Layer 1-4: [context, token] → FNN → split → [_context, _token]
            context += _context
            token += _token
- Output: token → next_token prediction

This is fundamentally different from NewLLMGated which uses:
- LSTM-style gated updates
- No token vector updates
- Prediction from hidden state instead of token
"""

import torch
import torch.nn as nn


class NewLLMResidual(nn.Module):
    """
    New-LLM with ResNet-style residual connections

    Args:
        vocab_size: Vocabulary size
        embed_dim: Token embedding dimension
        context_dim: Context vector dimension
        hidden_dim: Hidden dimension for FNN layers (must be embed_dim + context_dim)
        layer_structure: List specifying FNN layers between context updates
        use_can: Enable Cell-wise Activity Normalization (CAN)
        can_momentum: Momentum for EMA in CAN (default: 0.9)
        can_eps: Epsilon for numerical stability in CAN (default: 1e-5)
    """

    def __init__(self, vocab_size, embed_dim, context_dim, hidden_dim,
                 layer_structure):
        super().__init__()

        # Validate dimensions
        if hidden_dim != embed_dim + context_dim:
            raise ValueError(f"hidden_dim ({hidden_dim}) must equal embed_dim ({embed_dim}) + context_dim ({context_dim})")

        # Store configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.layer_structure = layer_structure

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

            # First layer of block: input is [context + token]
            input_dim = self.context_dim + self.embed_dim
            layers.append(nn.Linear(input_dim, self.hidden_dim))
            layers.append(nn.ReLU())

            # Additional layers in block: hidden_dim -> hidden_dim
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(nn.ReLU())

            self.fnn_blocks.append(nn.Sequential(*layers))

        # ========== Layer Normalization (after each residual update) ==========
        self.context_norms = nn.ModuleList()
        self.token_norms = nn.ModuleList()

        for _ in range(self.num_blocks):
            self.context_norms.append(nn.LayerNorm(self.context_dim))
            self.token_norms.append(nn.LayerNorm(self.embed_dim))

        # ========== Output Head ==========
        # Predict from context vector (Phase 2 uses fixed contexts)
        self.token_output = nn.Linear(self.context_dim, self.vocab_size)

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

    def _update_context_one_step(self, token_vec, context, return_token=False):
        """
        Update context and token for one iteration

        Args:
            token_vec: Token vector [batch, embed_dim]
            context: Current context [batch, context_dim]
            return_token: If True, return updated token vector

        Returns:
            context_new: Updated context [batch, context_dim]
            token_new: Updated token vector [batch, embed_dim] (if return_token=True)
        """
        context_temp = context
        token_temp = token_vec

        # Process through each FNN block
        for block_idx in range(self.num_blocks):
            # FNN input: [context, token]
            fnn_input = torch.cat([context_temp, token_temp], dim=-1)

            # Process through FNN block
            y = self.fnn_blocks[block_idx](fnn_input)  # [batch, hidden_dim]

            # Split output into _context and _token
            _context = y[:, :self.context_dim]  # [batch, context_dim]
            _token = y[:, self.context_dim:]     # [batch, embed_dim]

            # Residual connections (Addition)
            context_temp = context_temp + _context
            token_temp = token_temp + _token

            # Layer normalization
            context_temp = self.context_norms[block_idx](context_temp)
            token_temp = self.token_norms[block_idx](token_temp)

        if return_token:
            return context_temp, token_temp
        return context_temp

    def forward(self, input_ids, return_context_trajectory=False):
        """
        Forward pass with residual connections

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

            # Update context and token
            context, token_updated = self._update_context_one_step(current_token, context, return_token=True)

            # Token prediction (from updated token vector)
            token_logits = self.token_output(token_updated)
            logits[:, t, :] = token_logits

            if return_context_trajectory:
                context_trajectory[:, t, :] = context

        if return_context_trajectory:
            return logits, context_trajectory
        return logits

    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_residual_layerwise(vocab_size, embed_dim, context_dim, hidden_dim,
                               num_layers=4):
    """
    Create a pure layer-wise model with residual connections

    Args:
        vocab_size: Vocabulary size
        embed_dim: Token embedding dimension
        context_dim: Context vector dimension
        hidden_dim: Hidden dimension (must be embed_dim + context_dim)
        num_layers: Number of layers (default: 4)

    Returns:
        NewLLMResidual model with layer-wise structure
    """
    layer_structure = [1] * num_layers
    return NewLLMResidual(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        layer_structure=layer_structure
    )
