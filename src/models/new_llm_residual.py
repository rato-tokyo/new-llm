"""New-LLM: Residual Connection Architecture (Refactored Version 2)

Clean implementation using CVFPLayer for better encapsulation.

Key Improvements:
1. Uses CVFPLayer/CVFPBlock from layers.py
2. Distribution regularization handled internally
3. Cleaner forward pass
4. Better separation of concerns
"""

import torch
import torch.nn as nn
from .cvfp import CVFPBlock


class NewLLMResidual(nn.Module):
    """
    New-LLM with ResNet-style residual connections

    This version uses CVFPLayer for clean encapsulation of:
    - Context updates
    - Token updates
    - Distribution regularization (EMA-based)

    Args:
        vocab_size: Vocabulary size
        embed_dim: Token embedding dimension
        context_dim: Context vector dimension
        hidden_dim: Hidden dimension (must be embed_dim + context_dim)
        layer_structure: List specifying number of layers per block
        use_dist_reg: Enable distribution regularization (default: True)
        ema_momentum: EMA momentum for running stats (default: 0.99)
        layernorm_mix: LayerNorm mixing ratio, 0.0=disabled (default: 0.0)
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        context_dim,
        hidden_dim,
        layer_structure,
        use_dist_reg=True,
        ema_momentum=0.99,
        layernorm_mix=0.0
    ):
        super().__init__()

        # Validate dimensions
        if hidden_dim != embed_dim + context_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal "
                f"embed_dim ({embed_dim}) + context_dim ({context_dim})"
            )

        # Store configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.layer_structure = layer_structure
        self.use_dist_reg = use_dist_reg

        # ========== Token Embedding ==========
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.embed_norm = nn.LayerNorm(embed_dim)

        # ========== CVFP Blocks ==========
        self.blocks = nn.ModuleList([
            CVFPBlock(
                num_layers=num_layers,
                context_dim=context_dim,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                use_dist_reg=use_dist_reg,
                ema_momentum=ema_momentum,
                layernorm_mix=layernorm_mix
            )
            for num_layers in layer_structure
        ])

        # ========== Output Head ==========
        # Predict next token from context
        self.token_output = nn.Linear(context_dim, vocab_size)

        # Initialize embeddings
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding weights"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    def _update_context_one_step(self, token_vec, context, return_token=False):
        """
        Update context for one token step

        Args:
            token_vec: Token vector [batch, embed_dim]
            context: Current context [batch, context_dim]
            return_token: If True, also return updated token

        Returns:
            new_context: Updated context [batch, context_dim]
            new_token: Updated token (if return_token=True)
        """
        current_context = context
        current_token = token_vec

        # Process through all blocks
        for block in self.blocks:
            current_context, current_token = block(current_context, current_token)

        if return_token:
            return current_context, current_token
        return current_context

    def forward(self, input_ids, return_context_trajectory=False):
        """
        Forward pass through the model

        Args:
            input_ids: Input token IDs [batch, seq_len]
            return_context_trajectory: If True, return all intermediate contexts

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            context_trajectory: (optional) All contexts [batch, seq_len, context_dim]
        """
        batch_size, seq_len = input_ids.shape

        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch, seq_len, embed_dim]
        token_embeds = self.embed_norm(token_embeds)

        # Initialize context
        context = torch.zeros(
            batch_size, self.context_dim,
            device=input_ids.device,
            dtype=token_embeds.dtype
        )

        # Process sequence
        contexts = []
        for t in range(seq_len):
            token_vec = token_embeds[:, t, :]  # [batch, embed_dim]
            context = self._update_context_one_step(token_vec, context)
            contexts.append(context)

        # Stack contexts
        all_contexts = torch.stack(contexts, dim=1)  # [batch, seq_len, context_dim]

        # Predict next tokens
        logits = self.token_output(all_contexts)  # [batch, seq_len, vocab_size]

        if return_context_trajectory:
            return logits, all_contexts
        return logits

    def get_distribution_loss(self):
        """
        Get aggregated distribution regularization loss from all blocks

        This method provides clean access to internal statistics
        without exposing implementation details.

        Returns:
            dist_loss: Scalar tensor
        """
        if not self.use_dist_reg:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        total_loss = 0.0
        for block in self.blocks:
            total_loss += block.get_distribution_loss()

        return total_loss / len(self.blocks)  # Average across blocks

    def reset_running_stats(self):
        """Reset all running statistics (for new training runs)"""
        for block in self.blocks:
            block.reset_running_stats()
