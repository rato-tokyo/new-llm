"""
CVFP Layer Module

Encapsulated layer for Context Vector Fixed-Point (CVFP) learning
with built-in distribution regularization using Exponential Moving Average (EMA).
"""

import torch
import torch.nn as nn


class CVFPLayer(nn.Module):
    """
    Context update layer with built-in distribution tracking.

    This layer encapsulates:
    1. Context update via FNN (Feedforward Neural Network)
    2. Token embedding integration
    3. Residual connections
    4. Exponential Moving Average (EMA) statistics for distribution regularization

    Args:
        context_dim: Dimension of context vector
        embed_dim: Dimension of token embedding
        hidden_dim: Hidden dimension (must equal context_dim + embed_dim)
        use_dist_reg: Enable distribution regularization (default: True)
        ema_momentum: Momentum for EMA statistics (default: 0.99)
        layernorm_mix: Mixing ratio for LayerNorm (0.0 = disabled, 1.0 = full)
    """

    def __init__(
        self,
        context_dim,
        embed_dim,
        hidden_dim,
        use_dist_reg=True,
        ema_momentum=0.99,
        layernorm_mix=0.0  # Disabled by default
    ):
        super().__init__()

        # Validate dimensions
        if hidden_dim != context_dim + embed_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal "
                f"context_dim ({context_dim}) + embed_dim ({embed_dim})"
            )

        # Store configuration
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.use_dist_reg = use_dist_reg
        self.ema_momentum = ema_momentum
        self.layernorm_mix = layernorm_mix

        # FNN: [context + token] -> [hidden_dim]
        self.fnn = nn.Sequential(
            nn.Linear(context_dim + embed_dim, hidden_dim),
            nn.ReLU()
        )

        # Optional LayerNorm
        if layernorm_mix > 0:
            self.context_norm = nn.LayerNorm(context_dim)
            self.token_norm = nn.LayerNorm(embed_dim)

        # EMA statistics for distribution regularization
        if use_dist_reg:
            # Running mean and variance per dimension
            self.register_buffer('running_mean', torch.zeros(context_dim))
            self.register_buffer('running_var', torch.ones(context_dim))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use larger initialization to prevent identity mapping
                nn.init.normal_(module.weight, mean=0.0, std=0.05)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, context, token_embed):
        """
        Forward pass: update context and token

        Args:
            context: Current context [batch, context_dim]
            token_embed: Token embedding [batch, embed_dim]

        Returns:
            new_context: Updated context [batch, context_dim]
            new_token: Updated token embedding [batch, embed_dim]
        """
        # Concatenate inputs
        fnn_input = torch.cat([context, token_embed], dim=-1)

        # FNN forward
        fnn_output = self.fnn(fnn_input)

        # Split output
        delta_context = fnn_output[:, :self.context_dim]
        delta_token = fnn_output[:, self.context_dim:]

        # Residual connections
        new_context = context + delta_context
        new_token = token_embed + delta_token

        # Optional LayerNorm mixing
        if self.layernorm_mix > 0:
            context_normed = self.context_norm(new_context)
            token_normed = self.token_norm(new_token)

            mix = self.layernorm_mix
            new_context = (1 - mix) * new_context + mix * context_normed
            new_token = (1 - mix) * new_token + mix * token_normed

        # Update running statistics (training mode only)
        if self.training and self.use_dist_reg:
            self._update_running_stats(new_context)

        return new_context, new_token

    def _update_running_stats(self, context):
        """
        Update running mean and variance using EMA

        This is called automatically during forward pass in training mode.
        Hidden from external callers for clean encapsulation.

        Args:
            context: Current batch of contexts [batch, context_dim]
        """
        with torch.no_grad():
            # Compute batch statistics
            batch_mean = context.mean(dim=0)  # [context_dim]
            batch_var = context.var(dim=0, unbiased=False)  # [context_dim]

            # EMA update
            momentum = self.ema_momentum
            self.running_mean = momentum * self.running_mean + (1 - momentum) * batch_mean
            self.running_var = momentum * self.running_var + (1 - momentum) * batch_var

            # Track number of updates
            self.num_batches_tracked += 1

    def get_distribution_loss(self):
        """
        Calculate distribution regularization loss

        Goal: Each dimension should follow N(0, 1)
        Loss = mean_penalty + variance_penalty

        Returns:
            dist_loss: Scalar tensor representing distribution loss
        """
        if not self.use_dist_reg:
            return torch.tensor(0.0, device=self.running_mean.device)

        # Penalize deviation from N(0, 1)
        mean_penalty = (self.running_mean ** 2).mean()
        var_penalty = ((self.running_var - 1.0) ** 2).mean()

        return mean_penalty + var_penalty

    def reset_running_stats(self):
        """Reset running statistics (useful for new training runs)"""
        if self.use_dist_reg:
            self.running_mean.zero_()
            self.running_var.fill_(1.0)
            self.num_batches_tracked.zero_()

    def extra_repr(self):
        """String representation for debugging"""
        return (
            f'context_dim={self.context_dim}, '
            f'embed_dim={self.embed_dim}, '
            f'hidden_dim={self.hidden_dim}, '
            f'use_dist_reg={self.use_dist_reg}, '
            f'ema_momentum={self.ema_momentum}, '
            f'layernorm_mix={self.layernorm_mix}'
        )


class CVFPBlock(nn.Module):
    """
    Multi-layer CVFP block consisting of multiple CVFPLayer instances

    Args:
        num_layers: Number of CVFP layers in this block
        context_dim: Dimension of context vector
        embed_dim: Dimension of token embedding
        hidden_dim: Hidden dimension for each layer
        use_dist_reg: Enable distribution regularization
        ema_momentum: Momentum for EMA
        layernorm_mix: LayerNorm mixing ratio
    """

    def __init__(
        self,
        num_layers,
        context_dim,
        embed_dim,
        hidden_dim,
        use_dist_reg=True,
        ema_momentum=0.99,
        layernorm_mix=0.0
    ):
        super().__init__()

        self.num_layers = num_layers

        # Create stack of CVFP layers
        self.layers = nn.ModuleList([
            CVFPLayer(
                context_dim=context_dim,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                use_dist_reg=use_dist_reg,
                ema_momentum=ema_momentum,
                layernorm_mix=layernorm_mix
            )
            for _ in range(num_layers)
        ])

    def forward(self, context, token_embed):
        """
        Forward through all layers in block

        Args:
            context: [batch, context_dim]
            token_embed: [batch, embed_dim]

        Returns:
            context: Updated context [batch, context_dim]
            token_embed: Updated token [batch, embed_dim]
        """
        for layer in self.layers:
            context, token_embed = layer(context, token_embed)

        return context, token_embed

    def get_distribution_loss(self):
        """Aggregate distribution loss from all layers"""
        total_loss = 0.0
        for layer in self.layers:
            total_loss += layer.get_distribution_loss()
        return total_loss / len(self.layers)  # Average across layers

    def reset_running_stats(self):
        """Reset statistics for all layers"""
        for layer in self.layers:
            layer.reset_running_stats()
