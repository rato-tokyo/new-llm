"""Context Vector Convergence Loss for Repetition Training

This module implements a special loss function for training the context
update mechanism to reach a stable fixed point when processing repeated phrases.

Hypothesis: For a repeated phrase, context(n) ≈ context(n+1) when n is large.

Loss Types:
1. Context Stability Loss: MSE between consecutive repetition cycles
2. Context Identity Loss: MSE between context at t and context at t-cycle_length
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ContextConvergenceLoss(nn.Module):
    """Loss function for context vector convergence training

    This loss encourages the context vector to reach a stable fixed point
    when processing repeated patterns.

    Args:
        cycle_length: Length of the repeating phrase in tokens
        convergence_weight: Weight for convergence loss (default: 1.0)
        token_weight: Weight for token prediction loss (default: 0.0, disabled)
    """

    def __init__(
        self,
        cycle_length: int = 1,
        convergence_weight: float = 1.0,
        token_weight: float = 0.0
    ):
        super().__init__()
        self.cycle_length = cycle_length
        self.convergence_weight = convergence_weight
        self.token_weight = token_weight

    def forward(
        self,
        context_vectors: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute convergence loss

        Args:
            context_vectors: Context vectors at each timestep
                             [batch, seq_len, context_dim]
            logits: Token prediction logits (optional)
                    [batch, seq_len, vocab_size]
            targets: Target tokens (optional)
                     [batch, seq_len]

        Returns:
            loss: Total loss
            metrics: Dictionary of loss components
        """
        batch_size, seq_len, context_dim = context_vectors.shape

        # Context Convergence Loss
        # Compare context[t] with context[t - cycle_length]
        # For repeated phrases, these should be nearly identical
        if seq_len > self.cycle_length:
            # Get context vectors at t and t - cycle_length
            context_current = context_vectors[:, self.cycle_length:, :]
            context_previous = context_vectors[:, :-self.cycle_length, :]

            # MSE loss between corresponding positions
            convergence_loss = F.mse_loss(context_current, context_previous)
        else:
            # Not enough timesteps for convergence comparison
            convergence_loss = torch.tensor(0.0, device=context_vectors.device)

        # Token Prediction Loss (optional, usually disabled for this training)
        if self.token_weight > 0 and logits is not None and targets is not None:
            # Flatten for cross entropy
            logits_flat = logits[:, :-1, :].reshape(-1, logits.size(-1))
            targets_flat = targets[:, 1:].reshape(-1)

            # Cross entropy loss
            token_loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=0  # Ignore padding
            )
        else:
            token_loss = torch.tensor(0.0, device=context_vectors.device)

        # Total loss
        total_loss = (
            self.convergence_weight * convergence_loss +
            self.token_weight * token_loss
        )

        metrics = {
            'loss': total_loss.item(),
            'convergence_loss': convergence_loss.item(),
            'token_loss': token_loss.item(),
        }

        return total_loss, metrics


class ContextStabilityLoss(nn.Module):
    """Alternative: Context stability loss

    Measures how much the context vector changes between consecutive steps
    in a repeated pattern. Goal: minimize change.

    This is useful when we want the context to reach a stable state quickly.
    """

    def __init__(
        self,
        stability_weight: float = 1.0,
        token_weight: float = 0.0
    ):
        super().__init__()
        self.stability_weight = stability_weight
        self.token_weight = token_weight

    def forward(
        self,
        context_vectors: torch.Tensor,
        logits: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute stability loss

        Args:
            context_vectors: Context vectors at each timestep
                             [batch, seq_len, context_dim]
            logits: Token prediction logits (optional)
            targets: Target tokens (optional)

        Returns:
            loss: Total loss
            metrics: Dictionary of loss components
        """
        # Stability Loss: Minimize ||context[t] - context[t-1]||²
        context_current = context_vectors[:, 1:, :]
        context_previous = context_vectors[:, :-1, :]

        stability_loss = F.mse_loss(context_current, context_previous)

        # Token prediction loss (optional)
        if self.token_weight > 0 and logits is not None and targets is not None:
            logits_flat = logits[:, :-1, :].reshape(-1, logits.size(-1))
            targets_flat = targets[:, 1:].reshape(-1)
            token_loss = F.cross_entropy(
                logits_flat,
                targets_flat,
                ignore_index=0
            )
        else:
            token_loss = torch.tensor(0.0, device=context_vectors.device)

        total_loss = (
            self.stability_weight * stability_loss +
            self.token_weight * token_loss
        )

        metrics = {
            'loss': total_loss.item(),
            'stability_loss': stability_loss.item(),
            'token_loss': token_loss.item(),
        }

        return total_loss, metrics


def compute_context_change_rate(context_vectors: torch.Tensor) -> float:
    """Compute the average rate of change in context vectors

    Useful metric to monitor convergence during training.

    Args:
        context_vectors: [batch, seq_len, context_dim]

    Returns:
        Average L2 norm of context changes
    """
    context_current = context_vectors[:, 1:, :]
    context_previous = context_vectors[:, :-1, :]

    # L2 norm of change
    changes = torch.norm(context_current - context_previous, dim=-1)

    return changes.mean().item()


def compute_convergence_metric(
    context_vectors: torch.Tensor,
    cycle_length: int
) -> float:
    """Compute convergence metric for repeated patterns

    Measures how similar context[t] is to context[t - cycle_length]

    Args:
        context_vectors: [batch, seq_len, context_dim]
        cycle_length: Length of repeating cycle

    Returns:
        Average L2 distance between cyclic positions
    """
    seq_len = context_vectors.size(1)

    if seq_len <= cycle_length:
        return float('inf')

    context_current = context_vectors[:, cycle_length:, :]
    context_previous = context_vectors[:, :-cycle_length, :]

    # L2 distance
    distances = torch.norm(context_current - context_previous, dim=-1)

    return distances.mean().item()
