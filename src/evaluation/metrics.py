"""Evaluation metrics for language models"""

import torch
import torch.nn.functional as F
import math


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Compute cross-entropy loss ignoring padding tokens

    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        targets: Target token indices [batch_size, seq_len]
        pad_idx: Padding token index to ignore

    Returns:
        loss: Scalar loss value
    """
    # Flatten for cross entropy
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)

    # Compute loss with ignore_index for padding
    loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=pad_idx)
    return loss


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from loss

    Args:
        loss: Cross-entropy loss value

    Returns:
        perplexity: Perplexity score
    """
    return math.exp(loss)


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int = 0) -> float:
    """
    Compute token-level accuracy ignoring padding

    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        targets: Target token indices [batch_size, seq_len]
        pad_idx: Padding token index to ignore

    Returns:
        accuracy: Percentage of correct predictions
    """
    predictions = torch.argmax(logits, dim=-1)
    mask = (targets != pad_idx)
    correct = ((predictions == targets) & mask).sum().item()
    total = mask.sum().item()

    return correct / total if total > 0 else 0.0
