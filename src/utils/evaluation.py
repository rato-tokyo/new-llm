"""
Evaluation utilities for experiments.

共通の評価関数を提供する。
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_ppl(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    return_recon_loss: bool = False,
) -> Union[float, Dict[str, float]]:
    """
    Evaluate model perplexity.

    Args:
        model: Model to evaluate (must return (logits, recon_loss) if return_recon_loss=True)
        val_loader: Validation data loader
        device: Device
        return_recon_loss: Whether to return reconstruction loss

    Returns:
        If return_recon_loss=False: PPL as float
        If return_recon_loss=True: Dict with 'ppl' and 'recon_loss'
    """
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_tokens = 0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            if return_recon_loss:
                logits, recon_loss = model(input_ids, return_reconstruction_loss=True)
            else:
                output = model(input_ids)
                # Handle both (logits,) tuple and logits tensor
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                recon_loss = None

            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )

            total_loss += loss.item()
            total_tokens += labels.numel()

            if recon_loss is not None:
                total_recon_loss += recon_loss.item()
                num_batches += 1

    avg_loss = total_loss / total_tokens
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    if return_recon_loss:
        avg_recon = total_recon_loss / num_batches if num_batches > 0 else 0.0
        return {"ppl": ppl, "recon_loss": avg_recon}
    else:
        return ppl


def evaluate_position_wise_ppl(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    position_ranges: Optional[list] = None,
    return_recon_loss: bool = False,
) -> Dict[str, float]:
    """
    Evaluate position-wise perplexity.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device
        position_ranges: List of (start, end) tuples for position ranges.
                        If None, uses default ranges for seq_len.
        return_recon_loss: Whether model returns reconstruction loss

    Returns:
        Dictionary with position range keys and PPL values
    """
    model.eval()

    # Get sequence length from first batch
    first_batch = next(iter(val_loader))
    seq_len = first_batch[0].shape[1]

    # Default position ranges
    if position_ranges is None:
        position_ranges = get_default_position_ranges(seq_len)

    # Initialize accumulators for each range
    range_losses: Dict[str, float] = {}
    range_tokens: Dict[str, int] = {}
    for start, end in position_ranges:
        key = f"{start}-{end}"
        range_losses[key] = 0.0
        range_tokens[key] = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            if return_recon_loss:
                logits, _ = model(input_ids, return_reconstruction_loss=False)
            else:
                output = model(input_ids)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output

            # Compute per-position loss
            for start, end in position_ranges:
                key = f"{start}-{end}"

                range_logits = logits[:, start:end, :]
                range_labels = labels[:, start:end]

                loss = nn.functional.cross_entropy(
                    range_logits.reshape(-1, range_logits.size(-1)),
                    range_labels.reshape(-1),
                    reduction="sum",
                )

                range_losses[key] += loss.item()
                range_tokens[key] += range_labels.numel()

    # Compute PPL for each range
    results: Dict[str, float] = {}
    for key in range_losses:
        if range_tokens[key] > 0:
            avg_loss = range_losses[key] / range_tokens[key]
            ppl = torch.exp(torch.tensor(avg_loss)).item()
            results[key] = ppl
        else:
            results[key] = float("inf")

    return results


def get_default_position_ranges(seq_len: int) -> list:
    """
    Get default position ranges for position-wise PPL evaluation.

    Args:
        seq_len: Sequence length

    Returns:
        List of (start, end) tuples
    """
    return [
        (0, 16),
        (16, 32),
        (32, 64),
        (64, 96),
        (96, seq_len),
    ]
