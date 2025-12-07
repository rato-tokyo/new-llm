"""
Evaluation utilities for experiments.

共通の評価関数を提供する。

PPL評価方法:
1. Sliding Window (推奨): HuggingFace標準方式
2. Segment-based: セグメント分割方式（Infini-Attention訓練用）
3. Document-based: ドキュメント単位でメモリリセット
"""

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.io import print_flush


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
                logits, _ = model(input_ids, return_reconstruction_loss=True)
            else:
                output = model(input_ids)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output

            # Next-token prediction: logits[:-1] -> labels[1:]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Compute per-position loss (positions are 0-indexed on shifted sequence)
            for start, end in position_ranges:
                key = f"{start}-{end}"

                # Adjust for shifted sequence length
                adj_end = min(end - 1, shift_logits.size(1))
                adj_start = max(0, start - 1) if start > 0 else 0

                if adj_start >= adj_end:
                    continue

                range_logits = shift_logits[:, adj_start:adj_end, :]
                range_labels = shift_labels[:, adj_start:adj_end]

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


def evaluate_reversal_curse(
    model: nn.Module,
    tokenizer: Any,
    pairs: List[Dict[str, str]],
    device: torch.device,
    max_length: int = 64,
) -> Dict[str, float]:
    """
    Evaluate Reversal Curse.

    順方向（"A is B"）と逆方向（"B is A"）のPPLを比較し、
    モデルが双方向の知識を持っているかを評価する。

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        pairs: List of dicts with 'forward' and 'backward' keys
        device: Device
        max_length: Maximum sequence length

    Returns:
        Dict with forward_ppl, backward_ppl, reversal_ratio, etc.
    """
    model.eval()

    forward_losses: List[float] = []
    backward_losses: List[float] = []

    with torch.no_grad():
        for pair in pairs:
            # Forward direction
            forward_loss = _compute_sentence_loss(
                model, tokenizer, pair["forward"], device, max_length
            )
            forward_losses.append(forward_loss)

            # Backward direction
            backward_loss = _compute_sentence_loss(
                model, tokenizer, pair["backward"], device, max_length
            )
            backward_losses.append(backward_loss)

    # Compute average PPL
    avg_forward_loss = sum(forward_losses) / len(forward_losses)
    avg_backward_loss = sum(backward_losses) / len(backward_losses)

    forward_ppl = torch.exp(torch.tensor(avg_forward_loss)).item()
    backward_ppl = torch.exp(torch.tensor(avg_backward_loss)).item()

    # Reversal Ratio: closer to 1.0 = less reversal curse
    # < 1.0 means backward is harder (typical reversal curse)
    reversal_ratio = forward_ppl / backward_ppl if backward_ppl > 0 else float("inf")

    # Reversal Gap: difference in PPL
    reversal_gap = backward_ppl - forward_ppl

    return {
        "forward_ppl": forward_ppl,
        "backward_ppl": backward_ppl,
        "reversal_ratio": reversal_ratio,
        "reversal_gap": reversal_gap,
        "num_pairs": len(pairs),
    }


def _compute_sentence_loss(
    model: nn.Module,
    tokenizer: Any,
    sentence: str,
    device: torch.device,
    max_length: int,
) -> float:
    """
    Compute average cross-entropy loss for a single sentence.

    Args:
        model: Model
        tokenizer: Tokenizer
        sentence: Input sentence
        device: Device
        max_length: Max length

    Returns:
        Average loss per token
    """
    # Tokenize
    tokens = tokenizer.encode(sentence, add_special_tokens=False)

    # Truncate if needed
    if len(tokens) > max_length - 1:
        tokens = tokens[: max_length - 1]

    # Create input and label
    input_ids = torch.tensor([tokens[:-1]], device=device)
    labels = torch.tensor([tokens[1:]], device=device)

    if input_ids.size(1) == 0:
        return 0.0

    # Forward pass
    output = model(input_ids)
    if isinstance(output, tuple):
        logits = output[0]
    else:
        logits = output

    # Compute loss
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        reduction="mean",
    )

    return loss.item()


# =============================================================================
# Token-level PPL Evaluation Methods
# =============================================================================


def evaluate_ppl_sliding_window(
    model: nn.Module,
    tokens: torch.Tensor,
    device: torch.device,
    context_length: int = 2048,
    stride: int = 512,
    verbose: bool = True,
) -> float:
    """
    Sliding window方式でPPL評価（HuggingFace推奨）

    Args:
        model: 評価するモデル（HuggingFace形式）
        tokens: 1D tensor of tokens
        device: デバイス
        context_length: コンテキスト長
        stride: スライド幅
        verbose: 詳細出力

    Returns:
        PPL値
    """
    model.eval()

    total_loss = 0.0
    total_tokens: int = 0
    num_windows = 0

    tokens = tokens.to(device)
    seq_len = len(tokens)

    with torch.no_grad():
        for start in range(0, seq_len - 1, stride):
            end = min(start + context_length, seq_len)
            input_ids = tokens[start:end].unsqueeze(0)

            # ターゲットの開始位置（strideより前はコンテキスト）
            target_start = min(stride, end - start - 1)

            if target_start <= 0:
                continue

            # labels: 最初のtarget_start個は-100（無視）、残りは予測対象
            labels = input_ids.clone()
            labels[0, :target_start] = -100

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            # 実際に計算されたトークン数
            num_target_tokens = int((labels != -100).sum().item())
            if num_target_tokens > 0:
                total_loss += loss.item() * num_target_tokens
                total_tokens += num_target_tokens
                num_windows += 1

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()

    if verbose:
        print_flush(f"  Windows evaluated: {num_windows}")
        print_flush(f"  Tokens evaluated: {total_tokens:,}")

    return ppl


def evaluate_ppl_segment(
    model: nn.Module,
    tokens: torch.Tensor,
    device: torch.device,
    segment_length: int = 256,
    reset_memory_fn: Optional[Callable] = None,
) -> float:
    """
    セグメント分割方式でPPL評価

    Infini-Attention訓練時と同じ評価方法。
    メモリ付きモデルの場合はreset_memory_fnを指定。

    Args:
        model: 評価するモデル
        tokens: 1D tensor of tokens
        device: デバイス
        segment_length: セグメント長
        reset_memory_fn: メモリリセット関数（オプション）

    Returns:
        PPL値
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    tokens = tokens.to(device)
    seq_len = len(tokens)

    # メモリリセット（存在する場合）
    if reset_memory_fn is not None:
        reset_memory_fn()

    with torch.no_grad():
        for start in range(0, seq_len - 1, segment_length):
            end = min(start + segment_length, seq_len)
            segment = tokens[start:end]

            if len(segment) < 2:
                continue

            input_ids = segment[:-1].unsqueeze(0)
            labels = segment[1:].unsqueeze(0)

            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return ppl
