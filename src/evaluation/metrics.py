"""
Evaluation metrics for New-LLM

Fixed-point analysis, effective rank calculation, and other metrics.
"""

from typing import Any, Dict

import torch

from src.utils.io import print_flush
from src.utils.device import clear_gpu_cache


def compute_effective_rank(contexts: torch.Tensor, max_samples: int = 10000) -> float:
    """
    Effective Rank計算（SVDベース）

    コンテキストテンソルからEffective Rankを計算する共通関数。
    analyze_fixed_points() と MemoryPhase1Trainer._compute_effective_rank() で使用。

    Args:
        contexts: コンテキストテンソル [num_tokens, context_dim]
        max_samples: SVD計算用の最大サンプル数（メモリ効率化）

    Returns:
        effective_rank: Effective Rank値（0 〜 context_dim）
    """
    device = contexts.device
    num_tokens = contexts.shape[0]

    # サンプリング（大規模データの場合）
    if num_tokens > max_samples:
        svd_indices = torch.randperm(num_tokens, device=device)[:max_samples]
        svd_contexts = contexts[svd_indices]
    else:
        svd_contexts = contexts

    # SVDで特異値を計算
    _, S, _ = torch.svd(svd_contexts)

    # Effective rank (entropy-based)
    S_normalized = S / S.sum()
    entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()

    return float(torch.exp(entropy).item())


def analyze_fixed_points(
    contexts: torch.Tensor, label: str = "", verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze fixed-point contexts for quality metrics.

    Effective Rank（SVDベース）のみ計算。

    Args:
        contexts: Fixed-point contexts [num_tokens, context_dim]
        label: Label for display (e.g., "Train", "Val")
        verbose: If True, print detailed analysis

    Returns:
        dict: Analysis metrics including effective rank
    """
    device = contexts.device
    num_tokens = contexts.shape[0]
    context_dim = contexts.shape[1]

    # Effective Rank計算（共通関数を使用）
    effective_rank = compute_effective_rank(contexts)

    # SVDの特異値を取得（返り値に含めるため）
    max_svd_samples = 10000
    if num_tokens > max_svd_samples:
        svd_indices = torch.randperm(num_tokens, device=device)[:max_svd_samples]
        svd_contexts = contexts[svd_indices]
    else:
        svd_contexts = contexts

    _, S, _ = torch.svd(svd_contexts)
    actual_rank = (S > 1e-6).sum().item()

    # Free SVD memory
    del svd_contexts
    clear_gpu_cache(device)

    if verbose:
        er_ratio = effective_rank / context_dim * 100
        print_flush(f"  {label} Effective Rank: {effective_rank:.2f}/{context_dim} ({er_ratio:.1f}%)")

    return {
        "actual_rank": actual_rank,
        "effective_rank": effective_rank,
        "singular_values": S.tolist()
    }
