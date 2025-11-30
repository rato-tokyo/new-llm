"""
Evaluation metrics for New-LLM

Fixed-point analysis, effective rank calculation, and other metrics.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F  # check_identity_mapping で使用

from src.utils.io import print_flush
from src.utils.device import clear_gpu_cache


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

    # Effective Rank計算（SVD）
    max_svd_samples = 10000
    if num_tokens > max_svd_samples:
        svd_indices = torch.randperm(num_tokens, device=device)[:max_svd_samples]
        svd_contexts = contexts[svd_indices]
    else:
        svd_contexts = contexts

    # Compute SVD
    U, S, V = torch.svd(svd_contexts)

    # Actual rank
    actual_rank = (S > 1e-6).sum().item()

    # Effective rank (entropy-based)
    S_normalized = S / S.sum()
    entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
    effective_rank = torch.exp(entropy).item()

    # Free SVD memory
    del U, V, svd_contexts
    clear_gpu_cache(device)

    if verbose:
        er_ratio = effective_rank / context_dim * 100
        print_flush(f"  {label} Effective Rank: {effective_rank:.2f}/{context_dim} ({er_ratio:.1f}%)")

    return {
        "actual_rank": actual_rank,
        "effective_rank": effective_rank,
        "singular_values": S.tolist()
    }


def check_identity_mapping(
    model: Any,
    token_embeds: torch.Tensor,
    contexts: torch.Tensor,
    device: torch.device
) -> Dict[str, Any]:
    """
    恒等写像チェック: 学習が起きているかを確認

    ランダム初期化モデルと訓練済みモデルの出力を比較し、
    学習が実際に起きているかを確認。

    Args:
        model: 訓練済みモデル
        token_embeds: トークン埋め込み [num_tokens, embed_dim]
        contexts: 学習されたコンテキスト [num_tokens, context_dim]
        device: torch device

    Returns:
        dict: 恒等写像チェック結果
    """
    print_flush("\n" + "="*70)
    print_flush("IDENTITY MAPPING CHECK (恒等写像チェック)")
    print_flush("="*70 + "\n")

    model.eval()

    # 学習されたコンテキストのノルムを確認
    learned_norms = torch.norm(contexts, dim=1)
    avg_learned_norm = learned_norms.mean().item()
    std_learned_norm = learned_norms.std().item()

    print_flush("1. Learned Context Statistics:")
    print_flush(f"   Average Norm: {avg_learned_norm:.6f}")
    print_flush(f"   Std Dev: {std_learned_norm:.6f}")

    # ゼロコンテキストとの差分
    zero_distance = avg_learned_norm

    if zero_distance < 0.1:
        print_flush("   ❌ DEGENERATE: Contexts are near zero")
        print_flush("   → Model is NOT learning meaningful context updates")
        is_identity = True
    else:
        print_flush("   ✅ PASSED: Contexts are non-zero")

        # さらに詳細なチェック：コンテキストの多様性
        # 全コンテキストが同じ値か確認
        with torch.no_grad():
            context_mean = contexts.mean(dim=0)  # [context_dim]
            deviations = contexts - context_mean.unsqueeze(0)
            avg_deviation = torch.norm(deviations, dim=1).mean().item()

        print_flush("\n2. Context Diversity:")
        print_flush(f"   Average deviation from mean: {avg_deviation:.6f}")

        if avg_deviation < 0.1:
            print_flush("   ❌ DEGENERATE: All contexts are identical (global attractor)")
            is_identity = True
        else:
            print_flush("   ✅ PASSED: Contexts are diverse")
            is_identity = False

    # トークン埋め込みとの類似度（恒等写像確認）
    # コンテキストとトークン埋め込みの次元が異なる場合の対処
    if contexts.shape[1] == token_embeds.shape[1]:
        embed_similarity = F.cosine_similarity(contexts, token_embeds, dim=1).mean().item()

        print_flush("\n3. Context vs Token Embedding Similarity:")
        print_flush(f"   Cosine Similarity: {embed_similarity:.6f}")

        if embed_similarity > 0.95:
            print_flush("   ⚠️ WARNING: Contexts too similar to token embeddings")
            print_flush("   → Possible identity mapping (no transformation)")
        else:
            print_flush("   ✅ PASSED: Contexts are transformed from embeddings")
    else:
        embed_similarity = None
        print_flush("\n3. Context vs Token Embedding Similarity:")
        print_flush(f"   (Skipped: dimension mismatch {contexts.shape[1]} vs {token_embeds.shape[1]})")

    print_flush("="*70 + "\n")

    return {
        "context_diff_from_zero": avg_deviation if not is_identity else 0.0,
        "embed_similarity": embed_similarity,
        "is_identity": is_identity
    }


