"""
Diversity Loss Algorithms for Phase 1 Training

多様性損失アルゴリズム集。Phase 1訓練で使用。

現在の採用アルゴリズム:
- OACD: 原点固定重心分散（標準採用）

新しいアルゴリズムを追加する場合:
1. このファイルに関数を追加
2. DIVERSITY_ALGORITHMS辞書に登録
3. FlexibleDiversityTrainerで使用可能
"""

from typing import Callable, Dict

import torch


# =============================================================================
# 多様性損失アルゴリズム
# =============================================================================

def oacd_loss(contexts: torch.Tensor, centroid_weight: float = 0.1) -> torch.Tensor:
    """
    OACD (Origin-Anchored Centroid Dispersion) - 原点固定重心分散【標準採用】

    2つの目標を組み合わせた多様性損失:
    1. 各点を重心から離散させる（分散最大化）
    2. 重心を原点に引き寄せる（安定した平衡点）

    これにより:
    - 「自己平衡」効果を維持（相対的目標）
    - 重心が原点に固定されることで、より安定した平衡点

    計算コスト: O(n×d) - 最速クラス

    Args:
        contexts: コンテキストベクトル [num_tokens, context_dim]
        centroid_weight: 重心→原点の引力の重み（デフォルト: 0.1）

    Returns:
        diversity_loss: 多様性損失（スカラー）
    """
    context_mean = contexts.mean(dim=0)
    deviation = contexts - context_mean

    # Term 1: 重心からの分散を最大化（負の損失で最大化）
    dispersion_loss = -torch.norm(deviation, p=2) / len(contexts)

    # Term 2: 重心を原点に引き寄せる
    centroid_loss = torch.norm(context_mean, p=2) ** 2

    return dispersion_loss + centroid_weight * centroid_loss


# =============================================================================
# アルゴリズム辞書（拡張用）
# =============================================================================

DIVERSITY_ALGORITHMS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'OACD': oacd_loss,
}

ALGORITHM_DESCRIPTIONS: Dict[str, str] = {
    'OACD': 'Origin-Anchored Centroid Dispersion (原点固定重心分散)',
}

# デフォルトアルゴリズム
DEFAULT_ALGORITHM = 'OACD'
