"""
Diversity Loss Algorithms for CVFP Training

多様性損失アルゴリズム集。Phase 1訓練で使用。

採用アルゴリズム（3種類）:
- MCDL: 現行ベースライン（最速、唯一CVFPなしでも収束）
- ODCM: VICReg風（推奨、低コスト・高ER）
- OACD: 原点固定重心分散（MCDLの拡張、より安定した平衡点）
"""

from typing import Callable, Dict

import torch


# =============================================================================
# 採用アルゴリズム（3種類）
# =============================================================================

def mcdl_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    MCDL (Mean-Centered Dispersion Loss) - 現行ベースライン

    バッチ全体の平均（centroid）からの分散を最大化。
    計算コスト: O(n×d) - 最速
    """
    context_mean = contexts.mean(dim=0)
    deviation = contexts - context_mean
    return -torch.norm(deviation, p=2) / len(contexts)


def odcm_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    ODCM (Off-Diagonal Covariance Minimization) - VICReg風【推奨】

    VICRegの完全な実装:
    1. Variance Loss: 各次元の分散を1以上に維持
    2. Covariance Loss: 共分散行列の非対角成分を最小化

    計算コスト: O(n×d + d²) - 低コスト
    """
    centered = contexts - contexts.mean(dim=0)

    # Variance Loss: 各次元の標準偏差を1以上に
    std = torch.sqrt(centered.var(dim=0) + 1e-4)
    var_loss = torch.relu(1.0 - std).mean()

    # Covariance Loss: 非対角成分を最小化
    cov = (centered.T @ centered) / (len(contexts) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / contexts.size(1)

    return var_loss + 0.04 * cov_loss


def oacd_loss(contexts: torch.Tensor, centroid_weight: float = 0.1) -> torch.Tensor:
    """
    OACD (Origin-Anchored Centroid Dispersion) - 原点固定重心分散

    MCDLの拡張版:
    1. 各点を重心から離散させる（MCDL同様）
    2. 重心を原点に引き寄せる（新規）

    これにより:
    - MCDLの「自己平衡」効果を維持
    - 重心が原点に固定されることで、より安定した平衡点が期待できる

    計算コスト: O(n×d) - MCDLと同等
    """
    context_mean = contexts.mean(dim=0)
    deviation = contexts - context_mean

    # Term 1: 重心からの分散を最大化（MCDL）
    dispersion_loss = -torch.norm(deviation, p=2) / len(contexts)

    # Term 2: 重心を原点に引き寄せる
    centroid_loss = torch.norm(context_mean, p=2) ** 2

    return dispersion_loss + centroid_weight * centroid_loss


def wmse_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    W-MSE (Whitening MSE) - 白色化ベースの多様性損失

    目標:
    1. 各次元の分散を1に近づける（Variance Loss）
    2. 次元間の相関を0に近づける（Covariance Loss）

    ODCMと似ているが、共分散行列全体を単位行列に近づける点が異なる。

    計算コスト: O(n×d + d²)
    参考: W-MSE, VICReg
    """
    centered = contexts - contexts.mean(dim=0)
    n = len(contexts)
    d = contexts.size(1)

    # 共分散行列を計算
    cov = (centered.T @ centered) / (n - 1)

    # Variance Loss: 対角成分（分散）を1に近づける
    # 分散が1より小さいと損失が発生（崩壊防止）
    variances = torch.diag(cov)
    std = torch.sqrt(variances + 1e-4)
    var_loss = torch.relu(1.0 - std).mean()

    # Covariance Loss: 非対角成分（共分散）を0に近づける
    # 単位行列との差を最小化
    identity = torch.eye(d, device=contexts.device)
    cov_loss = ((cov - identity) ** 2).mean()

    # 分散維持を優先（var_loss）、共分散最小化を補助（cov_loss）
    return var_loss + 0.04 * cov_loss


# =============================================================================
# アルゴリズム辞書
# =============================================================================

DIVERSITY_ALGORITHMS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    'MCDL': mcdl_loss,
    'ODCM': odcm_loss,
    'OACD': oacd_loss,
    'WMSE': wmse_loss,
}

ALGORITHM_DESCRIPTIONS: Dict[str, str] = {
    'MCDL': 'Mean-Centered Dispersion Loss (現行ベースライン)',
    'ODCM': 'Off-Diagonal Covariance Minimization (VICReg風, 推奨)',
    'OACD': 'Origin-Anchored Centroid Dispersion (原点固定重心分散)',
    'WMSE': 'Whitening MSE (白色化ベース)',
}
