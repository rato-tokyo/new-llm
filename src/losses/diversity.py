"""
Diversity Loss Algorithms for CVFP Training

多様性損失アルゴリズム集。Phase 1訓練で使用。

採用アルゴリズム（5種類）:
- MCDL: 現行ベースライン（最速）
- ODCM: VICReg風（推奨、低コスト・高ER）
- SDL: ER直接最大化（最高ER、高コスト）
- NUC: 核ノルム最大化（高ER、高コスト）
- WMSE: 白色化ベース（中コスト）
"""

from typing import Callable, Dict

import torch


# =============================================================================
# 採用アルゴリズム（5種類）
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


def sdl_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    SDL (Spectral Diversity Loss) - Effective Rank直接最大化

    SVD特異値のエントロピーを直接最大化。
    最高のER達成が可能だが、計算コストが高い。

    計算コスト: O(n×d²) - 高コスト
    """
    _, S, _ = torch.svd(contexts)
    S_normalized = S / (S.sum() + 1e-10)
    entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
    return -entropy


def nuc_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    NUC (Nuclear Norm Maximization) - 核ノルム最大化

    特異値の和（核ノルム）を最大化。
    SDLに次ぐ高ER達成が可能。

    計算コスト: O(n×d²) - 高コスト
    """
    centered = contexts - contexts.mean(dim=0)
    nuclear_norm = torch.linalg.matrix_norm(centered, ord='nuc')
    return -nuclear_norm / len(contexts)


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
    'SDL': sdl_loss,
    'NUC': nuc_loss,
    'WMSE': wmse_loss,
}

ALGORITHM_DESCRIPTIONS: Dict[str, str] = {
    'MCDL': 'Mean-Centered Dispersion Loss (現行ベースライン)',
    'ODCM': 'Off-Diagonal Covariance Minimization (VICReg風, 推奨)',
    'SDL': 'Spectral Diversity Loss (ER直接最大化, 高コスト)',
    'NUC': 'Nuclear Norm Maximization (核ノルム最大化, 高コスト)',
    'WMSE': 'Whitening MSE (白色化ベース)',
}

# 高コストアルゴリズム
HIGH_COST_ALGORITHMS = {'SDL', 'NUC'}
