"""
Diversity Loss Algorithms for CVFP Training

多様性損失アルゴリズム集。Phase 1訓練で使用。
"""

from typing import Callable, Dict

import torch
import torch.nn.functional as F


# =============================================================================
# 基本アルゴリズム
# =============================================================================

def mcdl_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    MCDL (Mean-Centered Dispersion Loss) - 現行アルゴリズム

    バッチ全体の平均（centroid）からの分散を最大化。
    計算コスト: O(n×d) - 非常に高速
    """
    context_mean = contexts.mean(dim=0)
    deviation = contexts - context_mean
    return -torch.norm(deviation, p=2) / len(contexts)


def odcm_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    ODCM (Off-Diagonal Covariance Minimization) - VICReg風

    VICRegの完全な実装:
    1. Variance Loss: 各次元の分散を1以上に維持
    2. Covariance Loss: 共分散行列の非対角成分を最小化

    計算コスト: O(n×d + d²)
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


def due_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    DUE (Dimension Usage Entropy) - 次元活性度均一化

    各次元の使用頻度のエントロピーを最大化（死んだ次元を防ぐ）
    計算コスト: O(n×d)

    損失が負: エントロピーが高いほど損失が小さい（最大化）
    """
    dim_activation = contexts.abs().mean(dim=0)
    dim_probs = dim_activation / (dim_activation.sum() + 1e-10)
    entropy = -(dim_probs * torch.log(dim_probs + 1e-10)).sum()
    max_entropy = torch.log(torch.tensor(contexts.size(1), dtype=torch.float32, device=contexts.device))
    # エントロピーを最大化するため、負の値を返す
    return -(entropy / max_entropy)


def ctm_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    CTM (Covariance Trace Maximization) - 統計的分散最大化

    共分散行列のトレースを最大化（総分散の最大化）
    計算コスト: O(n×d + d²)
    """
    centered = contexts - contexts.mean(dim=0)
    cov = (centered.T @ centered) / (len(contexts) - 1)
    return -torch.trace(cov) / contexts.size(1)


def udel_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    UDEL (Uniform Distribution Entropy Loss) - Barlow Twins風

    各次元の標準偏差を均一化しつつ、全体の分散を維持
    計算コスト: O(n×d)

    2つの目標:
    1. 各次元の標準偏差を均一に（分散項）
    2. 平均標準偏差を大きく保つ（崩壊防止）
    """
    std = contexts.std(dim=0)
    std_mean = std.mean()
    # 分散項: 各次元のstdが平均から離れないように
    std_var = ((std - std_mean) ** 2).mean()
    # 崩壊防止: 平均stdを大きく保つ（負の値で最大化）
    return std_var - std_mean


def sdl_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    SDL (Spectral Diversity Loss) - Effective Rank直接最大化

    SVD特異値のエントロピーを直接最大化
    計算コスト: O(n×d²) - 高コスト、参考用
    """
    _, S, _ = torch.svd(contexts)
    S_normalized = S / (S.sum() + 1e-10)
    entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
    return -entropy


# =============================================================================
# 追加アルゴリズム（自己教師あり学習文献より）
# =============================================================================

def unif_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    UNIF (Uniformity Loss) - 球面上の一様分布への誘導

    L2正規化後のペア間距離の指数和を最小化
    計算コスト: O(n²×d) - ペア計算
    参考: Understanding Contrastive Learning
    """
    normalized = F.normalize(contexts, dim=1)
    dists_sq = torch.cdist(normalized, normalized, p=2).pow(2)
    n = contexts.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=contexts.device)
    dists_sq = dists_sq[mask]
    return torch.log(torch.exp(-2 * dists_sq).mean() + 1e-10)


def decorr_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    DECORR (Decorrelation Loss) - 相関行列の対角化

    相関行列を単位行列に近づける
    計算コスト: O(n×d + d²)
    参考: Decorrelated Batch Normalization
    """
    std = contexts.std(dim=0)
    std = torch.clamp(std, min=1e-4)
    normalized = (contexts - contexts.mean(dim=0)) / std
    corr = (normalized.T @ normalized) / len(contexts)
    identity = torch.eye(contexts.size(1), device=contexts.device)
    return ((corr - identity) ** 2).mean()


def nuc_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    NUC (Nuclear Norm Maximization) - 核ノルム最大化

    特異値の和（核ノルム）を最大化
    計算コスト: O(n×d²) - SVD計算
    """
    centered = contexts - contexts.mean(dim=0)
    nuclear_norm = torch.linalg.matrix_norm(centered, ord='nuc')
    return -nuclear_norm / len(contexts)


def hsic_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    HSIC (Hilbert-Schmidt Independence Criterion) - 次元間独立性

    各次元ペア間のHSICを最小化（独立性促進）
    計算コスト: O(n²×d) - カーネル計算
    参考: HSIC Bottleneck
    """
    n = contexts.size(0)
    d = contexts.size(1)

    # サンプル数が多すぎる場合はサブサンプリング
    if n > 1000:
        indices = torch.randperm(n)[:1000]
        contexts = contexts[indices]
        n = 1000

    H = torch.eye(n, device=contexts.device) - torch.ones(n, n, device=contexts.device) / n
    total_hsic = torch.tensor(0.0, device=contexts.device)
    num_pairs = min(50, d * (d - 1) // 2)

    for _ in range(num_pairs):
        i, j = torch.randint(0, d, (2,))
        if i == j:
            continue
        xi = contexts[:, i:i+1]
        xj = contexts[:, j:j+1]
        Ki = torch.exp(-torch.cdist(xi, xi, p=2).pow(2))
        Kj = torch.exp(-torch.cdist(xj, xj, p=2).pow(2))
        hsic = torch.trace(Ki @ H @ Kj @ H) / ((n - 1) ** 2)
        total_hsic = total_hsic + hsic

    return total_hsic / num_pairs


def infonce_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    InfoNCE - 情報量最大化コントラスティブ損失

    自己とのみ正例、他は負例として扱うInfoNCE
    計算コスト: O(n²×d)
    参考: CPC, MoCo
    """
    normalized = F.normalize(contexts, dim=1)
    sim = (normalized @ normalized.T) / 0.1
    n = contexts.size(0)
    labels = torch.arange(n, device=contexts.device)
    return F.cross_entropy(sim, labels)


def wmse_loss(contexts: torch.Tensor) -> torch.Tensor:
    """
    W-MSE (Whitening MSE) - 白色化変換後のMSE

    ZCA白色化後の表現のMSEを最小化
    計算コスト: O(n×d + d³) - 固有値分解
    参考: W-MSE, Shuffled-DBN
    """
    centered = contexts - contexts.mean(dim=0)
    cov = (centered.T @ centered) / (len(contexts) - 1)
    cov = cov + 1e-4 * torch.eye(contexts.size(1), device=contexts.device)

    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    eigenvalues = torch.clamp(eigenvalues, min=1e-4)

    whitening = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.T
    whitened = centered @ whitening

    whitened_cov = (whitened.T @ whitened) / (len(contexts) - 1)
    identity = torch.eye(contexts.size(1), device=contexts.device)

    return ((whitened_cov - identity) ** 2).mean()


# =============================================================================
# アルゴリズム辞書
# =============================================================================

DIVERSITY_ALGORITHMS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    # 基本アルゴリズム
    'MCDL': mcdl_loss,
    'ODCM': odcm_loss,
    'DUE': due_loss,
    'CTM': ctm_loss,
    'UDEL': udel_loss,
    'SDL': sdl_loss,
    # 追加アルゴリズム
    'UNIF': unif_loss,
    'DECORR': decorr_loss,
    'NUC': nuc_loss,
    'HSIC': hsic_loss,
    'InfoNCE': infonce_loss,
    'WMSE': wmse_loss,
}

ALGORITHM_DESCRIPTIONS: Dict[str, str] = {
    'MCDL': 'Mean-Centered Dispersion Loss (現行)',
    'ODCM': 'Off-Diagonal Covariance Minimization (VICReg風)',
    'DUE': 'Dimension Usage Entropy (次元活性度均一化)',
    'CTM': 'Covariance Trace Maximization (統計的分散)',
    'UDEL': 'Uniform Distribution Entropy Loss (Barlow Twins風)',
    'SDL': 'Spectral Diversity Loss (ER直接最大化, 高コスト)',
    'UNIF': 'Uniformity Loss (球面一様分布)',
    'DECORR': 'Decorrelation Loss (相関行列対角化)',
    'NUC': 'Nuclear Norm Maximization (核ノルム最大化)',
    'HSIC': 'HSIC (次元間独立性)',
    'InfoNCE': 'InfoNCE (コントラスティブ)',
    'WMSE': 'Whitening MSE (白色化)',
}

# 高コストアルゴリズム（デフォルトでスキップ）
HIGH_COST_ALGORITHMS = {'SDL', 'UNIF', 'HSIC', 'InfoNCE'}
