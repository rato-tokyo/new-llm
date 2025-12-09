"""
Memory Utilities for Linear Attention

Linear Attention用の共通ユーティリティ:
- ELU + 1 アクティベーション
- Causal Linear Attention
"""

import torch
import torch.nn.functional as F


def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """
    ELU + 1 activation (ensures positivity for linear attention)

    Linear Attentionでは非負の値が必要なため、ELU + 1を使用。
    """
    return F.elu(x) + 1.0


def causal_linear_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Causal Linear Attention (O(n) complexity)

    累積和を使用して効率的に計算。
    各位置iは位置0~iのK,Vにのみアテンション可能。

    Args:
        q: Query tensor [batch, seq, dim] or [batch, heads, seq, dim]
        k: Key tensor (same shape as q)
        v: Value tensor (same shape as q)
        eps: 数値安定性のための小さな値

    Returns:
        output: 同じ形状のテンソル
    """
    sigma_q = elu_plus_one(q)
    sigma_k = elu_plus_one(k)

    # 入力の次元数に応じて処理を分岐
    if q.dim() == 3:
        # [B, S, D] - シングルヘッド
        kv = torch.einsum('bsd,bse->bsde', sigma_k, v)
        kv_cumsum = torch.cumsum(kv, dim=1)
        k_cumsum = torch.cumsum(sigma_k, dim=1)

        numerator = torch.einsum('bsd,bsde->bse', sigma_q, kv_cumsum)
        denominator = torch.einsum('bsd,bsd->bs', sigma_q, k_cumsum)
        denominator = denominator.clamp(min=eps).unsqueeze(-1)
    else:
        # [B, H, S, D] - マルチヘッド
        kv = torch.einsum('bhsd,bhse->bhsde', sigma_k, v)
        kv_cumsum = torch.cumsum(kv, dim=2)
        k_cumsum = torch.cumsum(sigma_k, dim=2)

        numerator = torch.einsum('bhsd,bhsde->bhse', sigma_q, kv_cumsum)
        denominator = torch.einsum('bhsd,bhsd->bhs', sigma_q, k_cumsum)
        denominator = denominator.clamp(min=eps).unsqueeze(-1)

    return numerator / denominator
