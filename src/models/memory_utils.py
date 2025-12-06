"""
Memory Utilities for Infini-Attention Variants

圧縮メモリに関する共通機能:
- Linear Attention用のアクティベーション
- Causal Linear Attention
- メモリ状態管理（保存/復元）
"""

from typing import Any, Optional

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


class MemoryStateMixin:
    """
    メモリ状態管理のMixin

    get_memory_state()とset_memory_state()を提供。
    サブクラスでmemory_keysを定義して使用。

    使用例:
        class MyAttention(MemoryStateMixin, nn.Module):
            memory_keys = ["memories", "memory_norms"]

            def __init__(self):
                self.memories = None
                self.memory_norms = None
    """

    # サブクラスでオーバーライド
    memory_keys: list[str] = []

    def get_memory_state(self) -> dict:
        """
        メモリ状態を取得（CPU上のテンソルとして）

        Returns:
            dict: メモリ状態
        """
        state: dict[str, Any] = {}
        for key in self.memory_keys:
            value = getattr(self, key)
            if value is None:
                state[key] = None
            elif isinstance(value, list):
                state[key] = [t.cpu().clone() if t is not None else None for t in value]
            elif isinstance(value, torch.Tensor):
                state[key] = value.cpu().clone()
            else:
                state[key] = value
        return state

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        """
        メモリ状態を設定

        Args:
            state: get_memory_state()で取得した状態
            device: ターゲットデバイス（Noneの場合は自動検出）
        """
        if device is None:
            # 自動検出: 最初のパラメータのデバイスを使用
            for param in self.parameters():  # type: ignore
                device = param.device
                break

        for key in self.memory_keys:
            if key not in state:
                continue
            value = state[key]
            if value is None:
                setattr(self, key, None)
            elif isinstance(value, list):
                setattr(self, key, [t.to(device) if t is not None else None for t in value])
            elif isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
            else:
                setattr(self, key, value)


def create_memory_matrix(
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    メモリ行列を作成（ゼロ初期化）

    Args:
        dim: メモリ次元
        device: デバイス
        dtype: データ型

    Returns:
        [dim, dim] のゼロ行列
    """
    return torch.zeros(dim, dim, device=device, dtype=dtype)


def create_memory_norm(
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    メモリ正規化ベクトルを作成（ゼロ初期化）

    Args:
        dim: 次元
        device: デバイス
        dtype: データ型

    Returns:
        [dim] のゼロベクトル
    """
    return torch.zeros(dim, device=device, dtype=dtype)


def retrieve_from_memory(
    q: torch.Tensor,
    memory: torch.Tensor,
    memory_norm: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    メモリから情報を取得

    Args:
        q: Query [batch, seq, dim]
        memory: メモリ行列 [dim, dim]
        memory_norm: 正規化ベクトル [dim]
        eps: 数値安定性のための小さな値

    Returns:
        output: [batch, seq, dim]
    """
    sigma_q = elu_plus_one(q)

    if memory_norm.sum() < eps:
        return torch.zeros_like(q)

    # σ(Q) @ M: [B, S, D] @ [D, D] -> [B, S, D]
    a_mem_unnorm = torch.matmul(sigma_q, memory)
    # σ(Q) @ z: [B, S, D] @ [D] -> [B, S]
    norm = torch.matmul(sigma_q, memory_norm)
    norm = norm.clamp(min=eps).unsqueeze(-1)

    return a_mem_unnorm / norm


def update_memory_delta_rule(
    k: torch.Tensor,
    v: torch.Tensor,
    memory: torch.Tensor,
    memory_norm: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Delta Ruleでメモリを更新

    M_new = M + σ(K)^T @ (V - σ(K) @ M / σ(K) @ z)
    z_new = z + sum(σ(K))

    Args:
        k: Key [batch, seq, dim]
        v: Value [batch, seq, dim]
        memory: 現在のメモリ [dim, dim]
        memory_norm: 正規化ベクトル [dim]
        eps: 数値安定性のための小さな値

    Returns:
        (new_memory, new_memory_norm)
    """
    sigma_k = elu_plus_one(k)
    batch_size, seq_len, _ = k.shape

    # 現在のメモリから取得
    retrieved_unnorm = torch.matmul(sigma_k, memory)
    norm = torch.matmul(sigma_k, memory_norm)
    norm = norm.clamp(min=eps).unsqueeze(-1)
    retrieved = retrieved_unnorm / norm

    # Delta
    delta_v = v - retrieved

    # メモリ更新
    memory_update = torch.einsum('bsd,bse->de', sigma_k, delta_v)
    memory_update = memory_update / (batch_size * seq_len)
    new_memory = memory + memory_update

    # 正規化項の更新
    z_update = sigma_k.sum(dim=(0, 1)) / batch_size
    new_memory_norm = memory_norm + z_update

    return new_memory.detach(), new_memory_norm.detach()


def update_memory_simple(
    k: torch.Tensor,
    v: torch.Tensor,
    memory: torch.Tensor,
    memory_norm: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    シンプルなメモリ更新（Delta Ruleなし）

    M_new = M + σ(K)^T @ V
    z_new = z + sum(σ(K))

    Args:
        k: Key [batch, seq, dim]
        v: Value [batch, seq, dim]
        memory: 現在のメモリ [dim, dim]
        memory_norm: 正規化ベクトル [dim]

    Returns:
        (new_memory, new_memory_norm)
    """
    sigma_k = elu_plus_one(k)
    batch_size, seq_len, _ = k.shape

    # メモリ更新
    memory_update = torch.einsum('bsd,bse->de', sigma_k, v)
    memory_update = memory_update / (batch_size * seq_len)
    new_memory = memory + memory_update

    # 正規化項の更新
    z_update = sigma_k.sum(dim=(0, 1)) / batch_size
    new_memory_norm = memory_norm + z_update

    return new_memory.detach(), new_memory_norm.detach()
