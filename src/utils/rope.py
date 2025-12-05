"""
Rotary Position Embedding (RoPE) Utility

共通のRoPE実装。複数のモデルで再利用可能。
"""

from typing import Tuple

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding

    RoPEの共通実装。Query/Keyに回転行列を適用して相対位置情報を付与。

    Args:
        head_dim: アテンションヘッドの次元
        rotary_pct: 回転を適用する次元の割合 (default: 0.25)
        max_position_embeddings: 最大シーケンス長 (default: 2048)
        base: 周波数ベース (default: 10000)
    """

    def __init__(
        self,
        head_dim: int,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
        base: int = 10000,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = int(head_dim * rotary_pct)
        self.base = base

        # 周波数の逆数
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq)

        # キャッシュ初期化
        self._init_cache(max_position_embeddings)

    def _init_cache(self, max_len: int) -> None:
        """cos/sinキャッシュを初期化"""
        t = torch.arange(max_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
        self.max_seq_len_cached = max_len

    def _extend_cache(self, seq_len: int) -> None:
        """必要に応じてキャッシュを拡張"""
        if seq_len > self.max_seq_len_cached:
            self._init_cache(seq_len)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """回転行列の適用用ヘルパー"""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query/KeyにRoPEを適用

        Args:
            query: [batch, heads, seq, head_dim]
            key: [batch, heads, seq, head_dim]
            position_ids: [batch, seq]

        Returns:
            rotated_query, rotated_key
        """
        seq_len = int(position_ids.max().item()) + 1
        self._extend_cache(seq_len)

        # position_idsに対応するcos/sinを取得
        cos = self.cos_cached[position_ids].unsqueeze(1)  # [batch, 1, seq, rotary_dim]
        sin = self.sin_cached[position_ids].unsqueeze(1)

        # 回転部分と非回転部分に分割
        q_rot, q_pass = query[..., : self.rotary_dim], query[..., self.rotary_dim :]
        k_rot, k_pass = key[..., : self.rotary_dim], key[..., self.rotary_dim :]

        # 回転を適用
        q_rot = q_rot * cos + self.rotate_half(q_rot) * sin
        k_rot = k_rot * cos + self.rotate_half(k_rot) * sin

        # 結合して返す
        return (
            torch.cat([q_rot, q_pass], dim=-1),
            torch.cat([k_rot, k_pass], dim=-1),
        )


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RoPEを適用する関数版（キャッシュなし）

    Args:
        query: [batch, heads, seq, head_dim]
        key: [batch, heads, seq, head_dim]
        cos: [batch, 1, seq, rotary_dim]
        sin: [batch, 1, seq, rotary_dim]
        rotary_dim: 回転を適用する次元数

    Returns:
        rotated_query, rotated_key
    """
    q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin

    return (
        torch.cat([q_rot, q_pass], dim=-1),
        torch.cat([k_rot, k_pass], dim=-1),
    )
