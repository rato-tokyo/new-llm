"""
Position Encoding Implementations

位置エンコーディングの実装:
- RoPE (Rotary Position Embedding)
- 将来: ALiBi, YaRN等
"""

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)

    "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    https://arxiv.org/abs/2104.09864

    Pythia-70mでは隠れ層の25%にRoPEを適用 (rotary_pct=0.25)
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        """
        Args:
            dim: 回転を適用する次元（通常はhead_dim * rotary_pct）
            max_position_embeddings: 最大シーケンス長
            base: 周波数の基底
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 逆周波数を計算
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # cosとsinのキャッシュ
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        """cosとsinのキャッシュを設定"""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 入力テンソル（形状は使用されない、dtype取得用）
            seq_len: シーケンス長

        Returns:
            (cos, sin): 位置エンコーディング [1, 1, seq_len, dim]
        """
        if seq_len > self.max_position_embeddings:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    テンソルの後半を回転

    [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Query/KeyにRotary Position Embeddingを適用

    Args:
        q: Query [batch, num_heads, seq_len, head_dim]
        k: Key [batch, num_heads, seq_len, head_dim]
        cos: cos位置エンコーディング [1, 1, seq_len, rotary_dim]
        sin: sin位置エンコーディング [1, 1, seq_len, rotary_dim]

    Returns:
        (q_embed, k_embed): RoPE適用後のQuery/Key
    """
    # RoPEは次元の一部にのみ適用
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # RoPE適用
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # 非回転部分と結合
    q_embed = torch.cat((q_embed, q_pass), dim=-1)
    k_embed = torch.cat((k_embed, k_pass), dim=-1)

    return q_embed, k_embed
