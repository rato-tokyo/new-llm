"""
Position Encoding Module

位置エンコーディングの統一インターフェース。
RoPE, ALiBi, NoPE（なし）を簡単に切り替え可能。

Usage:
    # 設定で切り替え
    config = PositionEncodingConfig(type="rope", rotary_pct=0.25)
    config = PositionEncodingConfig(type="alibi", alibi_slope=0.0625)
    config = PositionEncodingConfig(type="none")

    # Attention内で使用
    pos_enc = create_position_encoding(config, hidden_size, num_heads)
    attn_weights = pos_enc.apply(query, key, attn_weights, seq_len)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class PositionEncodingType(Enum):
    """位置エンコーディングの種類"""
    NONE = "none"     # 位置情報なし
    ROPE = "rope"     # Rotary Position Embedding (2D)
    ROPE3D = "rope3d" # Rotary Position Embedding (3D)
    ALIBI = "alibi"   # Attention with Linear Biases


@dataclass
class PositionEncodingConfig:
    """位置エンコーディングの設定"""
    type: str = "rope"  # "none", "rope", "rope3d", "alibi"

    # RoPE parameters
    rotary_pct: float = 0.25  # RoPEを適用する次元の割合
    rope_base: int = 10000    # RoPE base frequency

    # RoPE3D parameters
    rope3d_pct: float = 0.25  # RoPE3Dを適用する次元の割合

    # ALiBi parameters
    alibi_slope: float = 0.0625  # ALiBi slope (unified)

    # Common
    max_position_embeddings: int = 2048

    def __post_init__(self) -> None:
        # Validate type
        valid_types = [e.value for e in PositionEncodingType]
        if self.type not in valid_types:
            raise ValueError(f"Invalid position encoding type: {self.type}. Must be one of {valid_types}")


class PositionEncoding(ABC, nn.Module):
    """位置エンコーディングの抽象基底クラス"""

    @abstractmethod
    def apply_to_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query/Keyに位置エンコーディングを適用

        Args:
            query: [batch, heads, seq, head_dim]
            key: [batch, heads, seq, head_dim]
            seq_len: シーケンス長

        Returns:
            query, key: 位置情報が適用されたQ, K
        """
        pass

    @abstractmethod
    def apply_to_scores(
        self,
        attn_scores: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Attentionスコアに位置バイアスを適用

        Args:
            attn_scores: [batch, heads, seq, seq]
            seq_len: シーケンス長

        Returns:
            attn_scores: バイアスが適用されたスコア
        """
        pass

    @property
    @abstractmethod
    def modifies_qk(self) -> bool:
        """Q/Kを変更するかどうか（RoPE: True, ALiBi: False）"""
        pass

    @property
    @abstractmethod
    def adds_score_bias(self) -> bool:
        """スコアにバイアスを加えるか（RoPE: False, ALiBi: True）"""
        pass


class NoPositionEncoding(PositionEncoding):
    """位置エンコーディングなし"""

    def __init__(self) -> None:
        super().__init__()

    def apply_to_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return query, key

    def apply_to_scores(
        self,
        attn_scores: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        # Causal maskのみ適用
        device = attn_scores.device
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1
        )
        return attn_scores + causal_mask

    @property
    def modifies_qk(self) -> bool:
        return False

    @property
    def adds_score_bias(self) -> bool:
        return False


class RotaryPositionEncoding(PositionEncoding):
    """Rotary Position Embedding (RoPE)"""

    def __init__(
        self,
        head_dim: int,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
        base: int = 10000,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = int(head_dim * rotary_pct)
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for sin/cos
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_to_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len)

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)

        # Split into rotary and pass-through parts
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim:]

        # Apply rotary
        query_rot = (query_rot * cos) + (self._rotate_half(query_rot) * sin)
        key_rot = (key_rot * cos) + (self._rotate_half(key_rot) * sin)

        # Concatenate back
        query = torch.cat([query_rot, query_pass], dim=-1)
        key = torch.cat([key_rot, key_pass], dim=-1)

        return query, key

    def apply_to_scores(
        self,
        attn_scores: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        # Causal maskのみ（RoPEはQ/Kに適用済み）
        device = attn_scores.device
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1
        )
        return attn_scores + causal_mask

    @property
    def modifies_qk(self) -> bool:
        return True

    @property
    def adds_score_bias(self) -> bool:
        return False


class ALiBiPositionEncoding(PositionEncoding):
    """Attention with Linear Biases (ALiBi)"""

    def __init__(self, slope: float = 0.0625) -> None:
        super().__init__()
        self.slope = slope
        self._cache: Dict[int, torch.Tensor] = {}

    def _build_alibi_bias_causal(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Causal ALiBiバイアス行列を生成"""
        positions = torch.arange(seq_len, device=device)

        # 相対位置: i - j (query位置 - key位置)
        relative_pos = positions.unsqueeze(1) - positions.unsqueeze(0)

        # 因果マスク付きALiBiバイアス
        alibi_bias = torch.where(
            relative_pos >= 0,
            -self.slope * relative_pos.float(),
            torch.tensor(float("-inf"), device=device),
        )

        return alibi_bias

    def _get_bias(self, seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """キャッシュからALiBiバイアスを取得"""
        if seq_len not in self._cache:
            self._cache[seq_len] = self._build_alibi_bias_causal(seq_len, device)

        cached = self._cache[seq_len]

        if device is not None and cached.device != device:
            cached = cached.to(device)
            self._cache[seq_len] = cached

        return cached

    def apply_to_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # ALiBiはQ/Kを変更しない
        return query, key

    def apply_to_scores(
        self,
        attn_scores: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        device = attn_scores.device
        alibi_bias = self._get_bias(seq_len, device)
        return attn_scores + alibi_bias

    @property
    def modifies_qk(self) -> bool:
        return False

    @property
    def adds_score_bias(self) -> bool:
        return True


class Rotary3DPositionEncoding(PositionEncoding):
    """
    3D Rotary Position Embedding (RoPE3D)

    3次元回転を位置エンコーディングに使用。
    回転軸は (1,2,3)/√14 の非対称な斜め方向で、各次元への影響が異なる。

    ロドリゲスの回転公式を使用:
    R(θ) = I + sin(θ) * K + (1 - cos(θ)) * K²

    ここで K は回転軸の外積行列。
    """

    def __init__(
        self,
        head_dim: int,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
        base: int = 10000,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        # 3の倍数に調整（3次元グループ）
        rotary_dim_raw = int(head_dim * rotary_pct)
        self.num_groups = rotary_dim_raw // 3
        self.rotary_dim = self.num_groups * 3  # 3の倍数
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 回転軸: (1, 2, 3) / √14（正規化）
        # 非対称な軸で各次元への影響が異なる
        axis = torch.tensor([1.0, 2.0, 3.0])
        axis = axis / axis.norm()
        self.register_buffer("rotation_axis", axis)

        # 外積行列 K を事前計算
        # K = [[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]]
        K = torch.zeros(3, 3)
        K[0, 1] = -axis[2]
        K[0, 2] = axis[1]
        K[1, 0] = axis[2]
        K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]
        K[2, 1] = axis[0]
        self.register_buffer("K_matrix", K)

        # K² を事前計算
        K2 = torch.matmul(K, K)
        self.register_buffer("K2_matrix", K2)

        # 各3次元グループの周波数（RoPE同様に指数的に減少）
        inv_freq = 1.0 / (base ** (torch.arange(0, self.num_groups).float() / self.num_groups))
        self.register_buffer("inv_freq", inv_freq)

        # 回転行列のキャッシュ
        self._set_rotation_cache(max_position_embeddings)

    def _compute_rotation_matrix(self, angle: torch.Tensor) -> torch.Tensor:
        """
        ロドリゲスの回転公式で3x3回転行列を計算

        R(θ) = I + sin(θ) * K + (1 - cos(θ)) * K²

        Args:
            angle: 回転角 [num_groups] or scalar

        Returns:
            rotation_matrix: [num_groups, 3, 3] or [3, 3]
        """
        if angle.dim() == 0:
            angle = angle.unsqueeze(0)

        sin_angle = torch.sin(angle)  # [num_groups]
        cos_angle = torch.cos(angle)  # [num_groups]

        # I + sin(θ) * K + (1 - cos(θ)) * K²
        # Broadcasting: [num_groups, 1, 1] * [3, 3]
        identity = torch.eye(3, device=angle.device, dtype=angle.dtype)
        R = (
            identity.unsqueeze(0)
            + sin_angle.view(-1, 1, 1) * self.K_matrix.unsqueeze(0)
            + (1 - cos_angle).view(-1, 1, 1) * self.K2_matrix.unsqueeze(0)
        )
        return R  # [num_groups, 3, 3]

    def _set_rotation_cache(self, seq_len: int) -> None:
        """各位置の回転行列をキャッシュ"""
        self.max_seq_len_cached = seq_len

        # positions: [seq_len]
        positions = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        # angles: [seq_len, num_groups] = positions @ inv_freq
        angles = torch.outer(positions, self.inv_freq)  # [seq_len, num_groups]

        # 各位置、各グループの回転行列を計算
        # [seq_len, num_groups, 3, 3]
        rotation_matrices = []
        for pos in range(seq_len):
            R = self._compute_rotation_matrix(angles[pos])  # [num_groups, 3, 3]
            rotation_matrices.append(R.unsqueeze(0))

        rotation_cache = torch.cat(rotation_matrices, dim=0)  # [seq_len, num_groups, 3, 3]
        self.register_buffer("rotation_cache", rotation_cache)

    def _apply_rotation(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        3D回転を適用

        Args:
            x: [batch, heads, seq, head_dim]
            seq_len: シーケンス長

        Returns:
            rotated: [batch, heads, seq, head_dim]
        """
        if seq_len > self.max_seq_len_cached:
            self._set_rotation_cache(seq_len)

        batch, heads, seq, dim = x.shape

        # rotary部分とpass-through部分に分割
        x_rot = x[..., :self.rotary_dim]  # [batch, heads, seq, rotary_dim]
        x_pass = x[..., self.rotary_dim:]

        # 3次元グループに reshape
        # [batch, heads, seq, num_groups, 3]
        x_grouped = x_rot.view(batch, heads, seq, self.num_groups, 3)

        # 回転行列を適用
        # rotation_cache: [seq_len, num_groups, 3, 3]
        # x_grouped: [batch, heads, seq, num_groups, 3]
        # 結果: [batch, heads, seq, num_groups, 3]

        # einsum: 各位置のグループに対応する回転行列を適用
        # "s g i j, b h s g j -> b h s g i"
        rotated = torch.einsum(
            "sgij,bhsgj->bhsgi",
            self.rotation_cache[:seq_len],
            x_grouped
        )

        # 元の形状に戻す（einsumの結果は非連続の場合があるため contiguous() が必要）
        rotated = rotated.contiguous().view(batch, heads, seq, self.rotary_dim)

        # pass-through部分と結合
        return torch.cat([rotated, x_pass], dim=-1)

    def apply_to_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Q, K両方に3D回転を適用"""
        query_rotated = self._apply_rotation(query, seq_len)
        key_rotated = self._apply_rotation(key, seq_len)
        return query_rotated, key_rotated

    def apply_to_scores(
        self,
        attn_scores: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Causal maskのみ適用"""
        device = attn_scores.device
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float("-inf"),
            diagonal=1
        )
        return attn_scores + causal_mask

    @property
    def modifies_qk(self) -> bool:
        return True

    @property
    def adds_score_bias(self) -> bool:
        return False


def create_position_encoding(
    config: PositionEncodingConfig,
    head_dim: int,
) -> PositionEncoding:
    """
    設定から位置エンコーディングを生成

    Args:
        config: 位置エンコーディング設定
        head_dim: Attention head dimension

    Returns:
        PositionEncoding instance
    """
    pos_type = PositionEncodingType(config.type)

    if pos_type == PositionEncodingType.NONE:
        return NoPositionEncoding()
    elif pos_type == PositionEncodingType.ROPE:
        return RotaryPositionEncoding(
            head_dim=head_dim,
            rotary_pct=config.rotary_pct,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_base,
        )
    elif pos_type == PositionEncodingType.ROPE3D:
        return Rotary3DPositionEncoding(
            head_dim=head_dim,
            rotary_pct=config.rope3d_pct,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_base,
        )
    elif pos_type == PositionEncodingType.ALIBI:
        return ALiBiPositionEncoding(slope=config.alibi_slope)
    else:
        raise ValueError(f"Unknown position encoding type: {config.type}")
