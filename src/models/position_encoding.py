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
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PositionEncodingType(Enum):
    """位置エンコーディングの種類"""
    NONE = "none"     # 位置情報なし
    ROPE = "rope"     # Rotary Position Embedding
    ALIBI = "alibi"   # Attention with Linear Biases
    LEARNABLE = "learnable"  # Learnable Position Encoding (K only)


@dataclass
class PositionEncodingConfig:
    """位置エンコーディングの設定"""
    type: str = "rope"  # "none", "rope", "alibi", "learnable"

    # RoPE parameters
    rotary_pct: float = 0.25  # RoPEを適用する次元の割合
    rope_base: int = 10000    # RoPE base frequency

    # ALiBi parameters
    alibi_slope: float = 0.0625  # ALiBi slope (unified)

    # Learnable parameters
    learnable_dim: Optional[int] = None  # 変換する次元数（Noneの場合はhead_dim全体）
    learnable_nonlinear: str = "gelu"    # 非線形関数: "gelu", "relu", "tanh", "none"

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
        self._cache: dict[int, torch.Tensor] = {}

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


class LearnablePositionEncoding(PositionEncoding):
    """
    Learnable Position Encoding (Kにのみ適用)

    学習可能な遷移行列を位置ごとに累積的に適用。
    初期値は回転行列に基づき、学習により最適化。

    K_pos = nonlinear(K @ M^pos)

    ここで M は学習可能な行列、pos は位置。
    指数的な適用により、遠い位置ほど多くの変換が累積される。
    """

    def __init__(
        self,
        head_dim: int,
        learnable_dim: Optional[int] = None,
        nonlinear: str = "gelu",
        max_position_embeddings: int = 2048,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.learnable_dim = learnable_dim if learnable_dim is not None else head_dim
        self.max_position_embeddings = max_position_embeddings
        self.nonlinear_type = nonlinear

        # 学習可能な遷移行列（learnable_dim x learnable_dim）
        # 初期値: ブロック対角回転行列に近い形
        self.transition_matrix = nn.Parameter(
            self._init_rotation_matrix(self.learnable_dim)
        )

        # 非線形関数の選択
        if nonlinear == "gelu":
            self.nonlinear = nn.GELU()
        elif nonlinear == "relu":
            self.nonlinear = nn.ReLU()
        elif nonlinear == "tanh":
            self.nonlinear = nn.Tanh()
        elif nonlinear == "none":
            self.nonlinear = nn.Identity()
        else:
            raise ValueError(f"Unknown nonlinear function: {nonlinear}")

        # 数値安定性のためのLayerNorm
        self.layer_norm = nn.LayerNorm(self.learnable_dim)

    def _init_rotation_matrix(self, dim: int) -> torch.Tensor:
        """
        回転行列に近い初期値を生成

        2x2ブロック対角行列として、各ブロックが小さな回転を表す
        """
        matrix = torch.eye(dim)

        for i in range(0, dim - 1, 2):
            # 各次元ペアに小さな回転角を設定
            # 低次元ほど大きな回転（RoPEの周波数設計に類似）
            angle = 0.1 / (10.0 ** (i / dim))  # 小さな初期回転
            cos_val = torch.cos(torch.tensor(angle))
            sin_val = torch.sin(torch.tensor(angle))

            matrix[i, i] = cos_val
            matrix[i, i + 1] = -sin_val
            matrix[i + 1, i] = sin_val
            matrix[i + 1, i + 1] = cos_val

        return matrix

    def _apply_position_transform(
        self,
        x: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        位置に応じた変換を適用

        Args:
            x: [batch, heads, seq, head_dim]
            seq_len: シーケンス長

        Returns:
            transformed: [batch, heads, seq, head_dim]
        """
        batch, heads, seq, dim = x.shape

        # learnable_dim部分とpass-through部分に分割
        x_transform = x[..., :self.learnable_dim]
        x_pass = x[..., self.learnable_dim:]

        # 各位置に対して累積的に行列を適用
        outputs = []

        # 位置0: 変換なし
        outputs.append(x_transform[:, :, 0:1, :])

        # 位置1以降: 累積的に行列を適用
        # M^1, M^2, M^3, ... の効果
        accumulated_matrix = self.transition_matrix.clone()

        for pos in range(1, seq_len):
            # x @ M^pos
            # x_transform[:, :, pos, :] has shape [batch, heads, learnable_dim]
            x_pos = x_transform[:, :, pos, :]

            # 行列適用: [batch, heads, learnable_dim] @ [learnable_dim, learnable_dim]
            transformed = torch.matmul(x_pos, accumulated_matrix)

            # 非線形変換
            transformed = self.nonlinear(transformed)

            # LayerNormで数値安定化
            transformed = self.layer_norm(transformed)

            outputs.append(transformed.unsqueeze(2))

            # 次の位置のために行列を累積
            accumulated_matrix = torch.matmul(accumulated_matrix, self.transition_matrix)

        # 結合
        x_transformed = torch.cat(outputs, dim=2)

        # pass-through部分と結合
        if x_pass.shape[-1] > 0:
            return torch.cat([x_transformed, x_pass], dim=-1)
        else:
            return x_transformed

    def apply_to_qk(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Keyにのみ位置変換を適用（Queryはそのまま）
        """
        # Kにのみ適用
        key_transformed = self._apply_position_transform(key, seq_len)

        return query, key_transformed

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
    elif pos_type == PositionEncodingType.ALIBI:
        return ALiBiPositionEncoding(slope=config.alibi_slope)
    elif pos_type == PositionEncodingType.LEARNABLE:
        return LearnablePositionEncoding(
            head_dim=head_dim,
            learnable_dim=config.learnable_dim,
            nonlinear=config.learnable_nonlinear,
            max_position_embeddings=config.max_position_embeddings,
        )
    else:
        raise ValueError(f"Unknown position encoding type: {config.type}")
