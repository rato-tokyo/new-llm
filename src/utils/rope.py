"""
Rotary Position Embedding (RoPE) Utility

共通のRoPE実装。複数のモデルで再利用可能。
カスタム周波数設定をサポート。
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional, Union

import torch
import torch.nn as nn


@dataclass
class RoPEConfig:
    """
    RoPE設定クラス

    3つの設定方法をサポート:
    1. Standard: rotary_pctとbaseを指定する標準方式
    2. Custom frequencies: 各2次元ペアの周波数を直接指定
    3. Custom config list: [[dim, freq], [dim, freq], ...] 形式で指定

    Examples:
        # Standard (Pythia style)
        config = RoPEConfig(mode="standard", rotary_pct=0.25, base=10000)

        # Custom frequencies (各2次元ペアの周波数を直接指定)
        config = RoPEConfig(mode="custom", frequencies=[0.1, 0.2, 0.3, 0.4])

        # Custom config list: [[次元数, 周波数], ...]
        config = RoPEConfig(mode="custom_list", config_list=[
            [2, 0.1],   # 最初の2次元: 周波数0.1
            [2, 0.2],   # 次の2次元: 周波数0.2
            [4, 0.05],  # 次の4次元: 周波数0.05
            [8, 0.01],  # 次の8次元: 周波数0.01
        ])
    """
    mode: str = "standard"  # "standard", "custom", "custom_list"

    # Standard mode parameters
    rotary_pct: float = 0.25
    base: int = 10000
    head_dim: int = 64  # Required for standard mode

    # Custom mode: 各2次元ペアの周波数リスト
    frequencies: Optional[List[float]] = None

    # Custom list mode: [[dim, freq], ...] 形式
    config_list: Optional[List[List[float]]] = None

    def get_frequencies(self, head_dim: int) -> torch.Tensor:
        """
        設定に基づいて周波数テンソルを生成

        Returns:
            inv_freq: [rotary_dim // 2] の周波数テンソル
        """
        if self.mode == "standard":
            rotary_dim = int(head_dim * self.rotary_pct)
            # 標準RoPE: base^(-2i/d) の周波数
            inv_freq = 1.0 / (self.base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
            return inv_freq

        elif self.mode == "custom":
            if self.frequencies is None:
                raise ValueError("frequencies must be provided for custom mode")
            return torch.tensor(self.frequencies, dtype=torch.float32)

        elif self.mode == "custom_list":
            if self.config_list is None:
                raise ValueError("config_list must be provided for custom_list mode")
            # [[dim, freq], ...] を展開
            frequencies = []
            for dim, freq in self.config_list:
                dim = int(dim)
                if dim % 2 != 0:
                    raise ValueError(f"Each dimension must be even, got {dim}")
                # 各2次元ペアに同じ周波数を割り当て
                for _ in range(dim // 2):
                    frequencies.append(freq)
            return torch.tensor(frequencies, dtype=torch.float32)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_rotary_dim(self, head_dim: int) -> int:
        """回転を適用する次元数を取得"""
        if self.mode == "standard":
            return int(head_dim * self.rotary_pct)
        elif self.mode == "custom":
            if self.frequencies is None:
                raise ValueError("frequencies must be provided for custom mode")
            return len(self.frequencies) * 2
        elif self.mode == "custom_list":
            if self.config_list is None:
                raise ValueError("config_list must be provided for custom_list mode")
            return sum(int(dim) for dim, _ in self.config_list)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


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


class CustomRotaryEmbedding(nn.Module):
    """
    カスタム設定可能なRotary Position Embedding

    RoPEConfigを使用して柔軟に周波数を設定可能。

    Args:
        head_dim: アテンションヘッドの次元
        config: RoPEConfig設定
        max_position_embeddings: 最大シーケンス長 (default: 2048)

    Examples:
        # 標準RoPE
        rope = CustomRotaryEmbedding(
            head_dim=64,
            config=RoPEConfig(mode="standard", rotary_pct=0.25, base=10000)
        )

        # カスタム周波数
        rope = CustomRotaryEmbedding(
            head_dim=64,
            config=RoPEConfig(mode="custom", frequencies=[0.1, 0.2, 0.3, 0.4])
        )

        # カスタムリスト形式 [[次元数, 周波数], ...]
        rope = CustomRotaryEmbedding(
            head_dim=64,
            config=RoPEConfig(mode="custom_list", config_list=[
                [2, 0.1],   # 最初の2次元: 周波数0.1
                [2, 0.2],   # 次の2次元: 周波数0.2
                [4, 0.05],  # 次の4次元: 周波数0.05
                [8, 0.01],  # 次の8次元: 周波数0.01
            ])
        )
    """

    def __init__(
        self,
        head_dim: int,
        config: RoPEConfig,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.config = config
        self.rotary_dim = config.get_rotary_dim(head_dim)

        # 周波数を取得
        inv_freq = config.get_frequencies(head_dim)
        self.register_buffer("inv_freq", inv_freq)

        # キャッシュ初期化
        self._init_cache(max_position_embeddings)

    def _init_cache(self, max_len: int) -> None:
        """cos/sinキャッシュを初期化"""
        t = torch.arange(max_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        # freqs: [max_len, rotary_dim // 2]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # emb: [max_len, rotary_dim] (各周波数をペアで複製)
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

    def get_config_summary(self) -> str:
        """設定のサマリーを返す"""
        if self.config.mode == "standard":
            return f"Standard RoPE (rotary_pct={self.config.rotary_pct}, base={self.config.base})"
        elif self.config.mode == "custom":
            return f"Custom RoPE (frequencies={self.config.frequencies})"
        elif self.config.mode == "custom_list":
            return f"Custom List RoPE (config_list={self.config.config_list})"
        return f"Unknown mode: {self.config.mode}"


def create_rope_from_config(
    head_dim: int,
    config: Union[RoPEConfig, dict, None] = None,
    max_position_embeddings: int = 2048,
) -> CustomRotaryEmbedding:
    """
    設定からRoPEを生成するファクトリ関数

    Args:
        head_dim: アテンションヘッドの次元
        config: RoPEConfig, dict, or None (Noneの場合は標準設定)
        max_position_embeddings: 最大シーケンス長

    Returns:
        CustomRotaryEmbedding instance

    Examples:
        # 標準RoPE
        rope = create_rope_from_config(64)

        # dict形式で指定
        rope = create_rope_from_config(64, {
            "mode": "custom_list",
            "config_list": [[2, 0.1], [2, 0.2], [4, 0.05], [8, 0.01]]
        })

        # RoPEConfig形式で指定
        rope = create_rope_from_config(64, RoPEConfig(
            mode="custom",
            frequencies=[0.1, 0.2, 0.3, 0.4]
        ))
    """
    if config is None:
        config = RoPEConfig(mode="standard", rotary_pct=0.25, base=10000)
    elif isinstance(config, dict):
        config = RoPEConfig(**config)

    return CustomRotaryEmbedding(
        head_dim=head_dim,
        config=config,
        max_position_embeddings=max_position_embeddings,
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


# Utility functions for creating common RoPE configurations
def standard_rope_config(rotary_pct: float = 0.25, base: int = 10000) -> RoPEConfig:
    """標準RoPE設定を生成"""
    return RoPEConfig(mode="standard", rotary_pct=rotary_pct, base=base)


def custom_frequencies_config(frequencies: List[float]) -> RoPEConfig:
    """カスタム周波数設定を生成"""
    return RoPEConfig(mode="custom", frequencies=frequencies)


def custom_list_config(config_list: List[List[float]]) -> RoPEConfig:
    """
    カスタムリスト設定を生成

    Args:
        config_list: [[次元数, 周波数], ...] 形式のリスト

    Example:
        config = custom_list_config([
            [2, 0.1],   # 最初の2次元: 周波数0.1
            [2, 0.2],   # 次の2次元: 周波数0.2
            [4, 0.05],  # 次の4次元: 周波数0.05
            [8, 0.01],  # 次の8次元: 周波数0.01
        ])
    """
    return RoPEConfig(mode="custom_list", config_list=config_list)


def linear_frequency_config(
    rotary_dim: int,
    min_freq: float = 0.001,
    max_freq: float = 1.0,
) -> RoPEConfig:
    """
    線形に変化する周波数設定を生成

    Args:
        rotary_dim: 回転次元数（偶数）
        min_freq: 最小周波数
        max_freq: 最大周波数

    Returns:
        RoPEConfig with linearly spaced frequencies
    """
    num_pairs = rotary_dim // 2
    frequencies = torch.linspace(min_freq, max_freq, num_pairs).tolist()
    return RoPEConfig(mode="custom", frequencies=frequencies)


def exponential_frequency_config(
    rotary_dim: int,
    base: float = 10000,
) -> RoPEConfig:
    """
    指数的に変化する周波数設定を生成（標準RoPEと同等）

    Args:
        rotary_dim: 回転次元数（偶数）
        base: 周波数ベース

    Returns:
        RoPEConfig with exponentially spaced frequencies
    """
    frequencies = (1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))).tolist()
    return RoPEConfig(mode="custom", frequencies=frequencies)
