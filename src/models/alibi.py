"""
ALiBi (Attention with Linear Biases)

位置エンコーディングを学習可能なパラメータではなく、
Attentionスコアへの線形バイアスとして実装。

特徴:
- 学習パラメータなし
- MLA吸収モードと完全互換
- 統一スロープ方式を採用（全ヘッド同一）
"""

from typing import Dict, Optional

import torch


def build_alibi_bias(
    seq_len: int,
    slope: float = 0.0625,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    ALiBiバイアス行列を生成

    Args:
        seq_len: シーケンス長
        slope: 距離あたりのペナルティ（統一スロープ）
        device: デバイス

    Returns:
        alibi_bias: [seq_len, seq_len] バイアス行列
            - 対角成分: 0
            - 距離dの位置: -slope * d
    """
    # 位置インデックス
    positions = torch.arange(seq_len, device=device)

    # 距離行列: |i - j|
    distance_matrix = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))

    # ALiBiバイアス: -slope * distance
    alibi_bias = -slope * distance_matrix.float()

    return alibi_bias


def build_alibi_bias_causal(
    seq_len: int,
    slope: float = 0.0625,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Causal（因果的）ALiBiバイアス行列を生成

    未来のトークンには-infを設定（因果マスクと統合）

    Args:
        seq_len: シーケンス長
        slope: 距離あたりのペナルティ
        device: デバイス

    Returns:
        alibi_bias: [seq_len, seq_len] バイアス行列
            - i >= j: -slope * (i - j)
            - i < j: -inf (因果マスク)
    """
    # 位置インデックス
    positions = torch.arange(seq_len, device=device)

    # 相対位置: i - j
    relative_pos = positions.unsqueeze(0) - positions.unsqueeze(1)

    # 因果マスク付きALiBiバイアス
    alibi_bias = torch.where(
        relative_pos >= 0,
        -slope * relative_pos.float(),  # 過去・現在: -slope * distance
        torch.tensor(float("-inf"), device=device),  # 未来: -inf
    )

    return alibi_bias


class ALiBiCache:
    """
    ALiBiバイアスのキャッシュ

    同じseq_lenに対して再計算を避ける
    """

    def __init__(self, slope: float = 0.0625, causal: bool = True):
        self.slope = slope
        self.causal = causal
        self._cache: Dict[int, torch.Tensor] = {}

    def get(self, seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """キャッシュからALiBiバイアスを取得（なければ生成）"""
        if seq_len not in self._cache:
            if self.causal:
                bias = build_alibi_bias_causal(seq_len, self.slope, device)
            else:
                bias = build_alibi_bias(seq_len, self.slope, device)
            self._cache[seq_len] = bias

        cached = self._cache[seq_len]

        # デバイスが異なる場合は移動
        if device is not None and cached.device != device:
            cached = cached.to(device)
            self._cache[seq_len] = cached

        return cached

    def clear(self) -> None:
        """キャッシュをクリア"""
        self._cache.clear()
