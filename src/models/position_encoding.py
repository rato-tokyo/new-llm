"""
Position Encoding Module

ALiBi (Attention with Linear Biases) 位置エンコーディング。

Usage:
    pos_enc = ALiBiPositionEncoding(slope=0.0625)
    attn_scores = pos_enc.apply_to_scores(attn_scores, seq_len)
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class ALiBiPositionEncoding(nn.Module):
    """
    Attention with Linear Biases (ALiBi)

    全ヘッドで統一スロープを使用。
    """

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
        device = attn_scores.device
        alibi_bias = self._get_bias(seq_len, device)
        return attn_scores + alibi_bias
