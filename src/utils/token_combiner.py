"""
TokenCombiner - トークン結合ロジックの統一クラス

複数トークン入力（num_input_tokens > 1）のための履歴結合処理を一箇所に集約。
"""

from typing import List
import torch


class TokenCombiner:
    """
    トークン結合処理を統一的に行うクラス

    num_input_tokens分のトークン履歴を結合して、
    [embed_dim * num_input_tokens]次元のベクトルを生成する。
    """

    def __init__(self, num_input_tokens: int, embed_dim: int):
        """
        Args:
            num_input_tokens: 入力トークン数（履歴の長さ）
            embed_dim: トークン埋め込みの次元数
        """
        self.num_input_tokens = num_input_tokens
        self.embed_dim = embed_dim
        self.combined_dim = embed_dim * num_input_tokens

    def combine_single(
        self,
        token_history: List[torch.Tensor],
        current_token: torch.Tensor
    ) -> torch.Tensor:
        """
        単一トークン位置の結合処理（シーケンシャル処理用）

        Args:
            token_history: 過去のトークン埋め込みリスト（最大num_input_tokens-1個）
            current_token: 現在のトークン埋め込み [embed_dim]

        Returns:
            combined: 結合されたトークン [embed_dim * num_input_tokens]
        """
        if self.num_input_tokens == 1:
            return current_token

        # 履歴 + 現在のトークンを結合
        full_history = token_history + [current_token]
        # 最新のnum_input_tokens個を取得
        window = full_history[-self.num_input_tokens:]

        # 不足分はゼロパディング
        if len(window) < self.num_input_tokens:
            padding_count = self.num_input_tokens - len(window)
            padding = [torch.zeros_like(current_token) for _ in range(padding_count)]
            window = padding + window

        return torch.cat(window, dim=-1)

    def combine_batch(
        self,
        token_embeds: torch.Tensor,
        start_idx: int,
        end_idx: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        バッチ範囲のトークン結合（並列処理用）

        Args:
            token_embeds: 全トークン埋め込み [num_tokens, embed_dim]
            start_idx: バッチ開始インデックス
            end_idx: バッチ終了インデックス
            device: 出力デバイス

        Returns:
            combined: 結合されたトークン [batch_size, embed_dim * num_input_tokens]
        """
        if self.num_input_tokens == 1:
            return token_embeds[start_idx:end_idx]

        batch_size = end_idx - start_idx
        combined = torch.zeros(
            batch_size,
            self.combined_dim,
            device=device,
            dtype=token_embeds.dtype
        )

        for batch_i, global_i in enumerate(range(start_idx, end_idx)):
            for j in range(self.num_input_tokens):
                # j=0 が最も古いトークン、j=num_input_tokens-1 が現在のトークン
                src_idx = global_i - (self.num_input_tokens - 1 - j)
                if src_idx >= 0:
                    start = j * self.embed_dim
                    end = (j + 1) * self.embed_dim
                    combined[batch_i, start:end] = token_embeds[src_idx]
                # src_idx < 0 の場合はゼロベクトル（初期化済み）

        return combined

    def combine_all(
        self,
        token_embeds: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        全トークンの結合（キャッシュ収集用）

        Args:
            token_embeds: 全トークン埋め込み [num_tokens, embed_dim]
            device: 出力デバイス

        Returns:
            combined: 結合されたトークン [num_tokens, embed_dim * num_input_tokens]
        """
        if self.num_input_tokens == 1:
            return token_embeds.to(device)

        num_tokens = len(token_embeds)
        combined = torch.zeros(
            num_tokens,
            self.combined_dim,
            device=device,
            dtype=token_embeds.dtype
        )

        for i in range(num_tokens):
            start_idx = max(0, i - self.num_input_tokens + 1)
            token_window = token_embeds[start_idx:i+1]

            if len(token_window) < self.num_input_tokens:
                # 不足分をゼロパディング
                padding = torch.zeros(
                    self.num_input_tokens - len(token_window),
                    self.embed_dim,
                    device=token_embeds.device,
                    dtype=token_embeds.dtype
                )
                token_window = torch.cat([padding, token_window], dim=0)

            combined[i] = token_window.flatten()

        return combined
