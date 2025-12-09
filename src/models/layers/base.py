"""
Base Layer Protocol

全レイヤーの基底クラスを定義。
"""

from typing import Optional

import torch
import torch.nn as nn


class BaseLayer(nn.Module):
    """
    全レイヤーの基底クラス

    共通インターフェース:
    - forward(hidden_states, attention_mask, **kwargs) -> Tensor
    - reset_memory() (メモリ系レイヤーのみ)
    - get_memory_state() / set_memory_state() (メモリ系レイヤーのみ)
    """

    def reset_memory(
        self,
        device: Optional[torch.device] = None,
        keep_frozen: bool = True,
    ) -> None:
        """メモリをリセット（メモリ系レイヤーでオーバーライド）"""
        pass

    def get_memory_state(self) -> Optional[dict]:
        """メモリ状態を取得（メモリ系レイヤーでオーバーライド）"""
        return None

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        """メモリ状態を設定（メモリ系レイヤーでオーバーライド）"""
        pass
