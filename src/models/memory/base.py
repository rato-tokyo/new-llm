"""
Compressive Memory - 圧縮メモリの基本実装

Linear Attention形式のメモリ:
- メモリ行列: M ∈ R^{d×d}
- メモリノルム: z ∈ R^{d} (正規化用)

検索: A_mem = σ(Q) @ M / (σ(Q) @ z)
更新: M = M + σ(K)^T @ V (or Delta Rule)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.memory_utils import elu_plus_one


class CompressiveMemory(nn.Module):
    """
    圧縮メモリ（Linear Attention形式）

    Features:
    - 複数メモリスロット対応
    - memory_norm方式によるLandmark選択
    - Delta Rule対応
    - freeze/unfreeze対応
    - export/import対応

    Args:
        memory_dim: メモリの次元（通常はhidden_size）
        num_memories: メモリスロット数
        use_delta_rule: Delta Ruleを使用するか
    """

    def __init__(
        self,
        memory_dim: int,
        num_memories: int = 1,
        use_delta_rule: bool = True,
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.num_memories = num_memories
        self.use_delta_rule = use_delta_rule

        # メモリ状態（動的に初期化）
        self.memories: Optional[list[torch.Tensor]] = None
        self.memory_norms: Optional[list[torch.Tensor]] = None
        self.frozen: Optional[list[bool]] = None
        self.register_buffer('current_idx', torch.tensor(0))

    def reset(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        keep_frozen: bool = True,
    ) -> None:
        """メモリをリセット

        Args:
            device: デバイス
            dtype: データ型
            keep_frozen: Trueの場合、frozenメモリはリセットしない
        """
        if self.memories is None or not keep_frozen:
            # 完全リセット
            self.memories = [
                torch.zeros(self.memory_dim, self.memory_dim, device=device, dtype=dtype)
                for _ in range(self.num_memories)
            ]
            self.memory_norms = [
                torch.zeros(self.memory_dim, device=device, dtype=dtype)
                for _ in range(self.num_memories)
            ]
            self.frozen = [False] * self.num_memories
        else:
            # unfrozenメモリのみリセット
            assert self.memories is not None and self.memory_norms is not None
            assert self.frozen is not None
            for i in range(self.num_memories):
                if not self.frozen[i]:
                    self.memories[i] = torch.zeros(
                        self.memory_dim, self.memory_dim, device=device, dtype=dtype
                    )
                    self.memory_norms[i] = torch.zeros(
                        self.memory_dim, device=device, dtype=dtype
                    )

        self.current_idx = torch.tensor(0, device=device)

    def _is_empty(self, idx: int) -> bool:
        """メモリが空かどうか"""
        if self.memory_norms is None:
            return True
        return self.memory_norms[idx].abs().sum().item() < 1e-6

    def _compute_relevance(self, sigma_q: torch.Tensor, idx: int) -> torch.Tensor:
        """クエリとメモリの関連度を計算（memory_norm方式）

        Landmark = memory_norm (Σσ(k))
        Score = σ(Q) @ Landmark

        Args:
            sigma_q: (batch, seq_len, memory_dim) - σ(Q)
            idx: メモリインデックス

        Returns:
            relevance: (batch,) - シーケンス平均のスコア
        """
        assert self.memory_norms is not None
        landmark = self.memory_norms[idx]
        rel = torch.einsum('bsd,d->bs', sigma_q, landmark)
        return rel.mean(dim=-1)

    def retrieve(self, queries: torch.Tensor) -> torch.Tensor:
        """メモリから検索

        Args:
            queries: (batch, seq_len, memory_dim) - クエリ

        Returns:
            output: (batch, seq_len, memory_dim) - 検索結果
        """
        if self.memories is None or self.memory_norms is None:
            return torch.zeros_like(queries)

        sigma_q = elu_plus_one(queries)

        # 単一メモリの場合（高速パス）
        if self.num_memories == 1:
            memory, memory_norm = self.memories[0], self.memory_norms[0]
            if memory_norm.sum() < 1e-6:
                return torch.zeros_like(queries)
            a_mem = torch.matmul(sigma_q, memory)
            norm = torch.matmul(sigma_q, memory_norm).clamp(min=1e-6).unsqueeze(-1)
            return a_mem / norm

        # 複数メモリの場合（Landmark選択）
        outputs, relevances = [], []
        has_non_empty = False

        for idx, (memory, memory_norm) in enumerate(
            zip(self.memories, self.memory_norms)
        ):
            if self._is_empty(idx):
                outputs.append(torch.zeros_like(queries))
                relevances.append(
                    torch.full((queries.size(0),), float('-inf'), device=queries.device)
                )
            else:
                has_non_empty = True
                a_mem = torch.matmul(sigma_q, memory)
                norm = torch.matmul(sigma_q, memory_norm).clamp(min=1e-6).unsqueeze(-1)
                outputs.append(a_mem / norm)
                relevances.append(self._compute_relevance(sigma_q, idx))

        if not has_non_empty:
            return torch.zeros_like(queries)

        stacked = torch.stack(outputs, dim=0)  # (num_memories, batch, seq, dim)
        weights = F.softmax(torch.stack(relevances, dim=0), dim=0)  # (num_memories, batch)
        return (stacked * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)

    def update(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """メモリを更新

        Args:
            keys: (batch, seq_len, memory_dim) - キー
            values: (batch, seq_len, memory_dim) - バリュー
        """
        sigma_k = elu_plus_one(keys)

        if self.memories is None:
            self.reset(keys.device, keys.dtype)

        assert self.memories is not None and self.memory_norms is not None
        assert self.frozen is not None

        idx = int(self.current_idx.item())

        # frozenメモリをスキップ
        if self.frozen[idx]:
            found = False
            for offset in range(1, self.num_memories + 1):
                candidate = (idx + offset) % self.num_memories
                if not self.frozen[candidate]:
                    idx = candidate
                    self.current_idx = torch.tensor(idx, device=keys.device)
                    found = True
                    break
            if not found:
                return  # 全メモリがfrozen

        memory = self.memories[idx]
        memory_norm = self.memory_norms[idx]
        batch_size, seq_len, _ = keys.shape

        if self.use_delta_rule:
            retrieved_unnorm = torch.matmul(sigma_k, memory)
            norm = torch.matmul(sigma_k, memory_norm).clamp(min=1e-6).unsqueeze(-1)
            delta_v = values - retrieved_unnorm / norm
            memory_update = torch.einsum('bsd,bse->de', sigma_k, delta_v) / (batch_size * seq_len)
        else:
            memory_update = torch.einsum('bsd,bse->de', sigma_k, values) / (batch_size * seq_len)

        self.memories[idx] = (memory + memory_update).detach()
        z_update = sigma_k.sum(dim=(0, 1)) / batch_size
        self.memory_norms[idx] = (memory_norm + z_update).detach()

        # ラウンドロビン（複数メモリ時）
        if self.num_memories > 1:
            next_idx = (idx + 1) % self.num_memories
            for _ in range(self.num_memories):
                if not self.frozen[next_idx]:
                    break
                next_idx = (next_idx + 1) % self.num_memories
            self.current_idx = torch.tensor(next_idx, device=keys.device)

    # =========================================================================
    # Freeze / Unfreeze
    # =========================================================================

    def freeze(self, indices: Optional[list[int]] = None) -> None:
        """メモリをfreeze（読み取り専用に）

        Args:
            indices: freezeするインデックス。Noneで全て
        """
        if self.frozen is None:
            self.frozen = [False] * self.num_memories

        if indices is None:
            indices = list(range(self.num_memories))

        for idx in indices:
            if 0 <= idx < self.num_memories:
                self.frozen[idx] = True

    def unfreeze(self, indices: Optional[list[int]] = None) -> None:
        """メモリをunfreeze（書き込み可能に）

        Args:
            indices: unfreezeするインデックス。Noneで全て
        """
        if self.frozen is None:
            self.frozen = [False] * self.num_memories
            return

        if indices is None:
            indices = list(range(self.num_memories))

        for idx in indices:
            if 0 <= idx < self.num_memories:
                self.frozen[idx] = False

    def is_frozen(self, idx: int) -> bool:
        """メモリがfrozenかどうか"""
        if self.frozen is None:
            return False
        return self.frozen[idx]

    # =========================================================================
    # Export / Import
    # =========================================================================

    def export(self, indices: Optional[list[int]] = None) -> dict:
        """メモリをエクスポート

        Args:
            indices: エクスポートするインデックス。Noneで全て

        Returns:
            メモリデータの辞書
        """
        if self.memories is None or self.memory_norms is None:
            raise ValueError("No memory to export. Call update() first.")

        if indices is None:
            indices = list(range(self.num_memories))

        return {
            "memories": {i: self.memories[i].cpu().clone() for i in indices},
            "memory_norms": {i: self.memory_norms[i].cpu().clone() for i in indices},
            "memory_dim": self.memory_dim,
        }

    def import_memory(
        self,
        data: dict,
        indices: Optional[list[int]] = None,
        freeze: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """メモリをインポート

        Args:
            data: export()で取得したデータ
            indices: インポート先インデックス。Noneでソースと同じ
            freeze: インポートしたメモリをfreezeするか
            device: デバイス
            dtype: データ型
        """
        if data["memory_dim"] != self.memory_dim:
            raise ValueError(
                f"Memory dimension mismatch: expected {self.memory_dim}, "
                f"got {data['memory_dim']}"
            )

        if self.memories is None:
            self.reset(device, dtype)

        assert self.memories is not None and self.memory_norms is not None
        assert self.frozen is not None

        source_indices = list(data["memories"].keys())
        if indices is None:
            indices = source_indices

        if len(indices) != len(source_indices):
            raise ValueError(
                f"Indices count mismatch: {len(indices)} targets "
                f"for {len(source_indices)} sources"
            )

        for src_idx, tgt_idx in zip(source_indices, indices):
            if tgt_idx >= self.num_memories:
                raise ValueError(f"Target index {tgt_idx} out of range")

            self.memories[tgt_idx] = data["memories"][src_idx].to(device, dtype)
            self.memory_norms[tgt_idx] = data["memory_norms"][src_idx].to(device, dtype)

            if freeze:
                self.frozen[tgt_idx] = True

    # =========================================================================
    # State Management
    # =========================================================================

    def get_state(self) -> dict:
        """状態を取得"""
        return {
            "memories": [m.cpu().clone() for m in self.memories] if self.memories else None,
            "memory_norms": [n.cpu().clone() for n in self.memory_norms] if self.memory_norms else None,
            "frozen": self.frozen.copy() if self.frozen else None,
            "current_idx": self.current_idx.cpu().clone(),
        }

    def set_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        """状態を設定"""
        if state["memories"]:
            self.memories = [m.to(device) for m in state["memories"]]
        if state["memory_norms"]:
            self.memory_norms = [n.to(device) for n in state["memory_norms"]]
        if state.get("frozen"):
            self.frozen = state["frozen"].copy()
        else:
            self.frozen = [False] * self.num_memories
        self.current_idx = state["current_idx"].to(device)
