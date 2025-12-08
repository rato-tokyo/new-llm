"""
Multi-Memory Layer (Multiple Independent Memories)

HSA方式のLandmark計算を採用:
- Landmark = mean(K): 各メモリに格納されたキーの平均方向
- 検索スコア = Q @ Landmark: クエリとLandmarkの内積
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_components import PythiaMLP
from src.models.memory_utils import causal_linear_attention, elu_plus_one
from .base import BaseLayer


class MultiMemoryAttention(nn.Module):
    """Multiple Independent Memories with HSA-style Landmark Selection

    HSA (Hierarchical Sparse Attention) 方式:
    - 各メモリにLandmark（キーの累積平均）を保持
    - クエリとLandmarkの内積でメモリ選択
    - 選択されたメモリからLinear Attentionで検索
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_memories: int = 4,
        use_delta_rule: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_memories = num_memories
        self.use_delta_rule = use_delta_rule

        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

        self.gate = nn.Parameter(torch.zeros(num_heads))

        # メモリ状態
        self.memories: Optional[list[torch.Tensor]] = None
        self.memory_norms: Optional[list[torch.Tensor]] = None
        self.landmarks: Optional[list[torch.Tensor]] = None  # HSA方式: mean(K)
        self.key_counts: Optional[list[torch.Tensor]] = None  # Landmark計算用カウンタ
        self.register_buffer('current_memory_idx', torch.tensor(0))

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = self.w_q.weight.device
        dtype = self.w_q.weight.dtype

        self.memories = [
            torch.zeros(self.num_heads, self.head_dim, self.head_dim, device=device, dtype=dtype)
            for _ in range(self.num_memories)
        ]
        self.memory_norms = [
            torch.zeros(self.num_heads, self.head_dim, device=device, dtype=dtype)
            for _ in range(self.num_memories)
        ]
        # HSA方式: Landmark = mean(K)
        self.landmarks = [
            torch.zeros(self.num_heads, self.head_dim, device=device, dtype=dtype)
            for _ in range(self.num_memories)
        ]
        self.key_counts = [
            torch.zeros(self.num_heads, device=device, dtype=dtype)
            for _ in range(self.num_memories)
        ]
        self.current_memory_idx = torch.tensor(0, device=device)

    def _compute_relevance(self, q: torch.Tensor, landmark: torch.Tensor) -> torch.Tensor:
        """クエリとLandmarkの関連度を計算（HSA方式）

        Args:
            q: (batch, num_heads, seq_len, head_dim) - 生のクエリ（σ変換なし）
            landmark: (num_heads, head_dim) - mean(K)

        Returns:
            relevance: (batch, num_heads) - シーケンス平均
        """
        # HSA方式: Q @ Landmark（σ変換なしの内積）
        rel = torch.einsum('bhsd,hd->bhs', q, landmark)
        return rel.mean(dim=-1)

    def _retrieve_from_memory(self, q: torch.Tensor) -> torch.Tensor:
        """全メモリから検索し、Landmark関連度で加重統合"""
        if self.memories is None or self.landmarks is None:
            return torch.zeros_like(q)

        sigma_q = elu_plus_one(q)
        outputs, relevances = [], []

        for memory, memory_norm, landmark, key_count in zip(
            self.memories, self.memory_norms, self.landmarks, self.key_counts  # type: ignore
        ):
            # メモリが空の場合
            if key_count.sum() < 1e-6:
                outputs.append(torch.zeros_like(q))
                relevances.append(
                    torch.full((q.size(0), self.num_heads), float('-inf'), device=q.device)
                )
            else:
                # Linear Attentionでメモリから検索
                a_mem = torch.einsum('bhsd,hde->bhse', sigma_q, memory)
                norm = torch.einsum('bhsd,hd->bhs', sigma_q, memory_norm)
                outputs.append(a_mem / norm.clamp(min=1e-6).unsqueeze(-1))

                # HSA方式: Q @ Landmark で関連度計算
                relevances.append(self._compute_relevance(q, landmark))

        stacked = torch.stack(outputs, dim=0)  # (num_memories, batch, heads, seq, dim)
        weights = F.softmax(torch.stack(relevances, dim=0), dim=0)  # (num_memories, batch, heads)
        return (stacked * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)

    def _update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """メモリとLandmarkを更新"""
        sigma_k = elu_plus_one(k)

        if self.memories is None:
            self.reset_memory(k.device)

        assert self.memories is not None and self.memory_norms is not None
        assert self.landmarks is not None and self.key_counts is not None

        idx = int(self.current_memory_idx.item())
        memory = self.memories[idx]
        memory_norm = self.memory_norms[idx]
        landmark = self.landmarks[idx]
        key_count = self.key_counts[idx]

        # メモリ更新（Linear Attention形式）
        if self.use_delta_rule:
            retrieved = torch.einsum('bhsd,hde->bhse', sigma_k, memory)
            norm = torch.einsum('bhsd,hd->bhs', sigma_k, memory_norm).clamp(min=1e-6).unsqueeze(-1)
            delta_v = v - retrieved / norm
            update = torch.einsum('bhsd,bhse->hde', sigma_k, delta_v) / (k.size(0) * k.size(2))
        else:
            update = torch.einsum('bhsd,bhse->hde', sigma_k, v) / (k.size(0) * k.size(2))

        self.memories[idx] = (memory + update).detach()
        self.memory_norms[idx] = (memory_norm + sigma_k.sum(dim=(0, 2)) / k.size(0)).detach()

        # HSA方式: Landmark = mean(K) をオンライン更新
        # 新しいキーの合計
        batch_size, num_heads, seq_len, head_dim = k.shape
        k_sum = k.sum(dim=(0, 2))  # (num_heads, head_dim)
        new_count = key_count + batch_size * seq_len

        # オンライン平均更新: new_mean = (old_mean * old_count + new_sum) / new_count
        self.landmarks[idx] = (
            (landmark * key_count.unsqueeze(-1) + k_sum) / new_count.clamp(min=1).unsqueeze(-1)
        ).detach()
        self.key_counts[idx] = new_count.detach()

        self.current_memory_idx = torch.tensor((idx + 1) % self.num_memories, device=k.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.w_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        memory_output = self._retrieve_from_memory(q)
        local_output = causal_linear_attention(q, k, v)

        gate = torch.sigmoid(self.gate).view(1, self.num_heads, 1, 1)
        output = gate * memory_output + (1 - gate) * local_output

        if update_memory:
            self._update_memory(k, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.w_o(output)

    def get_memory_state(self) -> dict:
        return {
            "memories": [m.cpu().clone() for m in self.memories] if self.memories else None,
            "memory_norms": [n.cpu().clone() for n in self.memory_norms] if self.memory_norms else None,
            "landmarks": [lm.cpu().clone() for lm in self.landmarks] if self.landmarks else None,
            "key_counts": [c.cpu().clone() for c in self.key_counts] if self.key_counts else None,
            "current_memory_idx": self.current_memory_idx.cpu().clone(),
        }

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = self.w_q.weight.device
        if state["memories"]:
            self.memories = [m.to(device) for m in state["memories"]]
        if state["memory_norms"]:
            self.memory_norms = [n.to(device) for n in state["memory_norms"]]
        if state.get("landmarks"):
            self.landmarks = [lm.to(device) for lm in state["landmarks"]]
        if state.get("key_counts"):
            self.key_counts = [c.to(device) for c in state["key_counts"]]
        self.current_memory_idx = state["current_memory_idx"].to(device)


class MultiMemoryLayer(BaseLayer):
    """Multi-Memory Infini-Attention Layer with HSA-style Landmarks"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_memories: int = 4,
        use_delta_rule: bool = True,
    ):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.attention = MultiMemoryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_memories=num_memories,
            use_delta_rule=use_delta_rule,
        )
        self.mlp = PythiaMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.attention(hidden_states, attention_mask, update_memory)
        mlp_output = self.mlp(hidden_states)
        return residual + attn_output + mlp_output

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        self.attention.reset_memory(device)

    def get_memory_state(self) -> dict:
        return self.attention.get_memory_state()

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        self.attention.set_memory_state(state, device)
