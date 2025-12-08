"""
Multi-Memory Layer (Multiple Independent Memories)

Landmark方式:
  - "memory_norm": memory_normを流用（デフォルト）
  - "learned": 学習可能な射影でLandmarkを計算
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_components import PythiaMLP
from src.models.memory_utils import causal_linear_attention, elu_plus_one
from .base import BaseLayer

LandmarkType = Literal["memory_norm", "learned"]


class MultiMemoryAttention(nn.Module):
    """Multiple Independent Memories with Attention-based Selection"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_memories: int = 4,
        use_delta_rule: bool = True,
        landmark_type: LandmarkType = "memory_norm",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_memories = num_memories
        self.use_delta_rule = use_delta_rule
        self.landmark_type = landmark_type

        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

        self.gate = nn.Parameter(torch.zeros(num_heads))

        # 学習可能Landmark用の射影（landmark_type="learned"の場合のみ使用）
        if landmark_type == "learned":
            self.landmark_proj = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self.memories: Optional[list[torch.Tensor]] = None
        self.memory_norms: Optional[list[torch.Tensor]] = None
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
        self.current_memory_idx = torch.tensor(0, device=device)

    def _compute_landmark(self, memory_norm: torch.Tensor) -> torch.Tensor:
        """Landmarkベクトルを計算

        Args:
            memory_norm: (num_heads, head_dim)

        Returns:
            landmark: (num_heads, head_dim)
        """
        if self.landmark_type == "learned":
            # 学習可能な射影を適用
            return self.landmark_proj(memory_norm)
        else:
            # memory_normをそのまま使用
            return memory_norm

    def _compute_relevance(
        self, sigma_q: torch.Tensor, landmark: torch.Tensor
    ) -> torch.Tensor:
        """クエリとLandmarkの関連度を計算

        Args:
            sigma_q: (batch, num_heads, seq_len, head_dim)
            landmark: (num_heads, head_dim)

        Returns:
            relevance: (batch, num_heads) - シーケンス平均
        """
        # (batch, num_heads, seq_len)
        rel = torch.einsum('bhsd,hd->bhs', sigma_q, landmark)
        # シーケンス平均
        return rel.mean(dim=-1)

    def _retrieve_from_memory(self, q: torch.Tensor) -> torch.Tensor:
        if self.memories is None or self.memory_norms is None:
            return torch.zeros_like(q)

        sigma_q = elu_plus_one(q)
        outputs, relevances = [], []

        for memory, memory_norm in zip(self.memories, self.memory_norms):
            if memory_norm.sum() < 1e-6:
                outputs.append(torch.zeros_like(q))
                relevances.append(torch.zeros(q.size(0), self.num_heads, device=q.device))
            else:
                # メモリからの検索（共通）
                a_mem = torch.einsum('bhsd,hde->bhse', sigma_q, memory)
                norm = torch.einsum('bhsd,hd->bhs', sigma_q, memory_norm)
                outputs.append(a_mem / norm.clamp(min=1e-6).unsqueeze(-1))

                # Landmark計算と関連度（方式により異なる）
                landmark = self._compute_landmark(memory_norm)
                relevances.append(self._compute_relevance(sigma_q, landmark))

        stacked = torch.stack(outputs, dim=0)
        weights = F.softmax(torch.stack(relevances, dim=0), dim=0)
        return (stacked * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)

    def _update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        sigma_k = elu_plus_one(k)

        if self.memories is None:
            self.reset_memory(k.device)

        assert self.memories is not None and self.memory_norms is not None

        idx = int(self.current_memory_idx.item())
        memory, memory_norm = self.memories[idx], self.memory_norms[idx]

        if self.use_delta_rule:
            retrieved = torch.einsum('bhsd,hde->bhse', sigma_k, memory)
            norm = torch.einsum('bhsd,hd->bhs', sigma_k, memory_norm).clamp(min=1e-6).unsqueeze(-1)
            delta_v = v - retrieved / norm
            update = torch.einsum('bhsd,bhse->hde', sigma_k, delta_v) / (k.size(0) * k.size(2))
        else:
            update = torch.einsum('bhsd,bhse->hde', sigma_k, v) / (k.size(0) * k.size(2))

        self.memories[idx] = (memory + update).detach()
        self.memory_norms[idx] = (memory_norm + sigma_k.sum(dim=(0, 2)) / k.size(0)).detach()
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
            "current_memory_idx": self.current_memory_idx.cpu().clone(),
        }

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = self.w_q.weight.device
        if state["memories"]:
            self.memories = [m.to(device) for m in state["memories"]]
        if state["memory_norms"]:
            self.memory_norms = [n.to(device) for n in state["memory_norms"]]
        self.current_memory_idx = state["current_memory_idx"].to(device)


class MultiMemoryLayer(BaseLayer):
    """Multi-Memory Infini-Attention Layer

    Args:
        landmark_type: Landmark計算方式
            - "memory_norm": memory_normを流用（デフォルト、追加パラメータなし）
            - "learned": 学習可能な射影でLandmarkを計算
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_memories: int = 4,
        use_delta_rule: bool = True,
        landmark_type: LandmarkType = "memory_norm",
    ):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.attention = MultiMemoryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_memories=num_memories,
            use_delta_rule=use_delta_rule,
            landmark_type=landmark_type,
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
