"""
Multi-Memory Layer (Multiple Independent Memories)

memory_norm方式を採用:
- Landmark = memory_norm（メモリ書き込み時の Σσ(k) を活用）
- 追加パラメータなしでメモリの「重要度」を表現
- シンプルかつ効率的な実装
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_components import PythiaMLP
from src.models.memory_utils import causal_linear_attention, elu_plus_one
from .base import BaseLayer


class MultiMemoryAttention(nn.Module):
    """Multiple Independent Memories with memory_norm-based Landmark Selection

    memory_norm方式:
    - Landmark = memory_norm（Σσ(k)）
    - 検索スコア = σ(Q) @ Landmark
    - 追加パラメータなし、メモリ更新の副産物を活用
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

        # Q/K/V 射影
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

        self.gate = nn.Parameter(torch.zeros(num_heads))

        # メモリ状態
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

    def _is_memory_empty(self, memory_idx: int) -> bool:
        """メモリが空かどうかをチェック"""
        if self.memory_norms is None:
            return True
        # memory_normの合計がほぼゼロなら空
        return self.memory_norms[memory_idx].abs().sum().item() < 1e-6

    def _compute_relevance(self, sigma_q: torch.Tensor, memory_idx: int) -> torch.Tensor:
        """Queryとメモリの関連度を計算（memory_norm方式）

        Landmark = memory_norm（Σσ(k)）
        スコア = σ(Q) @ Landmark

        Args:
            sigma_q: (batch, num_heads, seq_len, head_dim) - σ(Q)
            memory_idx: メモリインデックス

        Returns:
            relevance: (batch, num_heads) - シーケンス平均のスコア
        """
        assert self.memory_norms is not None

        # Landmark = memory_norm
        landmark = self.memory_norms[memory_idx]  # (num_heads, head_dim)

        # σ(Q) @ Landmark
        rel = torch.einsum('bhsd,hd->bhs', sigma_q, landmark)
        return rel.mean(dim=-1)

    def _retrieve_from_memory(self, q: torch.Tensor) -> torch.Tensor:
        """全メモリから検索し、Landmark関連度で加重統合

        Args:
            q: (batch, num_heads, seq_len, head_dim) - Query

        Returns:
            output: (batch, num_heads, seq_len, head_dim)
        """
        if self.memories is None or self.memory_norms is None:
            return torch.zeros_like(q)

        sigma_q = elu_plus_one(q)
        outputs, relevances = [], []
        has_non_empty = False

        for memory_idx, (memory, memory_norm) in enumerate(
            zip(self.memories, self.memory_norms)
        ):
            # メモリが空の場合
            if self._is_memory_empty(memory_idx):
                outputs.append(torch.zeros_like(q))
                relevances.append(
                    torch.full((q.size(0), self.num_heads), float('-inf'), device=q.device)
                )
            else:
                has_non_empty = True
                # Linear Attentionでメモリから検索
                a_mem = torch.einsum('bhsd,hde->bhse', sigma_q, memory)
                norm = torch.einsum('bhsd,hd->bhs', sigma_q, memory_norm)
                outputs.append(a_mem / norm.clamp(min=1e-6).unsqueeze(-1))

                # memory_norm方式: σ(Q) @ memory_norm で関連度計算
                relevances.append(self._compute_relevance(sigma_q, memory_idx))

        # 全メモリが空の場合はゼロを返す
        if not has_non_empty:
            return torch.zeros_like(q)

        stacked = torch.stack(outputs, dim=0)  # (num_memories, batch, heads, seq, dim)
        weights = F.softmax(torch.stack(relevances, dim=0), dim=0)  # (num_memories, batch, heads)
        return (stacked * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)

    def _update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """メモリを更新"""
        sigma_k = elu_plus_one(k)

        if self.memories is None:
            self.reset_memory(k.device)

        assert self.memories is not None and self.memory_norms is not None

        idx = int(self.current_memory_idx.item())
        memory = self.memories[idx]
        memory_norm = self.memory_norms[idx]

        batch_size, num_heads, seq_len, head_dim = k.shape

        # メモリ更新（Linear Attention形式）
        if self.use_delta_rule:
            retrieved = torch.einsum('bhsd,hde->bhse', sigma_k, memory)
            norm = torch.einsum('bhsd,hd->bhs', sigma_k, memory_norm).clamp(min=1e-6).unsqueeze(-1)
            delta_v = v - retrieved / norm
            update = torch.einsum('bhsd,bhse->hde', sigma_k, delta_v) / (batch_size * seq_len)
        else:
            update = torch.einsum('bhsd,bhse->hde', sigma_k, v) / (batch_size * seq_len)

        self.memories[idx] = (memory + update).detach()
        # memory_norm更新（これがLandmarkとしても使われる）
        self.memory_norms[idx] = (memory_norm + sigma_k.sum(dim=(0, 2)) / batch_size).detach()

        self.current_memory_idx = torch.tensor((idx + 1) % self.num_memories, device=k.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Q/K/V
        q = self.w_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # メモリ検索
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
    """Multi-Memory Infini-Attention Layer with memory_norm Landmarks"""

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
