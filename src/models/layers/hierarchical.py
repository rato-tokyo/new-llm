"""
Hierarchical Memory Layer
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_components import PythiaMLP
from src.models.memory_utils import causal_linear_attention, elu_plus_one
from .base import BaseLayer


class HierarchicalAttention(nn.Module):
    """Hierarchical Memory with Learned Expansion Gate"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_fine_memories: int = 4,
        use_delta_rule: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_fine_memories = num_fine_memories
        self.use_delta_rule = use_delta_rule

        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

        self.memory_gate = nn.Parameter(torch.zeros(num_heads))
        self.expansion_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )

        self.fine_memories: Optional[list[torch.Tensor]] = None
        self.fine_memory_norms: Optional[list[torch.Tensor]] = None
        self.register_buffer('current_memory_idx', torch.tensor(0))

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = self.w_q.weight.device
        dtype = self.w_q.weight.dtype

        self.fine_memories = [
            torch.zeros(self.num_heads, self.head_dim, self.head_dim, device=device, dtype=dtype)
            for _ in range(self.num_fine_memories)
        ]
        self.fine_memory_norms = [
            torch.zeros(self.num_heads, self.head_dim, device=device, dtype=dtype)
            for _ in range(self.num_fine_memories)
        ]
        self.current_memory_idx = torch.tensor(0, device=device)

    def _get_coarse_memory(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.fine_memories is not None and self.fine_memory_norms is not None
        return torch.stack(self.fine_memories).sum(dim=0), torch.stack(self.fine_memory_norms).sum(dim=0)

    def _retrieve_from_memory(
        self, sigma_q: torch.Tensor, memory: torch.Tensor, memory_norm: torch.Tensor
    ) -> torch.Tensor:
        if memory_norm.sum() < 1e-6:
            return torch.zeros_like(sigma_q)
        a_mem = torch.einsum('bhsd,hde->bhse', sigma_q, memory)
        norm = torch.einsum('bhsd,hd->bhs', sigma_q, memory_norm).clamp(min=1e-6).unsqueeze(-1)
        return a_mem / norm

    def _retrieve_fine_grained(self, sigma_q: torch.Tensor) -> torch.Tensor:
        if self.fine_memories is None or self.fine_memory_norms is None:
            return torch.zeros_like(sigma_q)

        outputs, relevances = [], []
        for memory, memory_norm in zip(self.fine_memories, self.fine_memory_norms):
            out = self._retrieve_from_memory(sigma_q, memory, memory_norm)
            outputs.append(out)
            relevances.append(torch.einsum('bhsd,hd->bhs', sigma_q, memory_norm).mean(dim=-1))

        stacked = torch.stack(outputs, dim=0)
        weights = F.softmax(torch.stack(relevances, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        return (stacked * weights).sum(dim=0)

    def _update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        sigma_k = elu_plus_one(k)

        if self.fine_memories is None:
            self.reset_memory(k.device)

        assert self.fine_memories is not None and self.fine_memory_norms is not None

        idx = int(self.current_memory_idx.item())
        memory, memory_norm = self.fine_memories[idx], self.fine_memory_norms[idx]

        if self.use_delta_rule:
            retrieved = torch.einsum('bhsd,hde->bhse', sigma_k, memory)
            norm = torch.einsum('bhsd,hd->bhs', sigma_k, memory_norm).clamp(min=1e-6).unsqueeze(-1)
            delta_v = v - retrieved / norm
            update = torch.einsum('bhsd,bhse->hde', sigma_k, delta_v) / (k.size(0) * k.size(2))
        else:
            update = torch.einsum('bhsd,bhse->hde', sigma_k, v) / (k.size(0) * k.size(2))

        self.fine_memories[idx] = (memory + update).detach()
        self.fine_memory_norms[idx] = (memory_norm + sigma_k.sum(dim=(0, 2)) / k.size(0)).detach()
        self.current_memory_idx = torch.tensor((idx + 1) % self.num_fine_memories, device=k.device)

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

        sigma_q = elu_plus_one(q)

        if self.fine_memories is None:
            self.reset_memory(q.device)

        # Coarse retrieval
        M_coarse, z_coarse = self._get_coarse_memory()
        output_coarse = self._retrieve_from_memory(sigma_q, M_coarse, z_coarse)

        # Expansion decision
        coarse_repr = output_coarse.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        expansion_prob = torch.sigmoid(self.expansion_gate(coarse_repr)).transpose(1, 2).unsqueeze(-1)

        # Fine retrieval and mix
        output_fine = self._retrieve_fine_grained(sigma_q)
        memory_output = expansion_prob * output_fine + (1 - expansion_prob) * output_coarse

        # Local attention
        local_output = causal_linear_attention(q, k, v)
        gate = torch.sigmoid(self.memory_gate).view(1, self.num_heads, 1, 1)
        output = gate * memory_output + (1 - gate) * local_output

        if update_memory:
            self._update_memory(k, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.w_o(output)

    def get_memory_state(self) -> dict:
        return {
            "fine_memories": [m.cpu().clone() for m in self.fine_memories] if self.fine_memories else None,
            "fine_memory_norms": [z.cpu().clone() for z in self.fine_memory_norms] if self.fine_memory_norms else None,
            "current_memory_idx": self.current_memory_idx.cpu().clone(),
        }

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = self.w_q.weight.device
        if state["fine_memories"]:
            self.fine_memories = [m.to(device) for m in state["fine_memories"]]
        if state["fine_memory_norms"]:
            self.fine_memory_norms = [z.to(device) for z in state["fine_memory_norms"]]
        self.current_memory_idx = state["current_memory_idx"].to(device)


class HierarchicalLayer(BaseLayer):
    """Hierarchical Memory Attention Layer"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_fine_memories: int = 4,
        use_delta_rule: bool = True,
    ):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.attention = HierarchicalAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_fine_memories=num_fine_memories,
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
