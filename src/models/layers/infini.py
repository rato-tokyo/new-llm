"""
Infini-Attention Layer (Memory + Linear Attention)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_components import PythiaMLP
from src.models.memory_utils import causal_linear_attention, elu_plus_one
from .base import BaseLayer


class InfiniAttention(nn.Module):
    """
    Infini-Attention with Compressive Memory

    Features:
    - Single-head memory (memory_head_dim = hidden_size)
    - Delta Rule for memory update
    - Multi-bank support
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_memory_banks: int = 1,
        segments_per_bank: int = 4,
        use_delta_rule: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.memory_head_dim = hidden_size  # Single-head for expressiveness
        self.num_memory_banks = num_memory_banks
        self.segments_per_bank = segments_per_bank
        self.use_delta_rule = use_delta_rule

        self.w_q = nn.Linear(hidden_size, self.memory_head_dim, bias=False)
        self.w_k = nn.Linear(hidden_size, self.memory_head_dim, bias=False)
        self.w_v = nn.Linear(hidden_size, self.memory_head_dim, bias=False)
        self.w_o = nn.Linear(self.memory_head_dim, hidden_size, bias=False)

        self.gate = nn.Parameter(torch.zeros(1))

        if num_memory_banks > 1:
            self.bank_weights = nn.Parameter(torch.zeros(num_memory_banks))
        else:
            self.register_buffer('bank_weights', None)

        self.memories: Optional[list[torch.Tensor]] = None
        self.memory_norms: Optional[list[torch.Tensor]] = None
        self.register_buffer('current_bank', torch.tensor(0))
        self.register_buffer('segment_counter', torch.tensor(0))

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = self.w_q.weight.device
        dtype = self.w_q.weight.dtype

        self.memories = [
            torch.zeros(self.memory_head_dim, self.memory_head_dim, device=device, dtype=dtype)
            for _ in range(self.num_memory_banks)
        ]
        self.memory_norms = [
            torch.zeros(self.memory_head_dim, device=device, dtype=dtype)
            for _ in range(self.num_memory_banks)
        ]
        self.current_bank = torch.tensor(0, device=device)
        self.segment_counter = torch.tensor(0, device=device)

    def _retrieve_from_memory(self, q: torch.Tensor) -> torch.Tensor:
        if self.memories is None or self.memory_norms is None:
            return torch.zeros_like(q)

        sigma_q = elu_plus_one(q)

        if self.num_memory_banks == 1:
            memory, memory_norm = self.memories[0], self.memory_norms[0]
            if memory_norm.sum() < 1e-6:
                return torch.zeros_like(q)
            a_mem = torch.matmul(sigma_q, memory)
            norm = torch.matmul(sigma_q, memory_norm).clamp(min=1e-6).unsqueeze(-1)
            return a_mem / norm

        # Multi-bank
        results = []
        for memory, memory_norm in zip(self.memories, self.memory_norms):
            if memory_norm.sum() < 1e-6:
                results.append(torch.zeros_like(q))
            else:
                a_mem = torch.matmul(sigma_q, memory)
                norm = torch.matmul(sigma_q, memory_norm).clamp(min=1e-6).unsqueeze(-1)
                results.append(a_mem / norm)

        stacked = torch.stack(results, dim=0)
        weights = F.softmax(self.bank_weights, dim=-1).view(-1, 1, 1, 1)
        return (stacked * weights).sum(dim=0)

    def _update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        sigma_k = elu_plus_one(k)

        if self.memories is None:
            self.reset_memory(k.device)

        assert self.memories is not None and self.memory_norms is not None

        bank_idx = int(self.current_bank.item())
        memory = self.memories[bank_idx]
        memory_norm = self.memory_norms[bank_idx]
        batch_size, seq_len, _ = k.shape

        if self.use_delta_rule:
            retrieved_unnorm = torch.matmul(sigma_k, memory)
            norm = torch.matmul(sigma_k, memory_norm).clamp(min=1e-6).unsqueeze(-1)
            delta_v = v - retrieved_unnorm / norm
            memory_update = torch.einsum('bsd,bse->de', sigma_k, delta_v) / (batch_size * seq_len)
        else:
            memory_update = torch.einsum('bsd,bse->de', sigma_k, v) / (batch_size * seq_len)

        self.memories[bank_idx] = (memory + memory_update).detach()
        z_update = sigma_k.sum(dim=(0, 1)) / batch_size
        self.memory_norms[bank_idx] = (memory_norm + z_update).detach()

        if self.num_memory_banks > 1:
            self.segment_counter = self.segment_counter + 1
            if self.segment_counter >= self.segments_per_bank:
                self.segment_counter = torch.tensor(0, device=k.device)
                self.current_bank = (self.current_bank + 1) % self.num_memory_banks

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        q = self.w_q(hidden_states)
        k = self.w_k(hidden_states)
        v = self.w_v(hidden_states)

        memory_output = self._retrieve_from_memory(q)
        local_output = causal_linear_attention(q, k, v)

        gate = torch.sigmoid(self.gate)
        output = gate * memory_output + (1 - gate) * local_output

        if update_memory:
            self._update_memory(k, v)

        return self.w_o(output)

    def get_memory_state(self) -> dict:
        return {
            "memories": [m.cpu().clone() for m in self.memories] if self.memories else None,
            "memory_norms": [n.cpu().clone() for n in self.memory_norms] if self.memory_norms else None,
            "current_bank": self.current_bank.cpu().clone(),
            "segment_counter": self.segment_counter.cpu().clone(),
        }

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = self.w_q.weight.device
        if state["memories"]:
            self.memories = [m.to(device) for m in state["memories"]]
        if state["memory_norms"]:
            self.memory_norms = [n.to(device) for n in state["memory_norms"]]
        self.current_bank = state["current_bank"].to(device)
        self.segment_counter = state["segment_counter"].to(device)


class InfiniLayer(BaseLayer):
    """
    Infini-Attention Transformer Layer

    Features:
    - Compressive memory (single-head)
    - Linear Attention for local context
    - No position encoding (NoPE)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_memory_banks: int = 1,
        segments_per_bank: int = 4,
        use_delta_rule: bool = True,
    ):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.attention = InfiniAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_memory_banks=num_memory_banks,
            segments_per_bank=segments_per_bank,
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
