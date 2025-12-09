"""
Senri Layer - Unified Memory Layer

Features:
- Configurable memory head dimension (default: hidden_size for single-head)
- Multiple memories with memory_norm-based selection
- Delta Rule for memory update
- Freeze/unfreeze capability
- Memory export/import for sharing

This layer unifies InfiniAttention and MultiMemoryAttention:
- num_memories=1: Equivalent to original InfiniAttention
- num_memories>1: Multiple independent memories with landmark selection
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.base_components import PythiaMLP
from src.models.memory import CompressiveMemory
from src.models.memory_utils import causal_linear_attention
from .base import BaseLayer


class SenriAttention(nn.Module):
    """
    Senri Attention with Compressive Memory

    Unified attention mechanism combining:
    - Configurable memory head dimension (default: hidden_size for single-head)
    - Multiple memories with memory_norm-based selection
    - Freeze capability for knowledge preservation

    Args:
        hidden_size: Hidden dimension
        num_heads: Number of attention heads (used only for local attention)
        num_memories: Number of memory slots (1 = original Infini-Attention)
        memory_head_dim: Memory head dimension (None = hidden_size for single-head)
        use_delta_rule: Use delta rule for memory update
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_memories: int = 1,
        memory_head_dim: Optional[int] = None,
        use_delta_rule: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.memory_head_dim = memory_head_dim if memory_head_dim is not None else hidden_size
        self.num_memories = num_memories

        # Projections
        self.w_q = nn.Linear(hidden_size, self.memory_head_dim, bias=False)
        self.w_k = nn.Linear(hidden_size, self.memory_head_dim, bias=False)
        self.w_v = nn.Linear(hidden_size, self.memory_head_dim, bias=False)
        self.w_o = nn.Linear(self.memory_head_dim, hidden_size, bias=False)

        # Gate for memory vs local balance
        self.gate = nn.Parameter(torch.zeros(1))

        # Compressive Memory
        self.memory = CompressiveMemory(
            memory_dim=self.memory_head_dim,
            num_memories=num_memories,
            use_delta_rule=use_delta_rule,
        )

    def reset_memory(
        self,
        device: Optional[torch.device] = None,
        keep_frozen: bool = True,
    ) -> None:
        """Reset memory state."""
        if device is None:
            device = self.w_q.weight.device
        dtype = self.w_q.weight.dtype
        self.memory.reset(device, dtype, keep_frozen)

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

        # Memory retrieval and local attention
        memory_output = self.memory.retrieve(q)
        local_output = causal_linear_attention(q, k, v)

        # Gate between memory and local
        gate = torch.sigmoid(self.gate)
        output = gate * memory_output + (1 - gate) * local_output

        # Update memory
        if update_memory:
            self.memory.update(k, v)

        return self.w_o(output)

    # =========================================================================
    # Memory Management (delegate to CompressiveMemory)
    # =========================================================================

    def freeze_memory(self, memory_indices: Optional[list[int]] = None) -> None:
        """Freeze memories (make them read-only)."""
        self.memory.freeze(memory_indices)

    def unfreeze_memory(self, memory_indices: Optional[list[int]] = None) -> None:
        """Unfreeze memories (make them writable)."""
        self.memory.unfreeze(memory_indices)

    def is_frozen(self, memory_idx: int) -> bool:
        """Check if a memory is frozen."""
        return self.memory.is_frozen(memory_idx)

    def export_memory(self, memory_indices: Optional[list[int]] = None) -> dict:
        """Export memories for sharing."""
        data = self.memory.export(memory_indices)
        # Add hidden_size for compatibility check
        data["hidden_size"] = self.hidden_size
        return data

    def import_memory(
        self,
        memory_data: dict,
        memory_indices: Optional[list[int]] = None,
        freeze: bool = True,
    ) -> None:
        """Import memory from another model or saved state."""
        if memory_data.get("hidden_size") and memory_data["hidden_size"] != self.hidden_size:
            raise ValueError(
                f"Hidden size mismatch: expected {self.hidden_size}, "
                f"got {memory_data['hidden_size']}"
            )
        device = self.w_q.weight.device
        dtype = self.w_q.weight.dtype
        self.memory.import_memory(memory_data, memory_indices, freeze, device, dtype)

    def get_memory_state(self) -> dict:
        """Get memory state for serialization."""
        return self.memory.get_state()

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        """Set memory state from serialized data."""
        if device is None:
            device = self.w_q.weight.device
        self.memory.set_state(state, device)


class SenriLayer(BaseLayer):
    """
    Senri Transformer Layer

    Features:
    - Configurable memory head dimension (default: hidden_size for single-head)
    - Multiple memories with landmark-based selection
    - Linear Attention for local context
    - No position encoding (NoPE)
    - Freeze/unfreeze and export/import capability

    Args:
        hidden_size: Hidden dimension
        num_heads: Number of attention heads
        intermediate_size: MLP intermediate dimension
        num_memories: Number of memory slots (1 = original Infini-Attention)
        memory_head_dim: Memory head dimension (None = hidden_size for single-head)
        use_delta_rule: Use delta rule for memory update
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_memories: int = 1,
        memory_head_dim: Optional[int] = None,
        use_delta_rule: bool = True,
    ):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.attention = SenriAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_memories=num_memories,
            memory_head_dim=memory_head_dim,
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

    def reset_memory(self, device: Optional[torch.device] = None, keep_frozen: bool = True) -> None:
        self.attention.reset_memory(device, keep_frozen)

    def get_memory_state(self) -> dict:
        return self.attention.get_memory_state()

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        self.attention.set_memory_state(state, device)

    # Freeze / Unfreeze
    def freeze_memory(self, memory_indices: Optional[list[int]] = None) -> None:
        self.attention.freeze_memory(memory_indices)

    def unfreeze_memory(self, memory_indices: Optional[list[int]] = None) -> None:
        self.attention.unfreeze_memory(memory_indices)

    def is_frozen(self, memory_idx: int) -> bool:
        return self.attention.is_frozen(memory_idx)

    # Export / Import
    def export_memory(self, memory_indices: Optional[list[int]] = None) -> dict:
        return self.attention.export_memory(memory_indices)

    def import_memory(
        self,
        memory_data: dict,
        memory_indices: Optional[list[int]] = None,
        freeze: bool = True,
    ) -> None:
        self.attention.import_memory(memory_data, memory_indices, freeze)
