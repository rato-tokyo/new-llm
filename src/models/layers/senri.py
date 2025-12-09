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
import torch.nn.functional as F

from src.models.base_components import PythiaMLP
from src.models.memory_utils import causal_linear_attention, elu_plus_one
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
        self.use_delta_rule = use_delta_rule

        # Projections
        self.w_q = nn.Linear(hidden_size, self.memory_head_dim, bias=False)
        self.w_k = nn.Linear(hidden_size, self.memory_head_dim, bias=False)
        self.w_v = nn.Linear(hidden_size, self.memory_head_dim, bias=False)
        self.w_o = nn.Linear(self.memory_head_dim, hidden_size, bias=False)

        # Gate for memory vs local balance
        self.gate = nn.Parameter(torch.zeros(1))

        # Memory state
        self.memories: Optional[list[torch.Tensor]] = None
        self.memory_norms: Optional[list[torch.Tensor]] = None
        self.frozen_memories: Optional[list[bool]] = None
        self.register_buffer('current_memory_idx', torch.tensor(0))

    def reset_memory(
        self,
        device: Optional[torch.device] = None,
        keep_frozen: bool = True,
    ) -> None:
        """Reset memory state.

        Args:
            device: Device to create tensors on
            keep_frozen: If True, only reset unfrozen memories
        """
        if device is None:
            device = self.w_q.weight.device
        dtype = self.w_q.weight.dtype

        if self.memories is None or not keep_frozen:
            # Full reset
            self.memories = [
                torch.zeros(self.memory_head_dim, self.memory_head_dim, device=device, dtype=dtype)
                for _ in range(self.num_memories)
            ]
            self.memory_norms = [
                torch.zeros(self.memory_head_dim, device=device, dtype=dtype)
                for _ in range(self.num_memories)
            ]
            self.frozen_memories = [False] * self.num_memories
        else:
            # Reset only unfrozen memories
            assert self.memories is not None and self.memory_norms is not None
            assert self.frozen_memories is not None
            for i in range(self.num_memories):
                if not self.frozen_memories[i]:
                    self.memories[i] = torch.zeros(
                        self.memory_head_dim, self.memory_head_dim, device=device, dtype=dtype
                    )
                    self.memory_norms[i] = torch.zeros(
                        self.memory_head_dim, device=device, dtype=dtype
                    )

        self.current_memory_idx = torch.tensor(0, device=device)

    def _is_memory_empty(self, memory_idx: int) -> bool:
        """Check if memory is empty."""
        if self.memory_norms is None:
            return True
        return self.memory_norms[memory_idx].abs().sum().item() < 1e-6

    def _compute_relevance(self, sigma_q: torch.Tensor, memory_idx: int) -> torch.Tensor:
        """Compute query-memory relevance using memory_norm as landmark.

        Landmark = memory_norm (Σσ(k))
        Score = σ(Q) @ Landmark

        Args:
            sigma_q: (batch, seq_len, memory_head_dim) - σ(Q)
            memory_idx: Memory index

        Returns:
            relevance: (batch,) - Sequence-averaged score
        """
        assert self.memory_norms is not None
        landmark = self.memory_norms[memory_idx]  # (memory_head_dim,)
        rel = torch.einsum('bsd,d->bs', sigma_q, landmark)
        return rel.mean(dim=-1)

    def _retrieve_from_memory(self, q: torch.Tensor) -> torch.Tensor:
        """Retrieve from all memories with landmark-based weighting.

        Args:
            q: (batch, seq_len, memory_head_dim) - Query

        Returns:
            output: (batch, seq_len, memory_head_dim)
        """
        if self.memories is None or self.memory_norms is None:
            return torch.zeros_like(q)

        sigma_q = elu_plus_one(q)

        # Single memory case (fast path)
        if self.num_memories == 1:
            memory, memory_norm = self.memories[0], self.memory_norms[0]
            if memory_norm.sum() < 1e-6:
                return torch.zeros_like(q)
            a_mem = torch.matmul(sigma_q, memory)
            norm = torch.matmul(sigma_q, memory_norm).clamp(min=1e-6).unsqueeze(-1)
            return a_mem / norm

        # Multi-memory case with landmark selection
        outputs, relevances = [], []
        has_non_empty = False

        for memory_idx, (memory, memory_norm) in enumerate(
            zip(self.memories, self.memory_norms)
        ):
            if self._is_memory_empty(memory_idx):
                outputs.append(torch.zeros_like(q))
                relevances.append(
                    torch.full((q.size(0),), float('-inf'), device=q.device)
                )
            else:
                has_non_empty = True
                a_mem = torch.matmul(sigma_q, memory)
                norm = torch.matmul(sigma_q, memory_norm).clamp(min=1e-6).unsqueeze(-1)
                outputs.append(a_mem / norm)
                relevances.append(self._compute_relevance(sigma_q, memory_idx))

        if not has_non_empty:
            return torch.zeros_like(q)

        stacked = torch.stack(outputs, dim=0)  # (num_memories, batch, seq, dim)
        weights = F.softmax(torch.stack(relevances, dim=0), dim=0)  # (num_memories, batch)
        return (stacked * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)

    def _update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Update memory with key-value pairs."""
        sigma_k = elu_plus_one(k)

        if self.memories is None:
            self.reset_memory(k.device)

        assert self.memories is not None and self.memory_norms is not None
        assert self.frozen_memories is not None

        idx = int(self.current_memory_idx.item())

        # Skip frozen memories
        if self.frozen_memories[idx]:
            found = False
            for offset in range(1, self.num_memories + 1):
                candidate = (idx + offset) % self.num_memories
                if not self.frozen_memories[candidate]:
                    idx = candidate
                    self.current_memory_idx = torch.tensor(idx, device=k.device)
                    found = True
                    break
            if not found:
                return  # All memories frozen

        memory = self.memories[idx]
        memory_norm = self.memory_norms[idx]
        batch_size, seq_len, _ = k.shape

        if self.use_delta_rule:
            retrieved_unnorm = torch.matmul(sigma_k, memory)
            norm = torch.matmul(sigma_k, memory_norm).clamp(min=1e-6).unsqueeze(-1)
            delta_v = v - retrieved_unnorm / norm
            memory_update = torch.einsum('bsd,bse->de', sigma_k, delta_v) / (batch_size * seq_len)
        else:
            memory_update = torch.einsum('bsd,bse->de', sigma_k, v) / (batch_size * seq_len)

        self.memories[idx] = (memory + memory_update).detach()
        z_update = sigma_k.sum(dim=(0, 1)) / batch_size
        self.memory_norms[idx] = (memory_norm + z_update).detach()

        # Round-robin for multi-memory
        if self.num_memories > 1:
            next_idx = (idx + 1) % self.num_memories
            for _ in range(self.num_memories):
                if not self.frozen_memories[next_idx]:
                    break
                next_idx = (next_idx + 1) % self.num_memories
            self.current_memory_idx = torch.tensor(next_idx, device=k.device)

    # =========================================================================
    # Freeze / Unfreeze Methods
    # =========================================================================

    def freeze_memory(self, memory_indices: Optional[list[int]] = None) -> None:
        """Freeze memories (make them read-only).

        Args:
            memory_indices: List of indices to freeze. If None, freeze all.
        """
        if self.frozen_memories is None:
            self.frozen_memories = [False] * self.num_memories

        if memory_indices is None:
            memory_indices = list(range(self.num_memories))

        for idx in memory_indices:
            if 0 <= idx < self.num_memories:
                self.frozen_memories[idx] = True

    def unfreeze_memory(self, memory_indices: Optional[list[int]] = None) -> None:
        """Unfreeze memories (make them writable).

        Args:
            memory_indices: List of indices to unfreeze. If None, unfreeze all.
        """
        if self.frozen_memories is None:
            self.frozen_memories = [False] * self.num_memories
            return

        if memory_indices is None:
            memory_indices = list(range(self.num_memories))

        for idx in memory_indices:
            if 0 <= idx < self.num_memories:
                self.frozen_memories[idx] = False

    def is_frozen(self, memory_idx: int) -> bool:
        """Check if a memory is frozen."""
        if self.frozen_memories is None:
            return False
        return self.frozen_memories[memory_idx]

    # =========================================================================
    # Export / Import Methods
    # =========================================================================

    def export_memory(
        self,
        memory_indices: Optional[list[int]] = None,
    ) -> dict:
        """Export memories for sharing with other models.

        Args:
            memory_indices: List of indices to export. If None, export all.

        Returns:
            Dictionary containing memory data.
        """
        if self.memories is None or self.memory_norms is None:
            raise ValueError("No memory to export. Call forward() first.")

        if memory_indices is None:
            memory_indices = list(range(self.num_memories))

        return {
            "memories": {i: self.memories[i].cpu().clone() for i in memory_indices},
            "memory_norms": {i: self.memory_norms[i].cpu().clone() for i in memory_indices},
            "hidden_size": self.hidden_size,
            "memory_head_dim": self.memory_head_dim,
        }

    def import_memory(
        self,
        memory_data: dict,
        memory_indices: Optional[list[int]] = None,
        freeze: bool = True,
    ) -> None:
        """Import memory from another model or saved state.

        Args:
            memory_data: Dictionary from export_memory()
            memory_indices: Target indices. If None, use source indices.
            freeze: Whether to freeze imported memories
        """
        if memory_data["hidden_size"] != self.hidden_size:
            raise ValueError(
                f"Hidden size mismatch: expected {self.hidden_size}, "
                f"got {memory_data['hidden_size']}"
            )

        device = self.w_q.weight.device
        dtype = self.w_q.weight.dtype

        if self.memories is None:
            self.reset_memory(device)

        assert self.memories is not None and self.memory_norms is not None
        assert self.frozen_memories is not None

        source_indices = list(memory_data["memories"].keys())
        if memory_indices is None:
            memory_indices = source_indices

        if len(memory_indices) != len(source_indices):
            raise ValueError(
                f"Indices count mismatch: {len(memory_indices)} targets "
                f"for {len(source_indices)} sources"
            )

        for src_idx, tgt_idx in zip(source_indices, memory_indices):
            if tgt_idx >= self.num_memories:
                raise ValueError(f"Target index {tgt_idx} out of range")

            self.memories[tgt_idx] = memory_data["memories"][src_idx].to(device, dtype)
            self.memory_norms[tgt_idx] = memory_data["memory_norms"][src_idx].to(device, dtype)

            if freeze:
                self.frozen_memories[tgt_idx] = True

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
            "frozen_memories": self.frozen_memories.copy() if self.frozen_memories else None,
            "current_memory_idx": self.current_memory_idx.cpu().clone(),
        }

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = self.w_q.weight.device
        if state["memories"]:
            self.memories = [m.to(device) for m in state["memories"]]
        if state["memory_norms"]:
            self.memory_norms = [n.to(device) for n in state["memory_norms"]]
        if state.get("frozen_memories"):
            self.frozen_memories = state["frozen_memories"].copy()
        else:
            self.frozen_memories = [False] * self.num_memories
        self.current_memory_idx = state["current_memory_idx"].to(device)


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
