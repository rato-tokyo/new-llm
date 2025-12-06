"""
Infini-Attention Layer (Memory + Linear Attention)

Features:
- Single-head memory for expressiveness
- Delta Rule for memory update
- Multi-bank support with freeze/unfreeze
- Memory export/import for sharing between models
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
    - Multi-bank support with freeze capability
    - Memory sharing between models

    Memory Banks:
    - Each bank can be independently frozen/unfrozen
    - Frozen banks are read-only (used as knowledge base)
    - Unfrozen banks are updated during forward pass (working memory)

    Usage:
        # Create model with 2 memory banks
        model = create_model("infini", num_memory_banks=2)

        # Write knowledge to bank 0
        for batch in knowledge_data:
            model(batch, update_memory=True)

        # Freeze bank 0 as knowledge base
        model.freeze_memory(bank_indices=[0])

        # Bank 1 remains as working memory
        # Export frozen knowledge for sharing
        knowledge = model.export_memory(bank_indices=[0])
        torch.save(knowledge, "knowledge.pt")

        # Another model can import the knowledge
        other_model.import_memory(knowledge, bank_indices=[0], freeze=True)
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
        self.frozen_banks: Optional[list[bool]] = None  # Track frozen state per bank
        self.register_buffer('current_bank', torch.tensor(0))
        self.register_buffer('segment_counter', torch.tensor(0))

    def reset_memory(
        self,
        device: Optional[torch.device] = None,
        keep_frozen: bool = True,
    ) -> None:
        """
        Reset memory banks.

        Args:
            device: Device to create tensors on
            keep_frozen: If True, only reset unfrozen banks. If False, reset all.
        """
        if device is None:
            device = self.w_q.weight.device
        dtype = self.w_q.weight.dtype

        if self.memories is None or not keep_frozen:
            # Full reset
            self.memories = [
                torch.zeros(self.memory_head_dim, self.memory_head_dim, device=device, dtype=dtype)
                for _ in range(self.num_memory_banks)
            ]
            self.memory_norms = [
                torch.zeros(self.memory_head_dim, device=device, dtype=dtype)
                for _ in range(self.num_memory_banks)
            ]
            self.frozen_banks = [False] * self.num_memory_banks
        else:
            # Reset only unfrozen banks
            assert self.memories is not None and self.memory_norms is not None and self.frozen_banks is not None
            for i in range(self.num_memory_banks):
                if not self.frozen_banks[i]:
                    self.memories[i] = torch.zeros(
                        self.memory_head_dim, self.memory_head_dim, device=device, dtype=dtype
                    )
                    self.memory_norms[i] = torch.zeros(
                        self.memory_head_dim, device=device, dtype=dtype
                    )

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
        assert self.frozen_banks is not None

        bank_idx = int(self.current_bank.item())

        # Skip frozen banks - find next unfrozen bank
        if self.frozen_banks[bank_idx]:
            # Find next unfrozen bank
            found = False
            for offset in range(1, self.num_memory_banks + 1):
                candidate = (bank_idx + offset) % self.num_memory_banks
                if not self.frozen_banks[candidate]:
                    bank_idx = candidate
                    self.current_bank = torch.tensor(bank_idx, device=k.device)
                    found = True
                    break
            if not found:
                # All banks frozen, skip update
                return

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
                # Find next unfrozen bank
                next_bank = (bank_idx + 1) % self.num_memory_banks
                for _ in range(self.num_memory_banks):
                    if not self.frozen_banks[next_bank]:
                        break
                    next_bank = (next_bank + 1) % self.num_memory_banks
                self.current_bank = torch.tensor(next_bank, device=k.device)

    # =========================================================================
    # Freeze / Unfreeze Methods
    # =========================================================================

    def freeze_memory(self, bank_indices: Optional[list[int]] = None) -> None:
        """
        Freeze memory banks (make them read-only).

        Args:
            bank_indices: List of bank indices to freeze. If None, freeze all.
        """
        if self.frozen_banks is None:
            self.frozen_banks = [False] * self.num_memory_banks

        if bank_indices is None:
            bank_indices = list(range(self.num_memory_banks))

        for idx in bank_indices:
            if 0 <= idx < self.num_memory_banks:
                self.frozen_banks[idx] = True

    def unfreeze_memory(self, bank_indices: Optional[list[int]] = None) -> None:
        """
        Unfreeze memory banks (make them writable).

        Args:
            bank_indices: List of bank indices to unfreeze. If None, unfreeze all.
        """
        if self.frozen_banks is None:
            self.frozen_banks = [False] * self.num_memory_banks
            return

        if bank_indices is None:
            bank_indices = list(range(self.num_memory_banks))

        for idx in bank_indices:
            if 0 <= idx < self.num_memory_banks:
                self.frozen_banks[idx] = False

    def is_frozen(self, bank_idx: int) -> bool:
        """Check if a bank is frozen."""
        if self.frozen_banks is None:
            return False
        return self.frozen_banks[bank_idx]

    # =========================================================================
    # Export / Import Methods for Memory Sharing
    # =========================================================================

    def export_memory(
        self,
        bank_indices: Optional[list[int]] = None,
    ) -> dict:
        """
        Export memory banks for sharing with other models.

        Args:
            bank_indices: List of bank indices to export. If None, export all.

        Returns:
            Dictionary containing memory data that can be saved and shared.
        """
        if self.memories is None or self.memory_norms is None:
            raise ValueError("No memory to export. Call forward() first.")

        if bank_indices is None:
            bank_indices = list(range(self.num_memory_banks))

        return {
            "memories": {i: self.memories[i].cpu().clone() for i in bank_indices},
            "memory_norms": {i: self.memory_norms[i].cpu().clone() for i in bank_indices},
            "hidden_size": self.hidden_size,
            "memory_head_dim": self.memory_head_dim,
        }

    def import_memory(
        self,
        memory_data: dict,
        bank_indices: Optional[list[int]] = None,
        freeze: bool = True,
    ) -> None:
        """
        Import memory from another model or saved state.

        Args:
            memory_data: Dictionary from export_memory()
            bank_indices: Target bank indices to import into. If None, use source indices.
            freeze: Whether to freeze imported banks (recommended for knowledge)
        """
        # Validate compatibility
        if memory_data["hidden_size"] != self.hidden_size:
            raise ValueError(
                f"Hidden size mismatch: expected {self.hidden_size}, "
                f"got {memory_data['hidden_size']}"
            )

        device = self.w_q.weight.device
        dtype = self.w_q.weight.dtype

        # Initialize if needed
        if self.memories is None:
            self.reset_memory(device)

        assert self.memories is not None and self.memory_norms is not None
        assert self.frozen_banks is not None

        source_indices = list(memory_data["memories"].keys())
        if bank_indices is None:
            bank_indices = source_indices

        if len(bank_indices) != len(source_indices):
            raise ValueError(
                f"Bank indices count mismatch: {len(bank_indices)} targets "
                f"for {len(source_indices)} sources"
            )

        for src_idx, tgt_idx in zip(source_indices, bank_indices):
            if tgt_idx >= self.num_memory_banks:
                raise ValueError(f"Target bank index {tgt_idx} out of range")

            self.memories[tgt_idx] = memory_data["memories"][src_idx].to(device, dtype)
            self.memory_norms[tgt_idx] = memory_data["memory_norms"][src_idx].to(device, dtype)

            if freeze:
                self.frozen_banks[tgt_idx] = True

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
            "frozen_banks": self.frozen_banks.copy() if self.frozen_banks else None,
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
        if state.get("frozen_banks"):
            self.frozen_banks = state["frozen_banks"].copy()
        else:
            self.frozen_banks = [False] * self.num_memory_banks
        self.current_bank = state["current_bank"].to(device)
        self.segment_counter = state["segment_counter"].to(device)


class InfiniLayer(BaseLayer):
    """
    Infini-Attention Transformer Layer

    Features:
    - Compressive memory (single-head)
    - Linear Attention for local context
    - No position encoding (NoPE)
    - Freeze/unfreeze memory banks
    - Export/import memory for sharing
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

    def reset_memory(self, device: Optional[torch.device] = None, keep_frozen: bool = True) -> None:
        self.attention.reset_memory(device, keep_frozen)

    def get_memory_state(self) -> dict:
        return self.attention.get_memory_state()

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        self.attention.set_memory_state(state, device)

    # Freeze / Unfreeze
    def freeze_memory(self, bank_indices: Optional[list[int]] = None) -> None:
        self.attention.freeze_memory(bank_indices)

    def unfreeze_memory(self, bank_indices: Optional[list[int]] = None) -> None:
        self.attention.unfreeze_memory(bank_indices)

    def is_frozen(self, bank_idx: int) -> bool:
        return self.attention.is_frozen(bank_idx)

    # Export / Import
    def export_memory(self, bank_indices: Optional[list[int]] = None) -> dict:
        return self.attention.export_memory(bank_indices)

    def import_memory(
        self,
        memory_data: dict,
        bank_indices: Optional[list[int]] = None,
        freeze: bool = True,
    ) -> None:
        self.attention.import_memory(memory_data, bank_indices, freeze)
