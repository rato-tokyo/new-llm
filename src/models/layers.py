"""
Transformer Layer Implementations

全レイヤータイプを統合:
- PythiaLayer: 標準Pythia (RoPE + Softmax Attention)
- InfiniLayer: Infini-Attention (Memory + Linear Attention)
- MultiMemoryLayer: 複数独立メモリ
- HierarchicalLayer: 階層的メモリ

使用例:
    from src.models.layers import PythiaLayer, InfiniLayer

    # 標準Pythiaレイヤー
    layer = PythiaLayer(hidden_size=512, num_heads=8, intermediate_size=2048)

    # Infini-Attentionレイヤー
    layer = InfiniLayer(hidden_size=512, num_heads=8, intermediate_size=2048)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_components import PythiaMLP
from src.models.memory_utils import causal_linear_attention, elu_plus_one
from src.models.position_encoding import RotaryEmbedding, apply_rotary_pos_emb


# =============================================================================
# Base Layer Protocol
# =============================================================================

class BaseLayer(nn.Module):
    """
    全レイヤーの基底クラス

    共通インターフェース:
    - forward(hidden_states, attention_mask, **kwargs) -> Tensor
    - reset_memory() (メモリ系レイヤーのみ)
    - get_memory_state() / set_memory_state() (メモリ系レイヤーのみ)
    """

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        """メモリをリセット（メモリ系レイヤーでオーバーライド）"""
        pass

    def get_memory_state(self) -> Optional[dict]:
        """メモリ状態を取得（メモリ系レイヤーでオーバーライド）"""
        return None

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        """メモリ状態を設定（メモリ系レイヤーでオーバーライド）"""
        pass


# =============================================================================
# Standard Pythia Layer (RoPE + Softmax Attention)
# =============================================================================

class PythiaAttention(nn.Module):
    """Pythia Multi-Head Attention with Rotary Embedding"""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_dim = int(self.head_dim * rotary_pct)

        self.query_key_value = nn.Linear(hidden_size, 3 * hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

        self.rotary_emb = RotaryEmbedding(
            self.rotary_dim,
            max_position_embeddings=max_position_embeddings,
        )
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        cos, sin = self.rotary_emb(query, seq_len)
        cos = cos.squeeze(0).squeeze(0).unsqueeze(0).unsqueeze(0)
        sin = sin.squeeze(0).squeeze(0).unsqueeze(0).unsqueeze(0)

        query_rot, key_rot = apply_rotary_pos_emb(
            query[..., :self.rotary_dim],
            key[..., :self.rotary_dim],
            cos, sin
        )
        query = torch.cat([query_rot, query[..., self.rotary_dim:]], dim=-1)
        key = torch.cat([key_rot, key[..., self.rotary_dim:]], dim=-1)

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        if attention_mask is None:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device) * float("-inf"),
                diagonal=1
            )
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        return self.dense(attn_output)


class PythiaLayer(BaseLayer):
    """
    Standard Pythia Transformer Layer

    Features:
    - RoPE (Rotary Position Embedding)
    - Softmax Attention
    - Parallel Attention + MLP
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        rotary_pct: float = 0.25,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.attention = PythiaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            rotary_pct=rotary_pct,
            max_position_embeddings=max_position_embeddings,
        )
        self.mlp = PythiaMLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.attention(hidden_states, attention_mask)
        mlp_output = self.mlp(hidden_states)
        return residual + attn_output + mlp_output


# =============================================================================
# Infini-Attention Layer (Memory + Linear Attention)
# =============================================================================

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


# =============================================================================
# Multi-Memory Layer (Multiple Independent Memories)
# =============================================================================

class MultiMemoryAttention(nn.Module):
    """Multiple Independent Memories with Attention-based Selection"""

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
                a_mem = torch.einsum('bhsd,hde->bhse', sigma_q, memory)
                rel = torch.einsum('bhsd,hd->bhs', sigma_q, memory_norm)
                outputs.append(a_mem / rel.clamp(min=1e-6).unsqueeze(-1))
                relevances.append(rel.mean(dim=-1))

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
    """Multi-Memory Infini-Attention Layer"""

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


# =============================================================================
# Hierarchical Memory Layer
# =============================================================================

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
