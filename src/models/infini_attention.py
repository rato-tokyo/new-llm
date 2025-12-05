"""
Infini-Attention Implementation

"Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"
https://arxiv.org/abs/2404.07143

Infini-Attentionは圧縮メモリを持つアテンション機構:
- ローカルアテンション（通常のdot-product attention）
- メモリアテンション（線形注意による長期記憶）
- β ゲートで両者を結合

メモリ更新:
  M_s = M_{s-1} + σ(K)^T V  (シンプル版)
  M_s = M_{s-1} + σ(K)^T (V - σ(K) M_{s-1} / σ(K) z_{s-1})  (delta rule)

メモリ取得:
  A_mem = σ(Q) M_{s-1} / σ(Q) z_{s-1}

結合:
  A = sigmoid(β) * A_mem + (1 - sigmoid(β)) * A_dot

ここで σ(x) = ELU(x) + 1
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """ELU + 1 activation (ensures positivity for linear attention)"""
    return F.elu(x) + 1.0


class InfiniAttention(nn.Module):
    """
    Infini-Attention Module

    圧縮メモリを持つアテンション機構。
    1層目（NoPE）での使用を想定。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        use_delta_rule: bool = True,
        memory_only: bool = False,
    ):
        """
        Args:
            hidden_size: 隠れ層の次元
            num_heads: アテンションヘッド数
            use_delta_rule: Delta ruleを使用するか（よりスマートなメモリ更新）
            memory_only: Trueの場合、Memory Attentionのみ使用（Local Attentionなし）
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_delta_rule = use_delta_rule
        self.memory_only = memory_only

        # Q, K, V projections
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output projection
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

        # Beta gate (learnable, per head)
        # sigmoid(beta) で local と memory の重みを決定
        # memory_only=True の場合、betaは使用されないがget_gate_values()のため保持
        if memory_only:
            self.register_buffer('beta', torch.full((num_heads,), 10.0))
        else:
            self.beta = nn.Parameter(torch.zeros(num_heads))

        # Scaling for dot-product attention
        self.scale = self.head_dim ** -0.5

        # Memory state (not a parameter, managed externally or via reset)
        # M: [num_heads, head_dim, head_dim]
        # z: [num_heads, head_dim]
        self.register_buffer('memory', None)
        self.register_buffer('memory_norm', None)

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        """メモリをリセット"""
        if device is None:
            device = self.w_q.weight.device

        self.memory = torch.zeros(
            self.num_heads, self.head_dim, self.head_dim,
            device=device, dtype=self.w_q.weight.dtype
        )
        self.memory_norm = torch.zeros(
            self.num_heads, self.head_dim,
            device=device, dtype=self.w_q.weight.dtype
        )

    def _retrieve_from_memory(
        self,
        q: torch.Tensor,  # [batch, heads, seq, head_dim]
    ) -> torch.Tensor:
        """
        メモリから情報を取得

        A_mem = σ(Q) @ M / (σ(Q) @ z)
        """
        # σ(Q): [batch, heads, seq, head_dim]
        sigma_q = elu_plus_one(q)

        if self.memory is None or self.memory_norm is None:
            # メモリが未初期化の場合はゼロを返す
            return torch.zeros_like(q)

        # A_mem_unnorm = σ(Q) @ M
        # [batch, heads, seq, head_dim] @ [heads, head_dim, head_dim]
        # -> [batch, heads, seq, head_dim]
        a_mem_unnorm = torch.einsum('bhsd,hde->bhse', sigma_q, self.memory)

        # normalization = σ(Q) @ z
        # [batch, heads, seq, head_dim] @ [heads, head_dim]
        # -> [batch, heads, seq]
        norm = torch.einsum('bhsd,hd->bhs', sigma_q, self.memory_norm)

        # Avoid division by zero
        norm = norm.clamp(min=1e-6).unsqueeze(-1)

        # A_mem = A_mem_unnorm / norm
        a_mem = a_mem_unnorm / norm

        return a_mem

    def _update_memory(
        self,
        k: torch.Tensor,  # [batch, heads, seq, head_dim]
        v: torch.Tensor,  # [batch, heads, seq, head_dim]
    ) -> None:
        """
        メモリを更新

        Simple: M = M + σ(K)^T @ V
        Delta:  M = M + σ(K)^T @ (V - retrieved_V)
        """
        # σ(K): [batch, heads, seq, head_dim]
        sigma_k = elu_plus_one(k)

        if self.memory is None or self.memory_norm is None:
            self.reset_memory(k.device)

        if self.use_delta_rule:
            # Delta rule: 既存の取得値を引く
            # retrieved = σ(K) @ M / (σ(K) @ z)
            retrieved_unnorm = torch.einsum('bhsd,hde->bhse', sigma_k, self.memory)
            norm = torch.einsum('bhsd,hd->bhs', sigma_k, self.memory_norm)
            norm = norm.clamp(min=1e-6).unsqueeze(-1)
            retrieved = retrieved_unnorm / norm

            # delta_v = V - retrieved
            delta_v = v - retrieved

            # M = M + σ(K)^T @ delta_v
            # Average over batch and sum over sequence
            # [batch, heads, seq, head_dim]^T @ [batch, heads, seq, head_dim]
            # = [heads, head_dim, head_dim]
            memory_update = torch.einsum('bhsd,bhse->hde', sigma_k, delta_v)
            memory_update = memory_update / (k.size(0) * k.size(2))  # normalize by batch * seq
        else:
            # Simple: M = M + σ(K)^T @ V
            memory_update = torch.einsum('bhsd,bhse->hde', sigma_k, v)
            memory_update = memory_update / (k.size(0) * k.size(2))

        # Update memory (detach to prevent gradient accumulation across batches)
        self.memory = (self.memory + memory_update).detach()

        # Update normalization term: z = z + sum(σ(K))
        z_update = sigma_k.sum(dim=(0, 2))  # [heads, head_dim]
        z_update = z_update / k.size(0)  # normalize by batch
        self.memory_norm = (self.memory_norm + z_update).detach()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional causal mask
            update_memory: メモリを更新するか（推論時は True）

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Q, K, V projections
        q = self.w_q(hidden_states)
        k = self.w_k(hidden_states)
        v = self.w_v(hidden_states)

        # Reshape to heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Now: [batch, heads, seq, head_dim]

        # Memory attention (always computed)
        a_mem = self._retrieve_from_memory(q)  # [batch, heads, seq, head_dim]

        if self.memory_only:
            # Memory-only mode: skip local attention entirely (faster)
            combined = a_mem
        else:
            # 1. Local attention (standard dot-product, causal)
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            # Causal mask (NoPE, no position encoding)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device) * float("-inf"),
                diagonal=1
            )
            attn_scores = attn_scores + causal_mask

            attn_weights = F.softmax(attn_scores, dim=-1)
            a_local = torch.matmul(attn_weights, v)  # [batch, heads, seq, head_dim]

            # 2. Combine with beta gate
            gate = torch.sigmoid(self.beta)  # [heads]
            gate = gate.view(1, self.num_heads, 1, 1)  # broadcast

            # A = gate * A_mem + (1 - gate) * A_local
            combined = gate * a_mem + (1 - gate) * a_local

        # 4. Update memory (after forward, using current K, V)
        if update_memory:
            self._update_memory(k, v)

        # 5. Output projection
        combined = combined.transpose(1, 2).contiguous()
        combined = combined.view(batch_size, seq_len, self.hidden_size)
        output = self.w_o(combined)

        return output

    def get_gate_values(self) -> torch.Tensor:
        """現在のゲート値を取得（デバッグ用）"""
        return torch.sigmoid(self.beta)


class InfiniAttentionLayer(nn.Module):
    """
    Infini-Attention Transformer Layer

    Pre-LayerNorm + Parallel Attention/MLP (Pythia style)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        use_delta_rule: bool = True,
        memory_only: bool = False,
    ):
        super().__init__()

        self.input_layernorm = nn.LayerNorm(hidden_size)

        self.attention = InfiniAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            use_delta_rule=use_delta_rule,
            memory_only=memory_only,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        # Pre-LayerNorm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Parallel Attention + MLP
        attn_output = self.attention(hidden_states, attention_mask, update_memory)
        mlp_output = self.mlp(hidden_states)

        # Residual
        hidden_states = residual + attn_output + mlp_output

        return hidden_states

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        """メモリをリセット"""
        self.attention.reset_memory(device)


class MultiMemoryInfiniAttention(nn.Module):
    """
    Multi-Memory Bank Infini-Attention

    複数のメモリバンクを持つInfini-Attention。
    情報の混合を防ぎ、より正確な検索を可能にする。

    設計:
    - 複数のメモリバンク（M₀, M₁, ...）を保持
    - セグメントごとに現在のバンクを更新
    - 検索時は全バンクから情報を取得し、Attention weightで統合
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_memory_banks: int = 2,
        segments_per_bank: int = 4,
        use_delta_rule: bool = True,
        memory_only: bool = False,
    ):
        """
        Args:
            hidden_size: 隠れ層の次元
            num_heads: アテンションヘッド数
            num_memory_banks: メモリバンク数（デフォルト: 2）
            segments_per_bank: 各バンクに蓄積するセグメント数（デフォルト: 4）
            use_delta_rule: Delta ruleを使用するか
            memory_only: Memory Attentionのみ使用
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_memory_banks = num_memory_banks
        self.segments_per_bank = segments_per_bank
        self.use_delta_rule = use_delta_rule
        self.memory_only = memory_only

        # Q, K, V projections
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output projection
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

        # Beta gate (learnable, per head)
        if memory_only:
            self.register_buffer('beta', torch.full((num_heads,), 10.0))
        else:
            self.beta = nn.Parameter(torch.zeros(num_heads))

        # Bank aggregation weights (learnable, per head per bank)
        # 各バンクの重要度を学習
        self.bank_weights = nn.Parameter(torch.zeros(num_heads, num_memory_banks))

        # Scaling for dot-product attention
        self.scale = self.head_dim ** -0.5

        # Memory banks: list of (memory, memory_norm) pairs
        # Each memory: [num_heads, head_dim, head_dim]
        # Each memory_norm: [num_heads, head_dim]
        self.memories: Optional[list[torch.Tensor]] = None
        self.memory_norms: Optional[list[torch.Tensor]] = None

        # Current bank index and segment counter
        self.register_buffer('current_bank', torch.tensor(0))
        self.register_buffer('segment_counter', torch.tensor(0))

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        """全メモリバンクをリセット"""
        if device is None:
            device = self.w_q.weight.device
        dtype = self.w_q.weight.dtype

        # Initialize all memory banks
        self.memories = [
            torch.zeros(self.num_heads, self.head_dim, self.head_dim, device=device, dtype=dtype)
            for _ in range(self.num_memory_banks)
        ]
        self.memory_norms = [
            torch.zeros(self.num_heads, self.head_dim, device=device, dtype=dtype)
            for _ in range(self.num_memory_banks)
        ]

        # Reset counters
        self.current_bank = torch.tensor(0, device=device)
        self.segment_counter = torch.tensor(0, device=device)

    def _retrieve_from_single_memory(
        self,
        q: torch.Tensor,  # [batch, heads, seq, head_dim]
        memory: torch.Tensor,  # [heads, head_dim, head_dim]
        memory_norm: torch.Tensor,  # [heads, head_dim]
    ) -> torch.Tensor:
        """単一メモリバンクから取得"""
        sigma_q = elu_plus_one(q)

        # Check if memory is empty
        if memory_norm.sum() < 1e-6:
            return torch.zeros_like(q)

        # A_mem_unnorm = σ(Q) @ M
        a_mem_unnorm = torch.einsum('bhsd,hde->bhse', sigma_q, memory)

        # normalization = σ(Q) @ z
        norm = torch.einsum('bhsd,hd->bhs', sigma_q, memory_norm)
        norm = norm.clamp(min=1e-6).unsqueeze(-1)

        return a_mem_unnorm / norm

    def _retrieve_from_all_memories(
        self,
        q: torch.Tensor,  # [batch, heads, seq, head_dim]
    ) -> torch.Tensor:
        """
        全メモリバンクから取得して統合

        各バンクからの取得結果を学習可能な重みで統合
        """
        if self.memories is None or self.memory_norms is None:
            return torch.zeros_like(q)

        # 各バンクから取得
        bank_results = []
        for memory, memory_norm in zip(self.memories, self.memory_norms):
            a_mem = self._retrieve_from_single_memory(q, memory, memory_norm)
            bank_results.append(a_mem)

        # Stack: [num_banks, batch, heads, seq, head_dim]
        stacked = torch.stack(bank_results, dim=0)

        # Softmax over bank weights: [heads, num_banks] -> [1, 1, heads, 1, 1, num_banks]
        weights = F.softmax(self.bank_weights, dim=-1)  # [heads, num_banks]
        weights = weights.view(1, self.num_heads, 1, 1, self.num_memory_banks)

        # Transpose stacked to [batch, heads, seq, head_dim, num_banks]
        stacked = stacked.permute(1, 2, 3, 4, 0)

        # Weighted sum: [batch, heads, seq, head_dim]
        combined = (stacked * weights).sum(dim=-1)

        return combined

    def _update_current_memory(
        self,
        k: torch.Tensor,  # [batch, heads, seq, head_dim]
        v: torch.Tensor,  # [batch, heads, seq, head_dim]
    ) -> None:
        """現在のメモリバンクを更新"""
        sigma_k = elu_plus_one(k)

        if self.memories is None or self.memory_norms is None:
            self.reset_memory(k.device)

        # Type assertion for mypy (reset_memory ensures these are not None)
        assert self.memories is not None and self.memory_norms is not None

        bank_idx = int(self.current_bank.item())
        memory = self.memories[bank_idx]
        memory_norm = self.memory_norms[bank_idx]

        if self.use_delta_rule:
            # Delta rule
            retrieved_unnorm = torch.einsum('bhsd,hde->bhse', sigma_k, memory)
            norm = torch.einsum('bhsd,hd->bhs', sigma_k, memory_norm)
            norm = norm.clamp(min=1e-6).unsqueeze(-1)
            retrieved = retrieved_unnorm / norm

            delta_v = v - retrieved
            memory_update = torch.einsum('bhsd,bhse->hde', sigma_k, delta_v)
            memory_update = memory_update / (k.size(0) * k.size(2))
        else:
            memory_update = torch.einsum('bhsd,bhse->hde', sigma_k, v)
            memory_update = memory_update / (k.size(0) * k.size(2))

        # Update current bank
        self.memories[bank_idx] = (memory + memory_update).detach()

        z_update = sigma_k.sum(dim=(0, 2))
        z_update = z_update / k.size(0)
        self.memory_norms[bank_idx] = (memory_norm + z_update).detach()

        # Increment segment counter and possibly switch bank
        self.segment_counter = self.segment_counter + 1
        if self.segment_counter >= self.segments_per_bank:
            self.segment_counter = torch.tensor(0, device=k.device)
            self.current_bank = (self.current_bank + 1) % self.num_memory_banks

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """Forward pass"""
        batch_size, seq_len, _ = hidden_states.shape

        # Q, K, V projections
        q = self.w_q(hidden_states)
        k = self.w_k(hidden_states)
        v = self.w_v(hidden_states)

        # Reshape to heads
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Memory attention from all banks
        a_mem = self._retrieve_from_all_memories(q)

        if self.memory_only:
            combined = a_mem
        else:
            # Local attention
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device) * float("-inf"),
                diagonal=1
            )
            attn_scores = attn_scores + causal_mask

            attn_weights = F.softmax(attn_scores, dim=-1)
            a_local = torch.matmul(attn_weights, v)

            # Combine with beta gate
            gate = torch.sigmoid(self.beta)
            gate = gate.view(1, self.num_heads, 1, 1)
            combined = gate * a_mem + (1 - gate) * a_local

        # Update memory
        if update_memory:
            self._update_current_memory(k, v)

        # Output projection
        combined = combined.transpose(1, 2).contiguous()
        combined = combined.view(batch_size, seq_len, self.hidden_size)
        output = self.w_o(combined)

        return output

    def get_gate_values(self) -> torch.Tensor:
        """現在のゲート値を取得"""
        return torch.sigmoid(self.beta)

    def get_bank_weights(self) -> torch.Tensor:
        """各バンクの重みを取得（デバッグ用）"""
        return F.softmax(self.bank_weights, dim=-1)

    def memory_info(self) -> dict:
        """メモリ情報を取得"""
        single_bank_size = self.num_heads * self.head_dim * self.head_dim * 4  # float32
        single_norm_size = self.num_heads * self.head_dim * 4
        total_per_bank = single_bank_size + single_norm_size
        return {
            "num_banks": self.num_memory_banks,
            "bytes_per_bank": total_per_bank,
            "total_bytes": total_per_bank * self.num_memory_banks,
            "current_bank": self.current_bank.item() if self.current_bank is not None else 0,
            "segment_counter": self.segment_counter.item() if self.segment_counter is not None else 0,
        }


class MultiMemoryInfiniAttentionLayer(nn.Module):
    """
    Multi-Memory Infini-Attention Transformer Layer

    Pre-LayerNorm + Parallel Attention/MLP (Pythia style)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_memory_banks: int = 2,
        segments_per_bank: int = 4,
        use_delta_rule: bool = True,
        memory_only: bool = False,
    ):
        super().__init__()

        self.input_layernorm = nn.LayerNorm(hidden_size)

        self.attention = MultiMemoryInfiniAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_memory_banks=num_memory_banks,
            segments_per_bank=segments_per_bank,
            use_delta_rule=use_delta_rule,
            memory_only=memory_only,
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output = self.attention(hidden_states, attention_mask, update_memory)
        mlp_output = self.mlp(hidden_states)

        hidden_states = residual + attn_output + mlp_output

        return hidden_states

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        """メモリをリセット"""
        self.attention.reset_memory(device)
