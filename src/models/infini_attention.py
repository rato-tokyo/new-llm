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
    ):
        """
        Args:
            hidden_size: 隠れ層の次元
            num_heads: アテンションヘッド数
            use_delta_rule: Delta ruleを使用するか（よりスマートなメモリ更新）
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_delta_rule = use_delta_rule

        # Q, K, V projections
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output projection
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

        # Beta gate (learnable, per head)
        # sigmoid(beta) で local と memory の重みを決定
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

        # 2. Memory attention
        a_mem = self._retrieve_from_memory(q)  # [batch, heads, seq, head_dim]

        # 3. Combine with beta gate
        # gate: sigmoid(beta) per head
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
    ):
        super().__init__()

        self.input_layernorm = nn.LayerNorm(hidden_size)

        self.attention = InfiniAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            use_delta_rule=use_delta_rule,
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
