"""
Infini-Attention Implementation (Memory-Only, Single Head)

"Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"
https://arxiv.org/abs/2404.07143

Memory-Only版: Local Attentionを使用せず、圧縮メモリのみで動作。
より高速で、メモリの効果を直接測定可能。

重要: memory_head_dim=hidden_size（シングルヘッド）で最大の表現力を確保。
小さいhead_dim（例: 64）だとキーベクトルが直交しやすく、メモリが機能しない。

メモリ更新 (Delta Rule):
  M_s = M_{s-1} + σ(K)^T @ (V - σ(K) @ M_{s-1} / σ(K) @ z_{s-1})

メモリ取得:
  A_mem = σ(Q) @ M_{s-1} / (σ(Q) @ z_{s-1})

σ(x) = ELU(x) + 1
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
    Infini-Attention Module with Causal Linear Attention

    圧縮メモリ + 現在セグメント内のCausal Linear Attentionを使用。
    Multi-Memory Bank対応（num_memory_banks >= 1）。

    特徴:
    - 過去セグメント: 圧縮メモリから取得
    - 現在セグメント: Causal Linear Attentionで計算
    - 両者を組み合わせて出力
    - Delta Ruleによる効率的なメモリ更新

    重要: memory_head_dimを大きく設定することで、メモリの表現力を向上。
    デフォルトはhidden_size（シングルヘッド相当）で最大の表現力を確保。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_memory_banks: int = 1,
        segments_per_bank: int = 4,
        use_delta_rule: bool = True,
        memory_head_dim: Optional[int] = None,
    ):
        """
        Args:
            hidden_size: 隠れ層の次元
            num_heads: アテンションヘッド数（出力用、メモリには影響しない）
            num_memory_banks: メモリバンク数（1=シングル、2以上=マルチ）
            segments_per_bank: 各バンクに蓄積するセグメント数
            use_delta_rule: Delta ruleを使用するか
            memory_head_dim: メモリのhead次元（デフォルト: hidden_size、シングルヘッド）
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        # メモリはシングルヘッド（head_dim=hidden_size）で最大表現力を確保
        # memory_head_dimを指定可能にして柔軟性を持たせる
        self.memory_head_dim = memory_head_dim if memory_head_dim is not None else hidden_size
        self.num_memory_banks = num_memory_banks
        self.segments_per_bank = segments_per_bank
        self.use_delta_rule = use_delta_rule

        # Q, K, V projections for memory (memory_head_dim次元)
        self.w_q = nn.Linear(hidden_size, self.memory_head_dim, bias=False)
        self.w_k = nn.Linear(hidden_size, self.memory_head_dim, bias=False)
        self.w_v = nn.Linear(hidden_size, self.memory_head_dim, bias=False)

        # Output projection (memory_head_dim -> hidden_size)
        self.w_o = nn.Linear(self.memory_head_dim, hidden_size, bias=False)

        # Learnable gate to combine memory and local attention (scalar)
        self.gate = nn.Parameter(torch.zeros(1))

        # Bank aggregation weights (learnable, per bank)
        if num_memory_banks > 1:
            self.bank_weights = nn.Parameter(torch.zeros(num_memory_banks))
        else:
            self.register_buffer('bank_weights', None)

        # Memory banks: [memory_head_dim, memory_head_dim] per bank
        self.memories: Optional[list[torch.Tensor]] = None
        self.memory_norms: Optional[list[torch.Tensor]] = None

        # Current bank index and segment counter (for multi-bank)
        self.register_buffer('current_bank', torch.tensor(0))
        self.register_buffer('segment_counter', torch.tensor(0))

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        """全メモリバンクをリセット"""
        if device is None:
            device = self.w_q.weight.device
        dtype = self.w_q.weight.dtype

        # シングルヘッド: [memory_head_dim, memory_head_dim]
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

    def _retrieve_from_single_memory(
        self,
        q: torch.Tensor,
        memory: torch.Tensor,
        memory_norm: torch.Tensor,
    ) -> torch.Tensor:
        """
        単一メモリバンクから取得（シングルヘッド版）

        Args:
            q: [batch, seq, memory_head_dim]
            memory: [memory_head_dim, memory_head_dim]
            memory_norm: [memory_head_dim]

        Returns:
            output: [batch, seq, memory_head_dim]
        """
        sigma_q = elu_plus_one(q)  # [B, S, D]

        if memory_norm.sum() < 1e-6:
            return torch.zeros_like(q)

        # σ(Q) @ M: [B, S, D] @ [D, D] -> [B, S, D]
        a_mem_unnorm = torch.matmul(sigma_q, memory)
        # σ(Q) @ z: [B, S, D] @ [D] -> [B, S]
        norm = torch.matmul(sigma_q, memory_norm)
        norm = norm.clamp(min=1e-6).unsqueeze(-1)  # [B, S, 1]

        return a_mem_unnorm / norm

    def _retrieve_from_memory(self, q: torch.Tensor) -> torch.Tensor:
        """メモリから情報を取得（シングル/マルチバンク対応）"""
        if self.memories is None or self.memory_norms is None:
            return torch.zeros_like(q)

        if self.num_memory_banks == 1:
            return self._retrieve_from_single_memory(
                q, self.memories[0], self.memory_norms[0]
            )

        # Multi-bank: 全バンクから取得して統合
        bank_results = []
        for memory, memory_norm in zip(self.memories, self.memory_norms):
            a_mem = self._retrieve_from_single_memory(q, memory, memory_norm)
            bank_results.append(a_mem)

        # [num_banks, B, S, D] -> weighted sum
        stacked = torch.stack(bank_results, dim=0)  # [num_banks, B, S, D]
        weights = F.softmax(self.bank_weights, dim=-1)  # [num_banks]
        weights = weights.view(-1, 1, 1, 1)  # [num_banks, 1, 1, 1]
        combined = (stacked * weights).sum(dim=0)  # [B, S, D]

        return combined

    def _update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        現在のメモリバンクを更新（シングルヘッド版）

        Args:
            k: [batch, seq, memory_head_dim]
            v: [batch, seq, memory_head_dim]
        """
        sigma_k = elu_plus_one(k)  # [B, S, D]

        if self.memories is None or self.memory_norms is None:
            self.reset_memory(k.device)

        assert self.memories is not None and self.memory_norms is not None

        bank_idx = int(self.current_bank.item())
        memory = self.memories[bank_idx]  # [D, D]
        memory_norm = self.memory_norms[bank_idx]  # [D]

        batch_size, seq_len, _ = k.shape

        if self.use_delta_rule:
            # σ(K) @ M: [B, S, D] @ [D, D] -> [B, S, D]
            retrieved_unnorm = torch.matmul(sigma_k, memory)
            # σ(K) @ z: [B, S, D] @ [D] -> [B, S]
            norm = torch.matmul(sigma_k, memory_norm)
            norm = norm.clamp(min=1e-6).unsqueeze(-1)  # [B, S, 1]
            retrieved = retrieved_unnorm / norm

            delta_v = v - retrieved  # [B, S, D]
            # σ(K)^T @ delta_V: sum over batch and seq
            # [B, S, D]^T @ [B, S, D] -> [D, D]
            memory_update = torch.einsum('bsd,bse->de', sigma_k, delta_v)
            memory_update = memory_update / (batch_size * seq_len)
        else:
            # σ(K)^T @ V: [B, S, D]^T @ [B, S, D] -> [D, D]
            memory_update = torch.einsum('bsd,bse->de', sigma_k, v)
            memory_update = memory_update / (batch_size * seq_len)

        self.memories[bank_idx] = (memory + memory_update).detach()

        # 正規化項の更新: sum over batch and seq
        z_update = sigma_k.sum(dim=(0, 1))  # [D]
        z_update = z_update / batch_size
        self.memory_norms[bank_idx] = (memory_norm + z_update).detach()

        # マルチバンク: セグメントカウントしてバンク切り替え
        if self.num_memory_banks > 1:
            self.segment_counter = self.segment_counter + 1
            if self.segment_counter >= self.segments_per_bank:
                self.segment_counter = torch.tensor(0, device=k.device)
                self.current_bank = (self.current_bank + 1) % self.num_memory_banks

    def _causal_linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        現在のセグメント内でCausal Linear Attentionを計算（シングルヘッド版）

        累積和を使用してO(n)で計算。各位置iは位置0~iのK,Vにのみアテンション可能。

        Args:
            q: [batch, seq, memory_head_dim]
            k: [batch, seq, memory_head_dim]
            v: [batch, seq, memory_head_dim]

        Returns:
            output: [batch, seq, memory_head_dim]
        """
        sigma_q = elu_plus_one(q)  # [B, S, D]
        sigma_k = elu_plus_one(k)  # [B, S, D]

        # 累積的にK^T @ Vを計算: [B, S, D, D]
        # kv_cumsum[i] = sum_{j=0}^{i} K_j^T @ V_j
        kv = torch.einsum('bsd,bse->bsde', sigma_k, v)  # [B, S, D, D]
        kv_cumsum = torch.cumsum(kv, dim=1)  # [B, S, D, D]

        # 累積的にKを計算（正規化用）: [B, S, D]
        k_cumsum = torch.cumsum(sigma_k, dim=1)  # [B, S, D]

        # Q @ (cumsum K^T V) / (Q @ cumsum K)
        numerator = torch.einsum('bsd,bsde->bse', sigma_q, kv_cumsum)  # [B, S, D]
        denominator = torch.einsum('bsd,bsd->bs', sigma_q, k_cumsum)  # [B, S]
        denominator = denominator.clamp(min=1e-6).unsqueeze(-1)  # [B, S, 1]

        output = numerator / denominator  # [B, S, D]

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass with Memory + Causal Linear Attention（シングルヘッド版）

        Args:
            hidden_states: [batch, seq, hidden_size]
            attention_mask: Optional attention mask (未使用)
            update_memory: メモリを更新するか

        Returns:
            output: [batch, seq, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Q, K, V projection: [B, S, hidden_size] -> [B, S, memory_head_dim]
        q = self.w_q(hidden_states)
        k = self.w_k(hidden_states)
        v = self.w_v(hidden_states)

        # 1. 過去のメモリから取得
        memory_output = self._retrieve_from_memory(q)

        # 2. 現在のセグメント内でCausal Linear Attention
        local_output = self._causal_linear_attention(q, k, v)

        # 3. 学習可能なゲートで組み合わせ（スカラー）
        gate = torch.sigmoid(self.gate)  # [1] -> scalar
        output = gate * memory_output + (1 - gate) * local_output

        # 4. メモリ更新
        if update_memory:
            self._update_memory(k, v)

        # Output projection: [B, S, memory_head_dim] -> [B, S, hidden_size]
        output = self.w_o(output)

        return output

    def get_bank_weights(self) -> Optional[torch.Tensor]:
        """各バンクの重みを取得（マルチバンクのみ）"""
        if self.bank_weights is not None:
            return F.softmax(self.bank_weights, dim=-1)
        return None

    def memory_info(self) -> dict:
        """メモリ情報を取得"""
        # シングルヘッド: [memory_head_dim, memory_head_dim]
        single_bank_size = self.memory_head_dim * self.memory_head_dim * 4  # float32 = 4 bytes
        single_norm_size = self.memory_head_dim * 4
        total_per_bank = single_bank_size + single_norm_size
        return {
            "num_banks": self.num_memory_banks,
            "memory_head_dim": self.memory_head_dim,
            "bytes_per_bank": total_per_bank,
            "total_bytes": total_per_bank * self.num_memory_banks,
            "current_bank": self.current_bank.item() if self.current_bank is not None else 0,
            "segment_counter": self.segment_counter.item() if self.segment_counter is not None else 0,
        }

    def get_memory_state(self) -> dict:
        """
        メモリ状態を取得（転送可能な形式）

        Returns:
            dict: メモリ状態（CPU上のテンソル）
        """
        state = {
            "memories": [m.cpu().clone() for m in self.memories] if self.memories else None,
            "memory_norms": [n.cpu().clone() for n in self.memory_norms] if self.memory_norms else None,
            "current_bank": self.current_bank.cpu().clone(),
            "segment_counter": self.segment_counter.cpu().clone(),
        }
        return state

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        """
        メモリ状態を設定

        Args:
            state: get_memory_state()で取得した状態
            device: 転送先デバイス（Noneの場合はモデルのデバイス）
        """
        if device is None:
            device = self.w_q.weight.device

        if state["memories"] is not None:
            self.memories = [m.to(device) for m in state["memories"]]
        if state["memory_norms"] is not None:
            self.memory_norms = [n.to(device) for n in state["memory_norms"]]
        self.current_bank = state["current_bank"].to(device)
        self.segment_counter = state["segment_counter"].to(device)


class InfiniAttentionLayer(nn.Module):
    """
    Infini-Attention Transformer Layer

    Pre-LayerNorm + Parallel Attention/MLP (Pythia style)

    シングルヘッド（memory_head_dim=hidden_size）で最大の表現力を確保。
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

    def get_memory_state(self) -> dict:
        """メモリ状態を取得"""
        return self.attention.get_memory_state()

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        """メモリ状態を設定"""
        self.attention.set_memory_state(state, device)
