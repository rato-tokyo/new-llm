"""
Infini-Attention Implementation (Memory-Only)

"Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention"
https://arxiv.org/abs/2404.07143

Memory-Only版: Local Attentionを使用せず、圧縮メモリのみで動作。
より高速で、メモリの効果を直接測定可能。

メモリ更新 (Delta Rule):
  M_s = M_{s-1} + σ(K)^T @ (V - σ(K) @ M_{s-1} / σ(K) @ z_{s-1})

メモリ取得:
  A_mem = σ(Q) @ M_{s-1} / (σ(Q) @ z_{s-1})

σ(x) = ELU(x) + 1

ALiBi版:
  メモリ更新時にALiBi重みを適用（線形化近似）:
  M_φ = Σ_i exp(-slope * d_i) * φ(K_i) * V_i^T

  これにより、遠いセグメントほど重みが小さくなり、
  位置情報がメモリに反映される。
"""

import math

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    """ELU + 1 activation (ensures positivity for linear attention)"""
    return F.elu(x) + 1.0


def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    ALiBiのヘッドごとのスロープを計算

    ALiBi論文に従い、2^(-8/n), 2^(-8*2/n), ..., 2^(-8) のスロープを使用
    nはヘッド数

    Args:
        num_heads: アテンションヘッド数

    Returns:
        slopes: [num_heads] スロープテンソル
    """
    def get_slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        slopes = get_slopes_power_of_2(num_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        slopes = (
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][: num_heads - closest_power_of_2]
        )

    return torch.tensor(slopes, dtype=torch.float32)


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
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_memory_banks: int = 1,
        segments_per_bank: int = 4,
        use_delta_rule: bool = True,
    ):
        """
        Args:
            hidden_size: 隠れ層の次元
            num_heads: アテンションヘッド数
            num_memory_banks: メモリバンク数（1=シングル、2以上=マルチ）
            segments_per_bank: 各バンクに蓄積するセグメント数
            use_delta_rule: Delta ruleを使用するか
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_memory_banks = num_memory_banks
        self.segments_per_bank = segments_per_bank
        self.use_delta_rule = use_delta_rule

        # Q, K, V projections
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output projection
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

        # Learnable gate to combine memory and local attention
        self.gate = nn.Parameter(torch.zeros(num_heads))

        # Bank aggregation weights (learnable, per head per bank)
        if num_memory_banks > 1:
            self.bank_weights = nn.Parameter(torch.zeros(num_heads, num_memory_banks))
        else:
            self.register_buffer('bank_weights', None)

        # Memory banks
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

        self.memories = [
            torch.zeros(self.num_heads, self.head_dim, self.head_dim, device=device, dtype=dtype)
            for _ in range(self.num_memory_banks)
        ]
        self.memory_norms = [
            torch.zeros(self.num_heads, self.head_dim, device=device, dtype=dtype)
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
        """単一メモリバンクから取得"""
        sigma_q = elu_plus_one(q)

        if memory_norm.sum() < 1e-6:
            return torch.zeros_like(q)

        a_mem_unnorm = torch.einsum('bhsd,hde->bhse', sigma_q, memory)
        norm = torch.einsum('bhsd,hd->bhs', sigma_q, memory_norm)
        norm = norm.clamp(min=1e-6).unsqueeze(-1)

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

        stacked = torch.stack(bank_results, dim=0)
        weights = F.softmax(self.bank_weights, dim=-1)
        weights = weights.view(1, self.num_heads, 1, 1, self.num_memory_banks)
        stacked = stacked.permute(1, 2, 3, 4, 0)
        combined = (stacked * weights).sum(dim=-1)

        return combined

    def _update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """現在のメモリバンクを更新"""
        sigma_k = elu_plus_one(k)

        if self.memories is None or self.memory_norms is None:
            self.reset_memory(k.device)

        assert self.memories is not None and self.memory_norms is not None

        bank_idx = int(self.current_bank.item())
        memory = self.memories[bank_idx]
        memory_norm = self.memory_norms[bank_idx]

        if self.use_delta_rule:
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

        self.memories[bank_idx] = (memory + memory_update).detach()

        z_update = sigma_k.sum(dim=(0, 2))
        z_update = z_update / k.size(0)
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
        現在のセグメント内でCausal Linear Attentionを計算

        累積和を使用してO(n)で計算。各位置iは位置0~iのK,Vにのみアテンション可能。

        Args:
            q: [batch, heads, seq, head_dim]
            k: [batch, heads, seq, head_dim]
            v: [batch, heads, seq, head_dim]

        Returns:
            output: [batch, heads, seq, head_dim]
        """
        batch_size, num_heads, seq_len, head_dim = q.shape

        sigma_q = elu_plus_one(q)  # [B, H, S, D]
        sigma_k = elu_plus_one(k)  # [B, H, S, D]

        # 累積的にK^T @ Vを計算: [B, H, S, D, D]
        # kv_cumsum[i] = sum_{j=0}^{i} K_j^T @ V_j
        kv = torch.einsum('bhsd,bhse->bhsde', sigma_k, v)  # [B, H, S, D, D]
        kv_cumsum = torch.cumsum(kv, dim=2)  # [B, H, S, D, D]

        # 累積的にKを計算（正規化用）: [B, H, S, D]
        k_cumsum = torch.cumsum(sigma_k, dim=2)  # [B, H, S, D]

        # Q @ (cumsum K^T V) / (Q @ cumsum K)
        numerator = torch.einsum('bhsd,bhsde->bhse', sigma_q, kv_cumsum)  # [B, H, S, D]
        denominator = torch.einsum('bhsd,bhsd->bhs', sigma_q, k_cumsum)  # [B, H, S]
        denominator = denominator.clamp(min=1e-6).unsqueeze(-1)  # [B, H, S, 1]

        output = numerator / denominator  # [B, H, S, D]

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """Forward pass with Memory + Causal Linear Attention"""
        batch_size, seq_len, _ = hidden_states.shape

        q = self.w_q(hidden_states)
        k = self.w_k(hidden_states)
        v = self.w_v(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 1. 過去のメモリから取得
        memory_output = self._retrieve_from_memory(q)

        # 2. 現在のセグメント内でCausal Linear Attention
        local_output = self._causal_linear_attention(q, k, v)

        # 3. 学習可能なゲートで組み合わせ
        # gate: [num_heads] -> sigmoid -> [1, num_heads, 1, 1]
        gate = torch.sigmoid(self.gate).view(1, self.num_heads, 1, 1)
        output = gate * memory_output + (1 - gate) * local_output

        # 4. メモリ更新
        if update_memory:
            self._update_memory(k, v)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_size)
        output = self.w_o(output)

        return output

    def get_bank_weights(self) -> Optional[torch.Tensor]:
        """各バンクの重みを取得（マルチバンクのみ）"""
        if self.bank_weights is not None:
            return F.softmax(self.bank_weights, dim=-1)
        return None

    def memory_info(self) -> dict:
        """メモリ情報を取得"""
        single_bank_size = self.num_heads * self.head_dim * self.head_dim * 4
        single_norm_size = self.num_heads * self.head_dim * 4
        total_per_bank = single_bank_size + single_norm_size
        return {
            "num_banks": self.num_memory_banks,
            "bytes_per_bank": total_per_bank,
            "total_bytes": total_per_bank * self.num_memory_banks,
            "current_bank": self.current_bank.item() if self.current_bank is not None else 0,
            "segment_counter": self.segment_counter.item() if self.segment_counter is not None else 0,
        }


class InfiniAttentionALiBi(nn.Module):
    """
    ALiBi付きInfini-Attention Module with Causal Linear Attention

    線形化近似でALiBiを組み込む:
    - メモリ更新時にALiBi重み exp(-slope * segment_distance) を適用
    - 遠いセグメントほど重みが小さくなり、位置バイアスが反映される
    - 現在セグメント内はCausal Linear Attentionで計算

    数式:
      M_φ = Σ_i w_i * φ(K_i) * V_i^T   where w_i = exp(-slope * d_i)
      z_φ = Σ_i w_i * φ(K_i)
      Output = φ(Q) @ M_φ / (φ(Q) @ z_φ)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        use_delta_rule: bool = True,
        alibi_scale: float = 1.0,
    ):
        """
        Args:
            hidden_size: 隠れ層の次元
            num_heads: アテンションヘッド数
            use_delta_rule: Delta ruleを使用するか
            alibi_scale: ALiBiスロープのスケール係数（大きいほど減衰が強い）
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.use_delta_rule = use_delta_rule
        self.alibi_scale = alibi_scale

        # Q, K, V projections
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output projection
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

        # Learnable gate to combine memory and local attention
        self.gate = nn.Parameter(torch.zeros(num_heads))

        # ALiBi slopes (non-learnable, per head)
        alibi_slopes = get_alibi_slopes(num_heads) * alibi_scale
        self.register_buffer('alibi_slopes', alibi_slopes)

        # Memory (weighted by ALiBi)
        self.memory: Optional[torch.Tensor] = None
        self.memory_norm: Optional[torch.Tensor] = None

        # Segment counter for ALiBi distance
        self.register_buffer('segment_count', torch.tensor(0))

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        """メモリをリセット"""
        if device is None:
            device = self.w_q.weight.device
        dtype = self.w_q.weight.dtype

        self.memory = torch.zeros(
            self.num_heads, self.head_dim, self.head_dim,
            device=device, dtype=dtype
        )
        self.memory_norm = torch.zeros(
            self.num_heads, self.head_dim,
            device=device, dtype=dtype
        )
        self.segment_count = torch.tensor(0, device=device)

    def _compute_alibi_weight(self) -> torch.Tensor:
        """
        現在のセグメント距離に基づくALiBi重みを計算

        Returns:
            weight: [num_heads] 各ヘッドの重み
        """
        # w = exp(-slope * segment_count)
        # segment_countが大きいほど（過去のセグメントほど）重みが小さくなる
        # ただし、メモリ更新時は「現在のセグメント」の重みなので、
        # 取得時に相対距離を考慮する必要がある
        # ここでは簡略化: 更新時に累積的に減衰を適用
        return torch.exp(-self.alibi_slopes * self.segment_count.float())

    def _retrieve_from_memory(self, q: torch.Tensor) -> torch.Tensor:
        """メモリから情報を取得"""
        if self.memory is None or self.memory_norm is None:
            return torch.zeros_like(q)

        sigma_q = elu_plus_one(q)

        if self.memory_norm.sum() < 1e-6:
            return torch.zeros_like(q)

        # φ(Q) @ M_φ
        a_mem_unnorm = torch.einsum('bhsd,hde->bhse', sigma_q, self.memory)
        # φ(Q) @ z_φ
        norm = torch.einsum('bhsd,hd->bhs', sigma_q, self.memory_norm)
        norm = norm.clamp(min=1e-6).unsqueeze(-1)

        return a_mem_unnorm / norm

    def _update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """ALiBi重み付きでメモリを更新"""
        sigma_k = elu_plus_one(k)

        if self.memory is None or self.memory_norm is None:
            self.reset_memory(k.device)

        assert self.memory is not None and self.memory_norm is not None

        # ALiBi重みを計算 [num_heads]
        alibi_weight = self._compute_alibi_weight()
        # [1, num_heads, 1, 1] にreshapeしてブロードキャスト
        alibi_weight = alibi_weight.view(1, self.num_heads, 1, 1)

        # 重み付きσ(K)
        weighted_sigma_k = sigma_k * alibi_weight

        if self.use_delta_rule:
            # Delta Rule: まず現在のメモリから取得
            retrieved_unnorm = torch.einsum('bhsd,hde->bhse', weighted_sigma_k, self.memory)
            norm = torch.einsum('bhsd,hd->bhs', weighted_sigma_k, self.memory_norm)
            norm = norm.clamp(min=1e-6).unsqueeze(-1)
            retrieved = retrieved_unnorm / norm

            # Delta = V - retrieved
            delta_v = v - retrieved
            memory_update = torch.einsum('bhsd,bhse->hde', weighted_sigma_k, delta_v)
            memory_update = memory_update / (k.size(0) * k.size(2))
        else:
            # 単純な外積更新
            memory_update = torch.einsum('bhsd,bhse->hde', weighted_sigma_k, v)
            memory_update = memory_update / (k.size(0) * k.size(2))

        self.memory = (self.memory + memory_update).detach()

        # 正規化項の更新
        z_update = weighted_sigma_k.sum(dim=(0, 2))  # [num_heads, head_dim]
        z_update = z_update / k.size(0)
        self.memory_norm = (self.memory_norm + z_update).detach()

        # セグメントカウンタを更新
        self.segment_count = self.segment_count + 1

    def _causal_linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        現在のセグメント内でCausal Linear Attentionを計算

        累積和を使用してO(n)で計算。各位置iは位置0~iのK,Vにのみアテンション可能。

        Args:
            q: [batch, heads, seq, head_dim]
            k: [batch, heads, seq, head_dim]
            v: [batch, heads, seq, head_dim]

        Returns:
            output: [batch, heads, seq, head_dim]
        """
        sigma_q = elu_plus_one(q)
        sigma_k = elu_plus_one(k)

        # 累積的にK^T @ Vを計算
        kv = torch.einsum('bhsd,bhse->bhsde', sigma_k, v)
        kv_cumsum = torch.cumsum(kv, dim=2)

        # 累積的にKを計算（正規化用）
        k_cumsum = torch.cumsum(sigma_k, dim=2)

        # Q @ (cumsum K^T V) / (Q @ cumsum K)
        numerator = torch.einsum('bhsd,bhsde->bhse', sigma_q, kv_cumsum)
        denominator = torch.einsum('bhsd,bhsd->bhs', sigma_q, k_cumsum)
        denominator = denominator.clamp(min=1e-6).unsqueeze(-1)

        return numerator / denominator

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """Forward pass with Memory + Causal Linear Attention (ALiBi)"""
        batch_size, seq_len, _ = hidden_states.shape

        q = self.w_q(hidden_states)
        k = self.w_k(hidden_states)
        v = self.w_v(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 1. 過去のメモリから取得
        memory_output = self._retrieve_from_memory(q)

        # 2. 現在のセグメント内でCausal Linear Attention
        local_output = self._causal_linear_attention(q, k, v)

        # 3. 学習可能なゲートで組み合わせ
        gate = torch.sigmoid(self.gate).view(1, self.num_heads, 1, 1)
        output = gate * memory_output + (1 - gate) * local_output

        # 4. メモリ更新
        if update_memory:
            self._update_memory(k, v)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_size)
        output = self.w_o(output)

        return output

    def memory_info(self) -> dict:
        """メモリ情報を取得"""
        memory_size = self.num_heads * self.head_dim * self.head_dim * 4
        norm_size = self.num_heads * self.head_dim * 4
        return {
            "total_bytes": memory_size + norm_size,
            "segment_count": self.segment_count.item() if self.segment_count is not None else 0,
            "alibi_scale": self.alibi_scale,
        }


class InfiniAttentionLayer(nn.Module):
    """
    Infini-Attention Transformer Layer

    Pre-LayerNorm + Parallel Attention/MLP (Pythia style)

    use_alibi=Trueの場合、ALiBi付きのInfiniAttentionを使用。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_memory_banks: int = 1,
        segments_per_bank: int = 4,
        use_delta_rule: bool = True,
        use_alibi: bool = False,
        alibi_scale: float = 1.0,
    ):
        super().__init__()

        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.use_alibi = use_alibi

        self.attention: Union[InfiniAttention, InfiniAttentionALiBi]
        if use_alibi:
            # ALiBi版（Multi-Memory Bankは非対応）
            self.attention = InfiniAttentionALiBi(
                hidden_size=hidden_size,
                num_heads=num_heads,
                use_delta_rule=use_delta_rule,
                alibi_scale=alibi_scale,
            )
        else:
            # 通常版（Multi-Memory Bank対応）
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
