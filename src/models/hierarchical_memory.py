"""
Hierarchical Memory Attention with Learned Expansion

階層的メモリシステム：粗粒度メモリで検索し、必要に応じて細粒度メモリに展開。

特徴:
- 細粒度メモリを常に保持（ストレージ想定）
- 粗粒度メモリは細粒度の加算で動的生成
- 展開判断は学習可能なゲートで決定
- 推論後の出力エントロピーを入力として使用

アーキテクチャ:
  Storage: [M_0, M_1, M_2, M_3] (細粒度、常に保存)

  推論時:
    1. C = M_0 + M_1 + M_2 + M_3 (粗粒度を動的生成)
    2. output_coarse = φ(Q) @ C / (φ(Q) @ z_C)
    3. expansion_gate(output_coarse) → 展開するか判断
    4. 展開する場合: 各細粒度メモリから取得して混合
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.infini_attention import elu_plus_one


class HierarchicalMemoryAttention(nn.Module):
    """
    Hierarchical Memory with Learned Expansion Gate

    細粒度メモリを保持し、粗粒度は動的に生成。
    展開判断は学習可能なゲートで決定。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_fine_memories: int = 4,
        use_delta_rule: bool = True,
    ):
        """
        Args:
            hidden_size: 隠れ層の次元
            num_heads: アテンションヘッド数
            num_fine_memories: 細粒度メモリの数
            use_delta_rule: Delta ruleを使用するか
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_fine_memories = num_fine_memories
        self.use_delta_rule = use_delta_rule

        # Q, K, V projections
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output projection
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

        # Gate to combine memory and local attention
        self.memory_gate = nn.Parameter(torch.zeros(num_heads))

        # Expansion gate: 出力の特徴から展開するかを判断
        # 入力: head_dim次元の出力ベクトル
        # 出力: スカラー（展開確率）
        self.expansion_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
        )

        # Fine memories (常に保持)
        self.fine_memories: Optional[list[torch.Tensor]] = None
        self.fine_memory_norms: Optional[list[torch.Tensor]] = None

        # Current memory index for round-robin update
        self.register_buffer('current_memory_idx', torch.tensor(0))

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        """全メモリをリセット"""
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
        """粗粒度メモリを動的に生成（全細粒度メモリの加算）"""
        if self.fine_memories is None or self.fine_memory_norms is None:
            raise RuntimeError("Memory not initialized")

        M_coarse = torch.stack(self.fine_memories).sum(dim=0)
        z_coarse = torch.stack(self.fine_memory_norms).sum(dim=0)
        return M_coarse, z_coarse

    def _retrieve_from_memory(
        self,
        sigma_q: torch.Tensor,
        memory: torch.Tensor,
        memory_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        単一メモリから取得

        Returns:
            output: [batch, heads, seq, head_dim]
            relevance: [batch, heads, seq]
        """
        # メモリが空の場合
        if memory_norm.sum() < 1e-6:
            batch_size, num_heads, seq_len, head_dim = sigma_q.shape
            return (
                torch.zeros_like(sigma_q),
                torch.zeros(batch_size, num_heads, seq_len, device=sigma_q.device, dtype=sigma_q.dtype)
            )

        # φ(Q) @ M
        a_mem_unnorm = torch.einsum('bhsd,hde->bhse', sigma_q, memory)

        # φ(Q) @ z (relevance)
        relevance = torch.einsum('bhsd,hd->bhs', sigma_q, memory_norm)

        # 正規化
        norm = relevance.clamp(min=1e-6).unsqueeze(-1)
        output = a_mem_unnorm / norm

        return output, relevance

    def _retrieve_fine_grained(self, sigma_q: torch.Tensor) -> torch.Tensor:
        """細粒度メモリから取得（Attention-based混合）"""
        if self.fine_memories is None or self.fine_memory_norms is None:
            return torch.zeros_like(sigma_q)

        outputs = []
        relevances = []

        for memory, memory_norm in zip(self.fine_memories, self.fine_memory_norms):
            output, relevance = self._retrieve_from_memory(sigma_q, memory, memory_norm)
            outputs.append(output)
            relevances.append(relevance.mean(dim=-1))  # [batch, heads]

        # Softmax重み付け
        stacked_outputs = torch.stack(outputs, dim=0)
        stacked_relevances = torch.stack(relevances, dim=0)
        weights = F.softmax(stacked_relevances, dim=0)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        combined = (stacked_outputs * weights).sum(dim=0)
        return combined

    def _update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """現在の細粒度メモリを更新（ラウンドロビン）"""
        sigma_k = elu_plus_one(k)

        if self.fine_memories is None or self.fine_memory_norms is None:
            self.reset_memory(k.device)

        assert self.fine_memories is not None and self.fine_memory_norms is not None

        idx = int(self.current_memory_idx.item())
        memory = self.fine_memories[idx]
        memory_norm = self.fine_memory_norms[idx]

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

        self.fine_memories[idx] = (memory + memory_update).detach()

        z_update = sigma_k.sum(dim=(0, 2))
        z_update = z_update / k.size(0)
        self.fine_memory_norms[idx] = (memory_norm + z_update).detach()

        # 次のメモリへ
        self.current_memory_idx = torch.tensor(
            (idx + 1) % self.num_fine_memories,
            device=k.device
        )

    def _causal_linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """現在セグメント内のCausal Linear Attention"""
        sigma_q = elu_plus_one(q)
        sigma_k = elu_plus_one(k)

        kv = torch.einsum('bhsd,bhse->bhsde', sigma_k, v)
        kv_cumsum = torch.cumsum(kv, dim=2)

        k_cumsum = torch.cumsum(sigma_k, dim=2)

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
        """Forward pass with hierarchical memory retrieval"""
        batch_size, seq_len, _ = hidden_states.shape

        q = self.w_q(hidden_states)
        k = self.w_k(hidden_states)
        v = self.w_v(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        sigma_q = elu_plus_one(q)

        # メモリが初期化されていない場合
        if self.fine_memories is None:
            self.reset_memory(q.device)

        # Step 1: 粗粒度メモリから取得
        M_coarse, z_coarse = self._get_coarse_memory()
        output_coarse, _ = self._retrieve_from_memory(sigma_q, M_coarse, z_coarse)

        # Step 2: 展開判断（学習可能）
        # 出力を平均してゲートに入力
        coarse_repr = output_coarse.transpose(1, 2).contiguous()  # [batch, seq, heads, head_dim]
        coarse_repr = coarse_repr.view(batch_size, seq_len, self.hidden_size)
        expansion_logit = self.expansion_gate(coarse_repr)  # [batch, seq, 1]
        expansion_prob = torch.sigmoid(expansion_logit)  # [batch, seq, 1]

        # Step 3: 細粒度メモリから取得
        output_fine = self._retrieve_fine_grained(sigma_q)

        # Step 4: 展開確率で混合（Soft decision、学習可能）
        # expansion_prob: [batch, seq, 1] -> [batch, 1, seq, 1]
        expansion_prob = expansion_prob.transpose(1, 2).unsqueeze(-1)
        memory_output = expansion_prob * output_fine + (1 - expansion_prob) * output_coarse

        # Step 5: ローカルアテンションと組み合わせ
        local_output = self._causal_linear_attention(q, k, v)

        gate = torch.sigmoid(self.memory_gate).view(1, self.num_heads, 1, 1)
        output = gate * memory_output + (1 - gate) * local_output

        # Step 6: メモリ更新
        if update_memory:
            self._update_memory(k, v)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_size)
        output = self.w_o(output)

        return output

    def get_expansion_stats(self) -> dict:
        """展開統計を取得（デバッグ用）"""
        return {
            "num_fine_memories": self.num_fine_memories,
            "current_memory_idx": self.current_memory_idx.item() if self.current_memory_idx is not None else 0,
        }

    def memory_info(self) -> dict:
        """メモリ情報を取得"""
        single_memory_size = self.num_heads * self.head_dim * self.head_dim * 4
        single_norm_size = self.num_heads * self.head_dim * 4
        per_memory = single_memory_size + single_norm_size

        return {
            "num_fine_memories": self.num_fine_memories,
            "bytes_per_memory": per_memory,
            "total_bytes": per_memory * self.num_fine_memories,
        }

    # メモリ保存/復元API
    def get_memory_state(self) -> dict:
        """メモリ状態を取得（保存用）"""
        if self.fine_memories is None or self.fine_memory_norms is None:
            return {"memories": [], "norms": [], "current_idx": 0}

        return {
            "memories": [m.clone() for m in self.fine_memories],
            "norms": [z.clone() for z in self.fine_memory_norms],
            "current_idx": self.current_memory_idx.item(),
        }

    def set_memory_state(self, state: dict) -> None:
        """メモリ状態を復元"""
        device = self.w_q.weight.device
        self.fine_memories = [m.to(device) for m in state["memories"]]
        self.fine_memory_norms = [z.to(device) for z in state["norms"]]
        self.current_memory_idx = torch.tensor(state["current_idx"], device=device)


class HierarchicalMemoryAttentionLayer(nn.Module):
    """
    Hierarchical Memory Attention Transformer Layer
    """

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

        self.attention = HierarchicalMemoryAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_fine_memories=num_fine_memories,
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
