"""
Multi-Memory Attention with Attention-based Selection

複数の独立した圧縮メモリを持ち、Attention-based方式で動的に選択・混合。

特徴:
- 各メモリは独立して更新される（1つのメモリに1つのセグメントが蓄積）
- クエリとメモリのz（正規化項）との内積で関連度を計算
- 関連度に基づいてSoftmax重み付けで混合
- 追加パラメータなし（学習が安定）

数式:
  各メモリi: (M_i, z_i)

  関連度計算:
    relevance_i = sum(phi(Q) @ z_i)  # スカラー

  重み計算:
    weight_i = softmax(relevances)

  出力混合:
    output = sum_i(weight_i * (phi(Q) @ M_i / (phi(Q) @ z_i)))
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.infini_attention import elu_plus_one


class MultiMemoryInfiniAttention(nn.Module):
    """
    Multiple Independent Memories with Attention-based Selection

    複数の独立したメモリを持ち、クエリとの関連度に基づいて動的に選択・混合。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_memories: int = 4,
        use_delta_rule: bool = True,
    ):
        """
        Args:
            hidden_size: 隠れ層の次元
            num_heads: アテンションヘッド数
            num_memories: メモリの数
            use_delta_rule: Delta ruleを使用するか
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_memories = num_memories
        self.use_delta_rule = use_delta_rule

        # Q, K, V projections
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)

        # Output projection
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

        # Learnable gate to combine memory and local attention
        self.gate = nn.Parameter(torch.zeros(num_heads))

        # Multiple independent memories
        self.memories: Optional[list[torch.Tensor]] = None
        self.memory_norms: Optional[list[torch.Tensor]] = None

        # Current memory index (round-robin for update)
        self.register_buffer('current_memory_idx', torch.tensor(0))

    def reset_memory(self, device: Optional[torch.device] = None) -> None:
        """全メモリをリセット"""
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

    def _retrieve_from_single_memory(
        self,
        sigma_q: torch.Tensor,
        memory: torch.Tensor,
        memory_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        単一メモリから取得し、関連度スコアも返す

        Args:
            sigma_q: phi(Q) [batch, heads, seq, head_dim]
            memory: M [heads, head_dim, head_dim]
            memory_norm: z [heads, head_dim]

        Returns:
            output: [batch, heads, seq, head_dim]
            relevance: [batch, heads, seq] 関連度スコア
        """
        # メモリが空の場合
        if memory_norm.sum() < 1e-6:
            batch_size, num_heads, seq_len, head_dim = sigma_q.shape
            return (
                torch.zeros_like(sigma_q),
                torch.zeros(batch_size, num_heads, seq_len, device=sigma_q.device, dtype=sigma_q.dtype)
            )

        # phi(Q) @ M
        a_mem_unnorm = torch.einsum('bhsd,hde->bhse', sigma_q, memory)

        # phi(Q) @ z (関連度スコア)
        relevance = torch.einsum('bhsd,hd->bhs', sigma_q, memory_norm)

        # 正規化
        norm = relevance.clamp(min=1e-6).unsqueeze(-1)
        output = a_mem_unnorm / norm

        return output, relevance

    def _retrieve_from_memory(self, q: torch.Tensor) -> torch.Tensor:
        """
        全メモリからAttention-based選択で取得

        処理:
        1. 各メモリから出力と関連度を計算
        2. 関連度でSoftmax重み付け
        3. 重み付き平均で最終出力
        """
        if self.memories is None or self.memory_norms is None:
            return torch.zeros_like(q)

        sigma_q = elu_plus_one(q)

        # 各メモリから取得
        outputs = []
        relevances = []

        for memory, memory_norm in zip(self.memories, self.memory_norms):
            output, relevance = self._retrieve_from_single_memory(
                sigma_q, memory, memory_norm
            )
            outputs.append(output)
            # シーケンス全体での関連度（平均）
            relevances.append(relevance.mean(dim=-1))  # [batch, heads]

        # スタック: [num_memories, batch, heads, seq, head_dim]
        stacked_outputs = torch.stack(outputs, dim=0)
        # スタック: [num_memories, batch, heads]
        stacked_relevances = torch.stack(relevances, dim=0)

        # Softmaxで重み計算: [num_memories, batch, heads]
        weights = F.softmax(stacked_relevances, dim=0)

        # 重み付き平均
        # weights: [num_memories, batch, heads, 1, 1]
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        # combined: [batch, heads, seq, head_dim]
        combined = (stacked_outputs * weights).sum(dim=0)

        return combined

    def _update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """現在のメモリを更新（ラウンドロビン）"""
        sigma_k = elu_plus_one(k)

        if self.memories is None or self.memory_norms is None:
            self.reset_memory(k.device)

        assert self.memories is not None and self.memory_norms is not None

        # 現在のメモリインデックス
        idx = int(self.current_memory_idx.item())
        memory = self.memories[idx]
        memory_norm = self.memory_norms[idx]

        if self.use_delta_rule:
            # Delta Rule
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

        self.memories[idx] = (memory + memory_update).detach()

        z_update = sigma_k.sum(dim=(0, 2))
        z_update = z_update / k.size(0)
        self.memory_norms[idx] = (memory_norm + z_update).detach()

        # 次のメモリへ（ラウンドロビン）
        self.current_memory_idx = torch.tensor(
            (idx + 1) % self.num_memories,
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
        """Forward pass"""
        batch_size, seq_len, _ = hidden_states.shape

        q = self.w_q(hidden_states)
        k = self.w_k(hidden_states)
        v = self.w_v(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 1. 過去のメモリから取得（Attention-based選択）
        memory_output = self._retrieve_from_memory(q)

        # 2. 現在セグメント内のCausal Linear Attention
        local_output = self._causal_linear_attention(q, k, v)

        # 3. ゲートで組み合わせ
        gate = torch.sigmoid(self.gate).view(1, self.num_heads, 1, 1)
        output = gate * memory_output + (1 - gate) * local_output

        # 4. メモリ更新
        if update_memory:
            self._update_memory(k, v)

        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_size)
        output = self.w_o(output)

        return output

    def get_memory_weights(self, q: torch.Tensor) -> torch.Tensor:
        """
        デバッグ用: 各メモリの重みを取得

        Returns:
            weights: [num_memories, batch, heads]
        """
        if self.memories is None or self.memory_norms is None:
            return torch.zeros(self.num_memories, 1, self.num_heads, device=q.device)

        sigma_q = elu_plus_one(q)

        relevances = []
        for memory_norm in self.memory_norms:
            if memory_norm.sum() < 1e-6:
                relevances.append(torch.zeros(q.size(0), self.num_heads, device=q.device))
            else:
                relevance = torch.einsum('bhsd,hd->bhs', sigma_q, memory_norm)
                relevances.append(relevance.mean(dim=-1))

        stacked = torch.stack(relevances, dim=0)
        weights = F.softmax(stacked, dim=0)

        return weights

    def memory_info(self) -> dict:
        """メモリ情報を取得"""
        single_memory_size = self.num_heads * self.head_dim * self.head_dim * 4
        single_norm_size = self.num_heads * self.head_dim * 4
        per_memory = single_memory_size + single_norm_size

        return {
            "num_memories": self.num_memories,
            "bytes_per_memory": per_memory,
            "total_bytes": per_memory * self.num_memories,
            "current_memory_idx": self.current_memory_idx.item() if self.current_memory_idx is not None else 0,
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
            "current_memory_idx": self.current_memory_idx.cpu().clone(),
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
        self.current_memory_idx = state["current_memory_idx"].to(device)


class MultiMemoryInfiniAttentionLayer(nn.Module):
    """
    Multi-Memory Infini-Attention Transformer Layer
    """

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

        self.attention = MultiMemoryInfiniAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_memories=num_memories,
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
