"""
Multi-Memory Layer (Multiple Independent Memories)

HSA (Hierarchical Sparse Attention) 方式を採用:
- 論文: arXiv:2511.23319v1

本プロジェクトへの適用:
- ChunkEncoder: 双方向エンコーダで[CLS]トークンからLandmarkを生成
- Q_slc: 検索専用のQuery射影 (Attention用Q_attnとは別)
- 検索スコア = Q_slc @ Landmark / sqrt(d)

論文との対応:
| HSA論文              | 本プロジェクト           |
|---------------------|------------------------|
| Chunk + [CLS]       | メモリ内キー列 + [CLS]   |
| Bi-directional Enc  | ChunkEncoder (2層)     |
| Q_slc               | w_q_slc (検索専用)      |
| Q_attn              | w_q (Attention用)      |
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base_components import PythiaMLP
from src.models.memory_utils import causal_linear_attention, elu_plus_one
from .base import BaseLayer


class ChunkEncoder(nn.Module):
    """双方向エンコーダでChunk（メモリ内キー列）を要約

    HSA論文 Section 3.1:
    - 各チャンクに[CLS]トークンを付加
    - 双方向Transformerエンコーダで処理
    - [CLS]の出力をLandmarkとして使用

    本プロジェクトでは、メモリに格納されたキー列を要約してLandmarkを生成。
    """

    def __init__(
        self,
        head_dim: int,
        num_encoder_heads: int = 4,
        num_encoder_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            head_dim: 各ヘッドの次元（Landmarkの次元）
            num_encoder_heads: エンコーダのヘッド数
            num_encoder_layers: エンコーダの層数
            dropout: ドロップアウト率
        """
        super().__init__()
        self.head_dim = head_dim

        # [CLS]トークン（学習可能）
        self.cls_token = nn.Parameter(torch.randn(1, 1, head_dim) * 0.02)

        # 双方向Transformerエンコーダ
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=head_dim,
            nhead=num_encoder_heads,
            dim_feedforward=head_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 出力射影（Landmark用）
        self.output_proj = nn.Linear(head_dim, head_dim, bias=False)

    def forward(self, key_sequence: torch.Tensor) -> torch.Tensor:
        """キー列からLandmarkを生成

        Args:
            key_sequence: (seq_len, head_dim) - メモリ内のキー列

        Returns:
            landmark: (head_dim,) - [CLS]トークンの出力
        """
        seq_len, _ = key_sequence.shape

        # [CLS] + キー列
        # (1, seq_len+1, head_dim)
        cls = self.cls_token  # (1, 1, head_dim)
        sequence = torch.cat([cls, key_sequence.unsqueeze(0)], dim=1)

        # 双方向エンコーダ
        encoded = self.encoder(sequence)  # (1, seq_len+1, head_dim)

        # [CLS]の出力をLandmarkとして使用
        cls_output = encoded[0, 0, :]  # (head_dim,)

        return self.output_proj(cls_output)


class MultiMemoryAttention(nn.Module):
    """Multiple Independent Memories with HSA-style Landmark Selection

    HSA (Hierarchical Sparse Attention) 方式 (arXiv:2511.23319v1):
    - ChunkEncoder: 双方向エンコーダで[CLS]トークンからLandmarkを生成
    - Q_slc: 検索専用Query（Attention用Qとは別の射影）
    - 検索スコア = Q_slc @ Landmark / sqrt(d)
    - 選択されたメモリからLinear Attentionで検索
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_memories: int = 4,
        use_delta_rule: bool = True,
        max_keys_per_memory: int = 256,  # メモリに保持するキーの最大数
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_memories = num_memories
        self.use_delta_rule = use_delta_rule
        self.max_keys_per_memory = max_keys_per_memory

        # Attention用 Q/K/V 射影
        self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

        # HSA: 検索専用 Q_slc 射影（論文Section 2.2）
        # Q_slc は Attention用 Q とは別の射影
        self.w_q_slc = nn.Linear(hidden_size, hidden_size, bias=False)

        # HSA: 各ヘッド用のChunkEncoder（双方向エンコーダ）
        # 論文ではチャンクごとにエンコーダを適用
        self.chunk_encoders = nn.ModuleList([
            ChunkEncoder(
                head_dim=self.head_dim,
                num_encoder_heads=max(1, self.head_dim // 16),  # head_dim/16 ヘッド
                num_encoder_layers=2,
            )
            for _ in range(num_heads)
        ])

        self.gate = nn.Parameter(torch.zeros(num_heads))

        # メモリ状態
        self.memories: Optional[list[torch.Tensor]] = None
        self.memory_norms: Optional[list[torch.Tensor]] = None
        # HSA: キー列を保持（ChunkEncoderの入力）
        # key_sequences[memory_idx][head_idx] = (seq_len, head_dim)
        self.key_sequences: Optional[list[list[torch.Tensor]]] = None
        self.register_buffer('current_memory_idx', torch.tensor(0))

        # スケーリング係数（論文: s_t,i = Q_slc @ K_slc / sqrt(d)）
        self.scale = self.head_dim ** -0.5

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
        # HSA: キー列を保持（ChunkEncoderの入力）
        # 各メモリ・各ヘッドごとに空のテンソルで初期化
        self.key_sequences = [
            [torch.zeros(0, self.head_dim, device=device, dtype=dtype) for _ in range(self.num_heads)]
            for _ in range(self.num_memories)
        ]
        self.current_memory_idx = torch.tensor(0, device=device)

    def _compute_landmark(self, memory_idx: int, head_idx: int) -> torch.Tensor:
        """ChunkEncoderでキー列からLandmarkを計算

        Args:
            memory_idx: メモリインデックス
            head_idx: ヘッドインデックス

        Returns:
            landmark: (head_dim,) - [CLS]トークンの出力
        """
        assert self.key_sequences is not None

        key_seq = self.key_sequences[memory_idx][head_idx]

        # キー列が空の場合はゼロベクトル
        if key_seq.size(0) == 0:
            return torch.zeros(self.head_dim, device=key_seq.device, dtype=key_seq.dtype)

        # ChunkEncoderでLandmarkを計算
        return self.chunk_encoders[head_idx](key_seq)

    def _compute_relevance(
        self, q_slc: torch.Tensor, memory_idx: int
    ) -> torch.Tensor:
        """検索用Queryとメモリの関連度を計算（HSA方式）

        論文Section 2.2: s_t,i = (Q_slc^T @ K_slc_i) / sqrt(d)

        Args:
            q_slc: (batch, num_heads, seq_len, head_dim) - 検索専用Query
            memory_idx: メモリインデックス

        Returns:
            relevance: (batch, num_heads) - シーケンス平均のスコア
        """
        batch_size, num_heads, seq_len, head_dim = q_slc.shape

        # 各ヘッドのLandmarkを計算
        landmarks = []
        for h in range(num_heads):
            landmark = self._compute_landmark(memory_idx, h)
            landmarks.append(landmark)

        # (num_heads, head_dim)
        landmarks_tensor = torch.stack(landmarks, dim=0)

        # HSA: Q_slc @ Landmark / sqrt(d)
        rel = torch.einsum('bhsd,hd->bhs', q_slc, landmarks_tensor) * self.scale
        return rel.mean(dim=-1)

    def _is_memory_empty(self, memory_idx: int) -> bool:
        """メモリが空かどうかをチェック"""
        if self.key_sequences is None:
            return True
        # 全ヘッドのキー列の合計長をチェック
        total_keys = sum(
            self.key_sequences[memory_idx][h].size(0)
            for h in range(self.num_heads)
        )
        return total_keys == 0

    def _retrieve_from_memory(
        self, q: torch.Tensor, q_slc: torch.Tensor
    ) -> torch.Tensor:
        """全メモリから検索し、Landmark関連度で加重統合

        Args:
            q: (batch, num_heads, seq_len, head_dim) - Attention用Query
            q_slc: (batch, num_heads, seq_len, head_dim) - 検索専用Query

        Returns:
            output: (batch, num_heads, seq_len, head_dim)
        """
        if self.memories is None or self.key_sequences is None:
            return torch.zeros_like(q)

        sigma_q = elu_plus_one(q)
        outputs, relevances = [], []
        has_non_empty = False

        for memory_idx, (memory, memory_norm) in enumerate(
            zip(self.memories, self.memory_norms)  # type: ignore
        ):
            # メモリが空の場合
            if self._is_memory_empty(memory_idx):
                outputs.append(torch.zeros_like(q))
                relevances.append(
                    torch.full((q.size(0), self.num_heads), float('-inf'), device=q.device)
                )
            else:
                has_non_empty = True
                # Linear Attentionでメモリから検索（σ(Q) を使用）
                a_mem = torch.einsum('bhsd,hde->bhse', sigma_q, memory)
                norm = torch.einsum('bhsd,hd->bhs', sigma_q, memory_norm)
                outputs.append(a_mem / norm.clamp(min=1e-6).unsqueeze(-1))

                # HSA: Q_slc @ Landmark で関連度計算（ChunkEncoderを使用）
                relevances.append(self._compute_relevance(q_slc, memory_idx))

        # 全メモリが空の場合はゼロを返す
        if not has_non_empty:
            return torch.zeros_like(q)

        stacked = torch.stack(outputs, dim=0)  # (num_memories, batch, heads, seq, dim)
        weights = F.softmax(torch.stack(relevances, dim=0), dim=0)  # (num_memories, batch, heads)
        return (stacked * weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=0)

    def _update_memory(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """メモリとキー列を更新"""
        sigma_k = elu_plus_one(k)

        if self.memories is None:
            self.reset_memory(k.device)

        assert self.memories is not None and self.memory_norms is not None
        assert self.key_sequences is not None

        idx = int(self.current_memory_idx.item())
        memory = self.memories[idx]
        memory_norm = self.memory_norms[idx]

        batch_size, num_heads, seq_len, head_dim = k.shape

        # メモリ更新（Linear Attention形式）
        if self.use_delta_rule:
            retrieved = torch.einsum('bhsd,hde->bhse', sigma_k, memory)
            norm = torch.einsum('bhsd,hd->bhs', sigma_k, memory_norm).clamp(min=1e-6).unsqueeze(-1)
            delta_v = v - retrieved / norm
            update = torch.einsum('bhsd,bhse->hde', sigma_k, delta_v) / (batch_size * seq_len)
        else:
            update = torch.einsum('bhsd,bhse->hde', sigma_k, v) / (batch_size * seq_len)

        self.memories[idx] = (memory + update).detach()
        self.memory_norms[idx] = (memory_norm + sigma_k.sum(dim=(0, 2)) / batch_size).detach()

        # HSA: キー列を更新（ChunkEncoderの入力）
        # k: (batch, num_heads, seq_len, head_dim)
        # 各ヘッドごとにキーを追加
        for h in range(num_heads):
            # (batch, seq_len, head_dim) -> (batch * seq_len, head_dim)
            new_keys = k[:, h, :, :].reshape(-1, head_dim).detach()

            # 既存のキー列に追加
            current_keys = self.key_sequences[idx][h]
            combined = torch.cat([current_keys, new_keys], dim=0)

            # max_keys_per_memory を超えたら古いキーを削除
            if combined.size(0) > self.max_keys_per_memory:
                combined = combined[-self.max_keys_per_memory:]

            self.key_sequences[idx][h] = combined

        self.current_memory_idx = torch.tensor((idx + 1) % self.num_memories, device=k.device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Attention用 Q/K/V
        q = self.w_q(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.w_k(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # HSA: 検索専用 Q_slc（Attention用Qとは別の射影）
        q_slc = self.w_q_slc(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # メモリ検索（Q_slc でメモリ選択、Q で検索）
        memory_output = self._retrieve_from_memory(q, q_slc)
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
            "key_sequences": [
                [ks.cpu().clone() for ks in mem_keys]
                for mem_keys in self.key_sequences
            ] if self.key_sequences else None,
            "current_memory_idx": self.current_memory_idx.cpu().clone(),
        }

    def set_memory_state(self, state: dict, device: Optional[torch.device] = None) -> None:
        if device is None:
            device = self.w_q.weight.device
        if state["memories"]:
            self.memories = [m.to(device) for m in state["memories"]]
        if state["memory_norms"]:
            self.memory_norms = [n.to(device) for n in state["memory_norms"]]
        if state.get("key_sequences"):
            self.key_sequences = [
                [ks.to(device) for ks in mem_keys]
                for mem_keys in state["key_sequences"]
            ]
        self.current_memory_idx = state["current_memory_idx"].to(device)


class MultiMemoryLayer(BaseLayer):
    """Multi-Memory Infini-Attention Layer with HSA-style Landmarks"""

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
