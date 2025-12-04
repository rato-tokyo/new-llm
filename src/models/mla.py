"""
MLA (Multi-head Latent Attention) with ALiBi

DeepSeek-V2のMLA方式をALiBiで実装。
KVを共通の低次元潜在ベクトルに圧縮し、吸収モードで計算。

アーキテクチャ:
  X (hidden_size) → W_DKV → c_kv (kv_dim)  # KV共通圧縮
  X (hidden_size) → W_DQ → c_q (q_dim)     # Q圧縮（オプション）

  吸収モード:
    score = c_q @ W_absorbed @ c_kv^T + alibi_bias
    W_absorbed = W_UQ^T @ W_UK (事前計算可能)

  output = softmax(score) @ c_kv @ W_UV

KVキャッシュ:
  - c_kv のみをキャッシュ（k_peは不要、ALiBiのため）
  - 例: hidden=512, kv_dim=128 → 75%削減
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.alibi import ALiBiCache


class MLAAttention(nn.Module):
    """
    MLA Attention with ALiBi

    KV共通圧縮 + 吸収モード + ALiBi位置エンコーディング
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        kv_dim: int = 128,
        q_compressed: bool = False,
        q_dim: Optional[int] = None,
        alibi_slope: float = 0.0625,
    ):
        """
        Args:
            hidden_size: 入力次元
            num_heads: ヘッド数
            kv_dim: KV圧縮後の次元
            q_compressed: Qも圧縮するか
            q_dim: Q圧縮後の次元（q_compressed=Trueの場合）
            alibi_slope: ALiBiスロープ（統一）
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.kv_dim = kv_dim
        self.head_dim = hidden_size // num_heads
        self.kv_head_dim = kv_dim // num_heads
        self.q_compressed = q_compressed
        self.alibi_slope = alibi_slope

        # KV共通圧縮: hidden_size → kv_dim
        self.w_dkv = nn.Linear(hidden_size, kv_dim, bias=False)

        # Q処理
        if q_compressed:
            self.q_dim = q_dim or kv_dim
            # Q圧縮: hidden_size → q_dim
            self.w_dq = nn.Linear(hidden_size, self.q_dim, bias=False)
            # 吸収行列: (q_dim, kv_dim) - 学習可能
            # W_absorbed = W_UQ^T @ W_UK を直接学習
            self.w_absorbed = nn.Linear(self.q_dim, kv_dim, bias=False)
        else:
            self.q_dim = hidden_size
            # Q射影: hidden_size → hidden_size（通常のMHA）
            self.w_q = nn.Linear(hidden_size, hidden_size, bias=False)
            # K復元: kv_dim → hidden_size
            self.w_uk = nn.Linear(kv_dim, hidden_size, bias=False)

        # V復元: kv_dim → hidden_size
        self.w_uv = nn.Linear(kv_dim, hidden_size, bias=False)

        # 出力射影
        self.w_o = nn.Linear(hidden_size, hidden_size, bias=False)

        # ALiBiキャッシュ
        self.alibi_cache = ALiBiCache(slope=alibi_slope, causal=True)

        # スケーリング
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: オプション（ALiBiに統合されるため通常不要）

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # KV共通圧縮: [batch, seq, hidden] → [batch, seq, kv_dim]
        c_kv = self.w_dkv(hidden_states)

        # ALiBiバイアス取得
        alibi_bias = self.alibi_cache.get(seq_len, hidden_states.device)

        if self.q_compressed:
            # Q圧縮モード（フルMLA）
            # c_q: [batch, seq, q_dim]
            c_q = self.w_dq(hidden_states)

            # 吸収モードでスコア計算
            # score = c_q @ W_absorbed @ c_kv^T
            # [batch, seq, q_dim] @ [q_dim, kv_dim] → [batch, seq, kv_dim]
            q_absorbed = self.w_absorbed(c_q)
            # [batch, seq, kv_dim] @ [batch, kv_dim, seq] → [batch, seq, seq]
            attn_weights = torch.bmm(q_absorbed, c_kv.transpose(1, 2))
        else:
            # Q非圧縮モード（KV圧縮のみ）
            # Q: [batch, seq, hidden]
            q = self.w_q(hidden_states)
            # K復元: [batch, seq, kv_dim] → [batch, seq, hidden]
            k = self.w_uk(c_kv)

            # Reshape to heads
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.transpose(1, 2)

            # score = Q @ K^T
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            # [batch, heads, seq, seq] → sum over heads for ALiBi
            # ALiBiは全ヘッド共通なので、そのまま加算
            attn_weights = attn_weights + alibi_bias.unsqueeze(0).unsqueeze(0)

            # Softmax
            attn_weights = F.softmax(attn_weights, dim=-1)

            # V復元 & Attention出力
            v = self.w_uv(c_kv)  # [batch, seq, hidden]
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.transpose(1, 2)  # [batch, heads, seq, head_dim]

            # output = attn @ V
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

            return self.w_o(attn_output)

        # Q圧縮モードの続き（吸収モード）
        # スケーリング & ALiBi
        attn_weights = attn_weights * self.scale + alibi_bias.unsqueeze(0)

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)

        # V処理: (attn @ c_kv) @ W_UV
        # [batch, seq, seq] @ [batch, seq, kv_dim] → [batch, seq, kv_dim]
        context = torch.bmm(attn_weights, c_kv)
        # [batch, seq, kv_dim] → [batch, seq, hidden]
        output = self.w_uv(context)

        return self.w_o(output)

    def num_parameters(self) -> Dict[str, int]:
        """パラメータ数を計算"""
        total = sum(p.numel() for p in self.parameters())

        compression_params = self.w_dkv.weight.numel()
        if self.q_compressed:
            compression_params += self.w_dq.weight.numel()
            compression_params += self.w_absorbed.weight.numel()
        else:
            compression_params += self.w_q.weight.numel()
            compression_params += self.w_uk.weight.numel()

        return {
            "total": total,
            "compression": compression_params,
            "output": self.w_o.weight.numel() + self.w_uv.weight.numel(),
        }

    def kv_cache_size(self, seq_len: int, batch_size: int = 1) -> Dict[str, Union[int, float]]:
        """KVキャッシュサイズを計算"""
        # 標準MHA: K + V = 2 * hidden_size
        standard = batch_size * seq_len * self.hidden_size * 2 * 4  # float32

        # MLA: c_kv のみ
        mla = batch_size * seq_len * self.kv_dim * 4

        reduction = (standard - mla) / standard * 100

        return {
            "standard_bytes": standard,
            "mla_bytes": mla,
            "reduction_percent": reduction,
        }


class MLALayer(nn.Module):
    """
    MLA Transformer Layer

    Pre-LayerNorm + Parallel Attention/MLP (Pythia style)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        kv_dim: int = 128,
        q_compressed: bool = False,
        alibi_slope: float = 0.0625,
    ):
        super().__init__()

        self.input_layernorm = nn.LayerNorm(hidden_size)

        self.attention = MLAAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            kv_dim=kv_dim,
            q_compressed=q_compressed,
            alibi_slope=alibi_slope,
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
    ) -> torch.Tensor:
        # Pre-LayerNorm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Parallel Attention + MLP
        attn_output = self.attention(hidden_states, attention_mask)
        mlp_output = self.mlp(hidden_states)

        # Residual
        hidden_states = residual + attn_output + mlp_output

        return hidden_states
