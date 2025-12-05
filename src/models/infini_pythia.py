"""
Infini-Pythia Model Implementation

1層目: Infini-Attention (NoPE - 位置エンコーディングなし)
2層目以降: MLA with ALiBi

アーキテクチャ:
  Token Embedding (512-dim)
         ↓
  Layer 0: InfiniAttentionLayer (NoPE, 圧縮メモリ)
         ↓
  Layer 1-5: MLALayer (ALiBi)
         ↓
  Output Head (512 → vocab)

Infini-Attentionのメモリは訓練中に蓄積され、
長距離依存関係の学習に貢献する。
"""

from typing import Optional, Dict

import torch
import torch.nn as nn

from src.models.infini_attention import InfiniAttentionLayer
from src.models.mla import MLALayer


class InfiniPythiaModel(nn.Module):
    """
    Infini-Pythia Model

    1層目: Infini-Attention (NoPE, compressive memory)
    2層目以降: MLA with ALiBi
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        kv_dim: int = 128,
        q_compressed: bool = False,
        alibi_slope: float = 0.0625,
        use_delta_rule: bool = True,
    ):
        """
        Args:
            vocab_size: 語彙サイズ
            hidden_size: 隠れ層次元
            num_layers: 総レイヤー数（1層目Infini + (num_layers-1)層MLA）
            num_heads: アテンションヘッド数
            intermediate_size: FFN中間層次元
            kv_dim: MLA KV圧縮次元
            q_compressed: MLAでQ圧縮を使用するか
            alibi_slope: ALiBiスロープ
            use_delta_rule: Infini-AttentionでDelta Ruleを使用するか
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.kv_dim = kv_dim

        # Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # Layer 0: Infini-Attention (NoPE)
        self.infini_layer = InfiniAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            use_delta_rule=use_delta_rule,
        )

        # Layers 1-(num_layers-1): MLA with ALiBi
        self.mla_layers = nn.ModuleList([
            MLALayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                kv_dim=kv_dim,
                q_compressed=q_compressed,
                alibi_slope=alibi_slope,
            )
            for _ in range(num_layers - 1)
        ])

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size)

        # LM Head
        self.embed_out = nn.Linear(hidden_size, vocab_size, bias=False)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def reset_memory(self) -> None:
        """Infini-Attentionのメモリをリセット"""
        self.infini_layer.reset_memory(self.embed_in.weight.device)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        update_memory: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            update_memory: Infini-Attentionのメモリを更新するか

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Embedding
        hidden_states = self.embed_in(input_ids)

        # Layer 0: Infini-Attention
        hidden_states = self.infini_layer(
            hidden_states,
            attention_mask,
            update_memory=update_memory,
        )

        # Layers 1+: MLA with ALiBi
        for layer in self.mla_layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        # LM Head
        logits = self.embed_out(hidden_states)

        return logits

    def num_parameters(self) -> Dict[str, int]:
        """Count parameters"""
        total = sum(p.numel() for p in self.parameters())
        embedding = self.embed_in.weight.numel()
        lm_head = self.embed_out.weight.numel()

        infini_params = sum(p.numel() for p in self.infini_layer.parameters())
        mla_params = sum(
            sum(p.numel() for p in layer.parameters())
            for layer in self.mla_layers
        )

        return {
            "total": total,
            "embedding": embedding,
            "lm_head": lm_head,
            "infini_layer": infini_params,
            "mla_layers": mla_params,
            "transformer": infini_params + mla_params,
        }

    def get_infini_gate_values(self) -> torch.Tensor:
        """Infini-Attentionのゲート値を取得"""
        return self.infini_layer.attention.get_gate_values()

    def kv_cache_info(self, seq_len: int, batch_size: int = 1) -> dict[str, float]:
        """KVキャッシュ情報を計算"""
        # Standard MHA: K + V per layer = 2 * hidden_size * num_layers
        standard_per_layer = batch_size * seq_len * self.hidden_size * 2 * 4  # float32
        standard_total = standard_per_layer * self.num_layers

        # Infini layer: メモリは固定サイズ (num_heads, head_dim, head_dim)
        head_dim = self.hidden_size // 8  # assuming 8 heads
        infini_memory = 8 * head_dim * head_dim * 4  # M matrix
        infini_memory += 8 * head_dim * 4  # z vector

        # MLA layers: c_kv only
        mla_per_layer = batch_size * seq_len * self.kv_dim * 4
        mla_total = mla_per_layer * (self.num_layers - 1)

        total = infini_memory + mla_total
        reduction = (standard_total - total) / standard_total * 100

        return {
            "standard_bytes": standard_total,
            "infini_memory_bytes": infini_memory,
            "mla_cache_bytes": mla_total,
            "total_bytes": total,
            "reduction_percent": reduction,
        }
