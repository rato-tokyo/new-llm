"""
Full Infini-Attention Model Implementation

全層でInfini-Attention (Memory-Only) を使用するモデル。
標準Attentionに比べてO(n²) → O(n×d²)の計算量削減。

アーキテクチャ:
  Token Embedding (512-dim)
         ↓
  Layer 0-5: InfiniAttentionLayer (Memory Only)
         ↓
  Output Head (512 → vocab)

特徴:
  - 全層で圧縮メモリを使用
  - 計算量: O(num_layers × seq_len × hidden_size²)
  - メモリ使用量: O(num_layers × num_heads × head_dim²) 固定
  - 長いシーケンスほど標準Attentionに対して有利
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.infini_attention import InfiniAttentionLayer


class FullInfiniModel(nn.Module):
    """
    Full Infini-Attention Model

    全層でInfini-Attention (Memory-Only) を使用。
    位置エンコーディングなし（NoPE）またはALiBi。
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        use_delta_rule: bool = True,
        num_memory_banks: int = 1,
        segments_per_bank: int = 4,
        use_alibi: bool = False,
        alibi_scale: float = 1.0,
    ):
        """
        Args:
            vocab_size: 語彙サイズ
            hidden_size: 隠れ層次元
            num_layers: Infini-Attentionレイヤー数
            num_heads: アテンションヘッド数
            intermediate_size: FFN中間層次元
            use_delta_rule: Delta Ruleを使用するか
            num_memory_banks: メモリバンク数（1=シングル、2以上=マルチ）
            segments_per_bank: 各バンクに蓄積するセグメント数
            use_alibi: ALiBi位置バイアスを使用するか
            alibi_scale: ALiBiスロープのスケール係数
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_memory_banks = num_memory_banks
        self.use_alibi = use_alibi

        # Embedding (位置エンコーディングなし)
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # All layers: Infini-Attention (Memory Only)
        self.layers = nn.ModuleList([
            InfiniAttentionLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                num_memory_banks=num_memory_banks,
                segments_per_bank=segments_per_bank,
                use_delta_rule=use_delta_rule,
                use_alibi=use_alibi,
                alibi_scale=alibi_scale,
            )
            for _ in range(num_layers)
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
        """全レイヤーのメモリをリセット"""
        device = self.embed_in.weight.device
        for layer in self.layers:
            layer.reset_memory(device)

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
            update_memory: メモリを更新するか

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        hidden_states = self.embed_in(input_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                update_memory=update_memory,
            )

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.embed_out(hidden_states)

        return logits

    def num_parameters(self) -> dict[str, int]:
        """Count parameters"""
        total = sum(p.numel() for p in self.parameters())
        embedding = self.embed_in.weight.numel()
        lm_head = self.embed_out.weight.numel()

        layer_params = sum(
            sum(p.numel() for p in layer.parameters())
            for layer in self.layers
        )

        return {
            "total": total,
            "embedding": embedding,
            "lm_head": lm_head,
            "infini_layers": layer_params,
            "transformer": layer_params,
        }

    def memory_info(self) -> dict[str, any]:
        """全レイヤーのメモリ情報を取得"""
        layer_info = self.layers[0].attention.memory_info()
        return {
            "num_layers": self.num_layers,
            "per_layer_bytes": layer_info.get("total_bytes", 0),
            "total_bytes": layer_info.get("total_bytes", 0) * self.num_layers,
            "use_alibi": self.use_alibi,
        }
