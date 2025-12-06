"""
Hierarchical Memory Pythia Model

1層目: Hierarchical Memory Attention (学習可能な展開判断)
2層目以降: 標準Pythia (RoPE)

アーキテクチャ:
  Token Embedding (512-dim)
         ↓
  Layer 0: HierarchicalMemoryAttentionLayer
    ├─ Fine memories: [M_0, M_1, ..., M_n] (常に保持)
    ├─ Coarse memory: sum(fine_memories) (動的生成)
    └─ Expansion gate: 出力から展開を判断（学習可能）
         ↓
  Layer 1-5: PythiaLayer (RoPE)
         ↓
  Output Head (512 → vocab)
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.hierarchical_memory import HierarchicalMemoryAttentionLayer
from src.models.pythia import PythiaLayer


class HierarchicalMemoryPythiaModel(nn.Module):
    """
    Hierarchical Memory Pythia Model

    1層目: Hierarchical Memory Attention (学習可能な展開)
    2層目以降: 標準Pythia (RoPE)
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 2048,
        rotary_pct: float = 0.25,
        use_delta_rule: bool = True,
        num_fine_memories: int = 4,
    ):
        """
        Args:
            vocab_size: 語彙サイズ
            hidden_size: 隠れ層次元
            num_layers: 総レイヤー数
            num_heads: アテンションヘッド数
            intermediate_size: FFN中間層次元
            max_position_embeddings: 最大位置エンベディング
            rotary_pct: RoPEを適用する次元の割合
            use_delta_rule: Delta Ruleを使用するか
            num_fine_memories: 細粒度メモリの数
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_fine_memories = num_fine_memories

        # Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # Layer 0: Hierarchical Memory Attention
        self.hierarchical_layer = HierarchicalMemoryAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_fine_memories=num_fine_memories,
            use_delta_rule=use_delta_rule,
        )

        # Layers 1-(num_layers-1): Standard Pythia (RoPE)
        self.pythia_layers = nn.ModuleList([
            PythiaLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                rotary_pct=rotary_pct,
                max_position_embeddings=max_position_embeddings,
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
        """メモリをリセット"""
        self.hierarchical_layer.reset_memory(self.embed_in.weight.device)

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

        hidden_states = self.hierarchical_layer(
            hidden_states,
            attention_mask,
            update_memory=update_memory,
        )

        for layer in self.pythia_layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.embed_out(hidden_states)

        return logits

    def num_parameters(self) -> dict[str, int]:
        """Count parameters"""
        total = sum(p.numel() for p in self.parameters())
        embedding = self.embed_in.weight.numel()
        lm_head = self.embed_out.weight.numel()

        hierarchical_params = sum(p.numel() for p in self.hierarchical_layer.parameters())
        pythia_params = sum(
            sum(p.numel() for p in layer.parameters())
            for layer in self.pythia_layers
        )

        # Expansion gate parameters
        expansion_gate_params = sum(
            p.numel() for p in self.hierarchical_layer.attention.expansion_gate.parameters()
        )

        return {
            "total": total,
            "embedding": embedding,
            "lm_head": lm_head,
            "hierarchical_layer": hierarchical_params,
            "expansion_gate": expansion_gate_params,
            "pythia_layers": pythia_params,
            "transformer": hierarchical_params + pythia_params,
        }

    def memory_info(self) -> dict:
        """メモリ情報を取得"""
        return self.hierarchical_layer.attention.memory_info()

    def get_memory_state(self) -> dict:
        """
        メモリ状態を取得（別PCへの転送用）

        Returns:
            dict: メモリ状態（CPU上のテンソル）

        Example:
            # PC Aでメモリを取得
            state = model.get_memory_state()
            torch.save(state, "memory.pt")

            # PC Bでメモリを設定
            state = torch.load("memory.pt")
            model.set_memory_state(state)
        """
        return self.hierarchical_layer.get_memory_state()

    def set_memory_state(self, state: dict) -> None:
        """
        メモリ状態を設定

        Args:
            state: get_memory_state()で取得した状態
        """
        device = self.embed_in.weight.device
        self.hierarchical_layer.set_memory_state(state, device)
