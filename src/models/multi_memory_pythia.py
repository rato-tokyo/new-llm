"""
Multi-Memory Infini-Pythia Model Implementation

1層目: Multi-Memory Infini-Attention (Attention-based selection)
2層目以降: 標準Pythia (RoPE)

アーキテクチャ:
  Token Embedding (512-dim)
         ↓
  Layer 0: MultiMemoryInfiniAttentionLayer (NoPE, Attention-based selection)
    ├─ Memory 0, 1, 2, ... (独立)
    └─ z との内積で関連度 → Softmax → 混合
         ↓
  Layer 1-5: PythiaLayer (RoPE)
         ↓
  Output Head (512 → vocab)

特徴:
- 複数の独立したメモリを保持
- クエリとの関連度（phi(Q) @ z）でメモリを動的選択
- 追加パラメータなし（学習が安定）
"""

from typing import Optional

import torch
import torch.nn as nn

from src.models.multi_memory_attention import MultiMemoryInfiniAttentionLayer
from src.models.pythia import PythiaLayer


class MultiMemoryInfiniPythiaModel(nn.Module):
    """
    Multi-Memory Infini-Pythia Model

    1層目: Multi-Memory Infini-Attention (Attention-based selection)
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
        num_memories: int = 4,
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
            num_memories: メモリの数
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_memories = num_memories

        # Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # Layer 0: Multi-Memory Infini-Attention
        self.infini_layer = MultiMemoryInfiniAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_memories=num_memories,
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
            update_memory: メモリを更新するか

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        hidden_states = self.embed_in(input_ids)

        hidden_states = self.infini_layer(
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

        infini_params = sum(p.numel() for p in self.infini_layer.parameters())
        pythia_params = sum(
            sum(p.numel() for p in layer.parameters())
            for layer in self.pythia_layers
        )

        return {
            "total": total,
            "embedding": embedding,
            "lm_head": lm_head,
            "infini_layer": infini_params,
            "pythia_layers": pythia_params,
            "transformer": infini_params + pythia_params,
        }

    def memory_info(self) -> dict:
        """メモリ情報を取得"""
        return self.infini_layer.attention.memory_info()

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
        return self.infini_layer.get_memory_state()

    def set_memory_state(self, state: dict) -> None:
        """
        メモリ状態を設定

        Args:
            state: get_memory_state()で取得した状態
        """
        device = self.embed_in.weight.device
        self.infini_layer.set_memory_state(state, device)

    def get_memory_weights(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        デバッグ用: 各メモリの重みを取得

        Args:
            input_ids: [batch, seq_len]

        Returns:
            weights: [num_memories, batch, heads]
        """
        hidden_states = self.embed_in(input_ids)
        hidden_states = self.infini_layer.input_layernorm(hidden_states)

        q = self.infini_layer.attention.w_q(hidden_states)
        batch_size, seq_len, _ = q.shape
        q = q.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        return self.infini_layer.attention.get_memory_weights(q)
