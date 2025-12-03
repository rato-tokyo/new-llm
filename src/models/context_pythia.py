"""
Context-Pythia: Pythia with Context-based Input

ContextBlockで生成されたcontext (256-dim) をhidden_size (512-dim)に射影し、
標準のPythia Transformer layersを使用する。

Architecture:
1. Token Embedding (512-dim) + LayerNorm
2. ContextBlock: prev_context + token_embed → context (256-dim)
   ⚠️ 既存の動作確認済みContextBlockを再利用
3. Context Projection: context (256-dim) → hidden_states (512-dim)
4. 6 Standard Pythia Layers (hidden_size=512)
5. Output Head: 512-dim → vocab_size

Training:
- Phase 1: ContextBlock のみ学習 (OACD)
- Phase 2: 全体をファインチューニング (ContextBlock frozen)

Note: この実装ではKVキャッシュは圧縮されません。
ContextBlockによる入力圧縮の効果を検証するための実装です。
"""

from typing import Optional, Dict

import torch
import torch.nn as nn

from src.models.pythia import PythiaLayer
from src.models.blocks import ContextBlock
from src.utils.io import print_flush


class ContextPythiaModel(nn.Module):
    """
    Context-Pythia Language Model

    Architecture:
    1. Embedding: vocab_size -> hidden_size (512)
    2. LayerNorm (embed_norm)
    3. ContextBlock: prev_context + token_embed -> context (256-dim)
    4. Context Projection: context (256) -> hidden_states (512)
    5. 6 Standard Pythia Transformer layers
    6. Final LayerNorm
    7. LM Head: hidden_size -> vocab_size
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        hidden_size: int = 512,
        context_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 2048,
        max_position_embeddings: int = 2048,
        rotary_pct: float = 0.25,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.context_dim = context_dim
        self.num_layers = num_layers

        # Embedding
        self.embed_in = nn.Embedding(vocab_size, hidden_size)

        # ⚠️ 重要: 埋め込み後の正規化（Phase 1収束に必須）
        self.embed_norm = nn.LayerNorm(hidden_size)

        # ContextBlock: prev_context + token_embed -> context (256-dim)
        # ⚠️ 既存の動作確認済みContextBlockを使用
        # 初期化は normal_(std=0.1) で行われる（削除禁止）
        self.context_block = ContextBlock(
            context_dim=context_dim,
            embed_dim=hidden_size,
        )

        # Context -> Hidden projection
        self.context_proj = nn.Linear(context_dim, hidden_size)

        # Standard Pythia Transformer layers
        self.layers = nn.ModuleList([
            PythiaLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                rotary_pct=rotary_pct,
                max_position_embeddings=max_position_embeddings,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_size)

        # LM Head
        self.embed_out = nn.Linear(hidden_size, vocab_size, bias=False)

        # Initialize weights (ContextBlock以外)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights (ContextBlock以外)"""
        for name, module in self.named_modules():
            # ContextBlockは既に初期化済み（normal_ std=0.1）なのでスキップ
            if "context_block" in name:
                continue

            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prev_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            prev_context: [batch, context_dim] 前回のcontext（オプション）

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Embedding + LayerNorm (⚠️ embed_norm必須)
        token_embeds = self.embed_in(input_ids)  # [batch, seq, hidden_size]
        token_embeds = self.embed_norm(token_embeds)  # ⚠️ Phase 1と同じ正規化を適用

        # ContextBlock: prev_context + token_embed -> context
        if prev_context is None:
            prev_context = torch.zeros(batch_size, self.context_dim, device=input_ids.device)

        # 各位置のcontextを計算（順次処理）
        contexts = []
        current_context = prev_context
        for i in range(seq_len):
            current_context = self.context_block(
                current_context,
                token_embeds[:, i, :]
            )
            contexts.append(current_context)

        context = torch.stack(contexts, dim=1)  # [batch, seq, context_dim]

        # Context -> Hidden projection
        hidden_states = self.context_proj(context)  # [batch, seq, hidden_size]

        # Standard Pythia Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        # LM Head
        logits = self.embed_out(hidden_states)

        return logits

    def freeze_context_block(self) -> None:
        """ContextBlockをfreeze（Phase 2用）"""
        for param in self.context_block.parameters():
            param.requires_grad = False
        print_flush("✓ ContextBlock frozen")

    def unfreeze_context_block(self) -> None:
        """ContextBlockをunfreeze"""
        for param in self.context_block.parameters():
            param.requires_grad = True

    def num_parameters(self) -> Dict[str, int]:
        """Count parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        context_block = sum(p.numel() for p in self.context_block.parameters())
        embedding = self.embed_in.weight.numel()
        lm_head = self.embed_out.weight.numel()

        return {
            "total": total,
            "trainable": trainable,
            "context_block": context_block,
            "embedding": embedding,
            "lm_head": lm_head,
            "transformer": total - context_block - embedding - lm_head,
        }
