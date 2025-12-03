"""
Context-Pythia: Pythia with Compressed Hidden Dimension

Token Embeddingの後にContextBlockで圧縮し、
圧縮された次元(context_dim)でTransformer Layersを動作させる。

Architecture:
1. Token Embedding (512-dim) + LayerNorm
2. ContextBlock: prev_context + token_embed → context (context_dim)
3. PythiaLayer × 6 (context_dim, RoPE)  ← 圧縮されたまま処理
4. Final LayerNorm
5. Output Head: context_dim → vocab_size

Training:
- Phase 1: ContextBlock のみ学習 (OACD)
- Phase 2: 全体をファインチューニング (ContextBlock frozen)

KV Cache削減:
- Baseline: hidden_size (512) × seq_len × num_layers
- Context-Pythia: context_dim (256) × seq_len × num_layers
- 削減率: 1 - (context_dim / hidden_size) = 50%
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

    Baselineとの違いは「Token Embedding → ContextBlock」の圧縮部分のみ。
    PythiaLayer自体は同じ構造（RoPE含む）で、hidden_size=context_dimで動作。

    Architecture:
    1. Embedding: vocab_size → embed_dim (512)
    2. LayerNorm (embed_norm)
    3. ContextBlock: prev_context + token_embed → context (context_dim)
    4. PythiaLayer × 6 (hidden_size=context_dim, RoPE)
    5. Final LayerNorm
    6. LM Head: context_dim → vocab_size
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        embed_dim: int = 512,           # Token Embedding dimension
        context_dim: int = 256,         # Compressed dimension (used for all layers)
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 1024,  # Scaled down: 2048 * (256/512) = 1024
        max_position_embeddings: int = 2048,
        rotary_pct: float = 0.25,
    ):
        super().__init__()

        # context_dimがnum_headsで割り切れない場合は自動調整
        if context_dim % num_heads != 0:
            original_context_dim = context_dim
            # 切り上げて割り切れる値にする
            context_dim = ((context_dim + num_heads - 1) // num_heads) * num_heads
            print_flush(f"⚠️ context_dim adjusted: {original_context_dim} → {context_dim} (divisible by num_heads={num_heads})")

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.num_layers = num_layers

        # Token Embedding (vocab → embed_dim)
        self.embed_in = nn.Embedding(vocab_size, embed_dim)

        # ⚠️ 重要: 埋め込み後の正規化（Phase 1収束に必須）
        self.embed_norm = nn.LayerNorm(embed_dim)

        # ContextBlock: prev_context + token_embed → context (context_dim)
        # Phase 1でOACD学習済み
        self.context_block = ContextBlock(
            context_dim=context_dim,
            embed_dim=embed_dim,
        )

        # PythiaLayer × 6 (hidden_size=context_dim)
        # Baselineと同じ構造（RoPE含む）、ただしhidden_sizeが小さい
        self.layers = nn.ModuleList([
            PythiaLayer(
                hidden_size=context_dim,  # ← ここがBaselineと違う
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                rotary_pct=rotary_pct,
                max_position_embeddings=max_position_embeddings,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(context_dim)

        # LM Head (context_dim → vocab)
        self.embed_out = nn.Linear(context_dim, vocab_size, bias=False)

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

        # Token Embedding + LayerNorm (⚠️ embed_norm必須)
        token_embeds = self.embed_in(input_ids)  # [batch, seq, embed_dim]
        token_embeds = self.embed_norm(token_embeds)

        # ContextBlock: prev_context + token_embed → context
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

        hidden_states = torch.stack(contexts, dim=1)  # [batch, seq, context_dim]

        # PythiaLayer × 6 (context_dimのまま処理)
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
