"""
DProj-Pythia: Pythia with Diverse Projection

Token Embeddingの後にDiverseProjectionで圧縮し、
圧縮された次元(proj_dim)でTransformer Layersを動作させる。

Architecture:
1. Token Embedding (512-dim) + LayerNorm
2. DiverseProjection: prev_proj + token_embed → proj (proj_dim)
3. PythiaLayer × 6 (proj_dim, RoPE)  ← 圧縮されたまま処理
4. Final LayerNorm
5. Output Head: proj_dim → vocab_size

Training:
- DProj Training: DiverseProjection のみ学習 (OACD)
- Main Training: 全体をファインチューニング (DiverseProjection frozen)

KV Cache削減:
- Baseline: hidden_size (512) × seq_len × num_layers
- DProj-Pythia: proj_dim (320) × seq_len × num_layers
- 削減率: 1 - (proj_dim / hidden_size)
"""

from typing import Optional, Dict

import torch
import torch.nn as nn

from src.models.pythia import PythiaLayer
from src.models.dproj import DiverseProjection
from src.utils.io import print_flush


class DProjPythiaModel(nn.Module):
    """
    DProj-Pythia Language Model

    Baselineとの違いは「Token Embedding → DiverseProjection」の圧縮部分のみ。
    PythiaLayer自体は同じ構造（RoPE含む）で、hidden_size=proj_dimで動作。

    Architecture:
    1. Embedding: vocab_size → embed_dim (512)
    2. LayerNorm (embed_norm)
    3. DiverseProjection: prev_proj + token_embed → proj (proj_dim)
    4. PythiaLayer × 6 (hidden_size=proj_dim, RoPE)
    5. Final LayerNorm
    6. LM Head: proj_dim → vocab_size
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        embed_dim: int = 512,           # Token Embedding dimension
        proj_dim: int = 320,            # Compressed dimension (used for all layers)
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_size: int = 1024,  # Scaled down: 2048 * (320/512) ≈ 1280
        max_position_embeddings: int = 2048,
        rotary_pct: float = 0.25,
    ):
        super().__init__()

        # proj_dimがnum_headsで割り切れない場合は自動調整
        if proj_dim % num_heads != 0:
            original_proj_dim = proj_dim
            # 切り上げて割り切れる値にする
            proj_dim = ((proj_dim + num_heads - 1) // num_heads) * num_heads
            print_flush(
                f"⚠️ proj_dim adjusted: {original_proj_dim} → {proj_dim} "
                f"(divisible by num_heads={num_heads})"
            )

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.num_layers = num_layers

        # Token Embedding (vocab → embed_dim)
        self.embed_in = nn.Embedding(vocab_size, embed_dim)

        # ⚠️ 重要: 埋め込み後の正規化（DProj学習収束に必須）
        self.embed_norm = nn.LayerNorm(embed_dim)

        # DiverseProjection: prev_proj + token_embed → proj (proj_dim)
        # OACD学習済み
        self.dproj = DiverseProjection(
            proj_dim=proj_dim,
            embed_dim=embed_dim,
        )

        # PythiaLayer × 6 (hidden_size=proj_dim)
        # Baselineと同じ構造（RoPE含む）、ただしhidden_sizeが小さい
        self.layers = nn.ModuleList([
            PythiaLayer(
                hidden_size=proj_dim,  # ← ここがBaselineと違う
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                rotary_pct=rotary_pct,
                max_position_embeddings=max_position_embeddings,
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(proj_dim)

        # LM Head (proj_dim → vocab)
        self.embed_out = nn.Linear(proj_dim, vocab_size, bias=False)

        # Initialize weights (DiverseProjection以外)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights (DiverseProjection以外)"""
        for name, module in self.named_modules():
            # DiverseProjectionは既に初期化済み（normal_ std=0.1）なのでスキップ
            if "dproj" in name:
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
        prev_proj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            prev_proj: [batch, proj_dim] 前回のprojection（オプション）

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # Token Embedding + LayerNorm (⚠️ embed_norm必須)
        token_embeds = self.embed_in(input_ids)  # [batch, seq, embed_dim]
        token_embeds = self.embed_norm(token_embeds)

        # DiverseProjection: prev_proj + token_embed → proj
        if prev_proj is None:
            prev_proj = torch.zeros(batch_size, self.proj_dim, device=input_ids.device)

        # 各位置のprojectionを計算（順次処理）
        projections = []
        current_proj = prev_proj
        for i in range(seq_len):
            current_proj = self.dproj(
                current_proj,
                token_embeds[:, i, :]
            )
            projections.append(current_proj)

        hidden_states = torch.stack(projections, dim=1)  # [batch, seq, proj_dim]

        # PythiaLayer × 6 (proj_dimのまま処理)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)

        # LM Head
        logits = self.embed_out(hidden_states)

        return logits

    def freeze_dproj(self) -> None:
        """DiverseProjectionをfreeze（Main Training用）"""
        for param in self.dproj.parameters():
            param.requires_grad = False
        print_flush("✓ DiverseProjection frozen")

    def unfreeze_dproj(self) -> None:
        """DiverseProjectionをunfreeze"""
        for param in self.dproj.parameters():
            param.requires_grad = True

    def num_parameters(self) -> Dict[str, int]:
        """Count parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        dproj = sum(p.numel() for p in self.dproj.parameters())
        embedding = self.embed_in.weight.numel()
        lm_head = self.embed_out.weight.numel()

        return {
            "total": total,
            "trainable": trainable,
            "dproj": dproj,
            "embedding": embedding,
            "lm_head": lm_head,
            "transformer": total - dproj - embedding - lm_head,
        }


# Backward compatibility alias
ContextPythiaModel = DProjPythiaModel
