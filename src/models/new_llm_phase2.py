"""
New-LLM Phase 2 Model: Multi-Output Architecture

Expands Phase 1 model (single output) to Phase 2 (per-block outputs).
Clones trained token_output to each block for efficient initialization.
"""

import torch
import torch.nn as nn
from .new_llm_residual import NewLLMResidual


class NewLLMPhase2(nn.Module):
    """
    Phase 2 model with per-block token outputs

    Inherits Phase 1's trained context generation.
    Each block has independent token prediction head.
    """

    def __init__(self, phase1_model):
        """
        Initialize Phase 2 model from Phase 1

        Args:
            phase1_model: Trained NewLLMResidual from Phase 1
        """
        super().__init__()

        # Copy architecture parameters
        self.vocab_size = phase1_model.vocab_size
        self.embed_dim = phase1_model.embed_dim
        self.context_dim = phase1_model.context_dim
        self.hidden_dim = phase1_model.hidden_dim
        self.layer_structure = phase1_model.layer_structure

        # ========== Inherit Phase 1 Components ==========
        # Token embedding (frozen, GPT-2 pretrained)
        self.token_embedding = phase1_model.token_embedding
        self.embed_norm = phase1_model.embed_norm

        # CVFP blocks (trained in Phase 1)
        self.blocks = phase1_model.blocks

        # ========== Create Per-Block Outputs ==========
        # Clone trained token_output to each block
        num_blocks = len(self.blocks)
        trained_output = phase1_model.token_output

        self.block_outputs = nn.ModuleList([
            self._clone_linear(trained_output)
            for _ in range(num_blocks)
        ])

        print(f"Phase 2 model created:")
        print(f"  Blocks: {num_blocks}")
        print(f"  Block outputs: {num_blocks} × {self.context_dim} → {self.vocab_size}")
        print(f"  Total output params: {num_blocks * (self.context_dim * self.vocab_size):,}")

    def _clone_linear(self, layer):
        """
        Clone a Linear layer with same weights

        Creates independent copy for fine-tuning.
        """
        new_layer = nn.Linear(layer.in_features, layer.out_features, bias=(layer.bias is not None))
        new_layer.weight.data.copy_(layer.weight.data)
        if layer.bias is not None:
            new_layer.bias.data.copy_(layer.bias.data)
        return new_layer

    def _update_context_one_step(self, token_vec, context, return_token=False):
        """
        1トークンステップのコンテキスト更新（ブロック別コンテキスト収集）

        Args:
            token_vec: トークンベクトル [batch, embed_dim]
            context: 現在のコンテキスト [batch, context_dim]
            return_token: Trueの場合、更新されたトークンも返す

        Returns:
            new_context: 更新されたコンテキスト [batch, context_dim]
            block_contexts: 各ブロックのコンテキスト [num_blocks, batch, context_dim]
        """
        current_context = context
        current_token = token_vec
        block_contexts = []

        # 全ブロックを通して処理
        for block in self.blocks:
            current_context, current_token = block(current_context, current_token)
            block_contexts.append(current_context)

        return current_context, torch.stack(block_contexts, dim=0)

    def forward(self, input_ids, return_all_logits=True):
        """
        モデルの順伝播

        Args:
            input_ids: 入力トークンID [batch, seq_len]
            return_all_logits: Trueの場合、全ブロックのlogitsを返す

        Returns:
            logits: 出力ロジット
                - return_all_logits=True: [num_blocks, batch, seq_len, vocab_size]
                - return_all_logits=False: [batch, seq_len, vocab_size] (最終ブロックのみ)
        """
        batch_size, seq_len = input_ids.shape

        # トークン埋め込みを取得
        token_embeds = self.token_embedding(input_ids)  # [batch, seq_len, embed_dim]
        token_embeds = self.embed_norm(token_embeds)

        # コンテキストを初期化
        context = torch.zeros(
            batch_size, self.context_dim,
            device=input_ids.device,
            dtype=token_embeds.dtype
        )

        # シーケンスを処理
        all_block_contexts = []  # [seq_len, num_blocks, batch, context_dim]

        for t in range(seq_len):
            token_vec = token_embeds[:, t, :]  # [batch, embed_dim]
            context, block_contexts = self._update_context_one_step(token_vec, context)
            all_block_contexts.append(block_contexts)

        # [seq_len, num_blocks, batch, context_dim] → [num_blocks, batch, seq_len, context_dim]
        all_block_contexts = torch.stack(all_block_contexts, dim=0)  # [seq_len, num_blocks, batch, context_dim]
        all_block_contexts = all_block_contexts.permute(1, 2, 0, 3)  # [num_blocks, batch, seq_len, context_dim]

        # 各ブロックの出力を計算
        block_logits = []
        for block_idx, block_output in enumerate(self.block_outputs):
            contexts = all_block_contexts[block_idx]  # [batch, seq_len, context_dim]
            logits = block_output(contexts)  # [batch, seq_len, vocab_size]
            block_logits.append(logits)

        block_logits = torch.stack(block_logits, dim=0)  # [num_blocks, batch, seq_len, vocab_size]

        if return_all_logits:
            return block_logits
        else:
            return block_logits[-1]  # 最終ブロックのみ


def expand_to_phase2(phase1_model):
    """
    Expand Phase 1 model to Phase 2 multi-output architecture

    Args:
        phase1_model: Trained NewLLMResidual model

    Returns:
        NewLLMPhase2 model with cloned outputs
    """
    return NewLLMPhase2(phase1_model)
