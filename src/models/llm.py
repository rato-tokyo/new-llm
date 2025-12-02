"""
New-LLM: Separated Context/Token Architecture

分離アーキテクチャ:
1. ContextBlock: 文脈処理専用（Phase 1で学習、Phase 2でfreeze）
2. TokenBlock: トークン処理専用（Phase 2で学習）

Main features:
1. Context vector updates with residual connections (ContextBlock)
2. Token prediction with residual connections (TokenBlock)
3. LayerNorm for stable training
4. GPT-2 pretrained embeddings support
"""

from typing import Dict, List

import torch
import torch.nn as nn

from .blocks import ContextBlock, TokenBlock


class LLM(nn.Module):
    """
    New-LLM with Separated Context/Token Architecture

    分離アーキテクチャ:
    - ContextBlock: 文脈処理専用（Phase 1で学習、Phase 2でfreeze）
    - TokenBlock: トークン処理専用（Phase 2で学習）

    context_mode:
    - E案 (default): TokenBlock Layer i は ContextBlock Layer i の出力を参照
    - A案 (use_final_context_only=True): 全TokenBlockレイヤーがContextBlockの最終出力のみを参照
    - F案 (use_first_layer_context_only=True): 1層目のみに最終contextを注入、2層目以降はcontextなし
    - G案 (use_prev_and_current_context=True): 1層目に前のcontext、最終層に現在のcontext

    token継ぎ足し方式（2025-11-29に一本化）:
    - 全レイヤーでtoken入力

    Args:
        vocab_size: Vocabulary size
        embed_dim: Token embedding dimension (単一トークンの次元)
        context_dim: Context vector dimension
        num_layers: Number of layers in both ContextBlock and TokenBlock
        num_input_tokens: Number of input tokens (1 = current only, 2+ = with history)
        use_pretrained_embeddings: Whether to use GPT-2 pretrained embeddings
        use_weight_tying: Whether to tie token_embedding and token_output weights
                          (reduces parameters by ~38M, GPT-2 style)
        use_final_context_only: If True, use A案 (all TokenBlock layers use final context)
        use_first_layer_context_only: If True, use F案 (only first layer uses context)
        use_prev_and_current_context: If True, use G案 (first layer uses prev, last uses current)
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        context_dim: int,
        num_layers: int = 6,
        num_input_tokens: int = 1,
        use_pretrained_embeddings: bool = True,
        use_weight_tying: bool = False,
        use_final_context_only: bool = False,
        use_first_layer_context_only: bool = False,
        use_prev_and_current_context: bool = False
    ) -> None:
        super().__init__()

        # Save configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.num_input_tokens = num_input_tokens
        self.use_separated_architecture = True  # Always true now
        self.use_weight_tying = use_weight_tying
        self.use_final_context_only = use_final_context_only
        self.use_first_layer_context_only = use_first_layer_context_only
        self.use_prev_and_current_context = use_prev_and_current_context

        # ========== Token Embeddings ==========
        if use_pretrained_embeddings:
            self._load_pretrained_embeddings()
        else:
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.embed_norm = nn.LayerNorm(embed_dim)

        # ========== Separated Architecture ==========
        if use_prev_and_current_context:
            context_mode = "G案 (prev and current context)"
        elif use_first_layer_context_only:
            context_mode = "F案 (first layer context only)"
        elif use_final_context_only:
            context_mode = "A案 (final context only)"
        else:
            context_mode = "E案 (layerwise)"
        print(f"Using {context_mode} architecture: ContextBlock({num_layers} layers) + TokenBlock({num_layers} layers)")
        print(f"  num_input_tokens: {num_input_tokens}")
        print("  token継ぎ足し方式: 全レイヤーでtoken入力")

        # ContextBlock: 文脈処理専用
        self.context_block = ContextBlock(
            num_layers=num_layers,
            context_dim=context_dim,
            embed_dim=embed_dim,
            num_input_tokens=num_input_tokens,
        )

        # TokenBlock: トークン処理専用
        if use_prev_and_current_context:
            # G案: 1層目に前のcontext、最終層に現在のcontext
            self.token_block = TokenBlock(
                num_layers=num_layers,
                context_dim=context_dim,
                embed_dim=embed_dim,
                num_input_tokens=num_input_tokens,
                use_prev_and_current_context=True,
            )
        elif use_first_layer_context_only:
            # F案: 1層目のみcontext入力、2層目以降はcontextなし
            self.token_block = TokenBlock(
                num_layers=num_layers,
                context_dim=context_dim,
                embed_dim=embed_dim,
                num_input_tokens=num_input_tokens,
                use_first_layer_context_only=True,
            )
        elif use_final_context_only:
            # A案: 全レイヤーで最終context出力のみ使用
            self.token_block = TokenBlock(
                num_layers=num_layers,
                context_dim=context_dim,
                embed_dim=embed_dim,
                num_input_tokens=num_input_tokens,
                use_final_context_only=True,
            )
        else:
            # E案: ContextBlockの各レイヤー出力次元をTokenBlockに渡す
            context_dims_for_token = self.context_block.context_dims[1:]
            self.token_block = TokenBlock(
                num_layers=num_layers,
                context_dim=context_dim,
                embed_dim=embed_dim,
                num_input_tokens=num_input_tokens,
                context_dims_list=context_dims_for_token,
            )

        # ========== Output Head (Weight Tying) ==========
        # Token EmbeddingとOutput Headで重みを共有（GPT-2と同じ手法）
        self.token_output = nn.Linear(embed_dim, vocab_size, bias=False)
        self.token_output.weight = self.token_embedding.weight
        saved_params = vocab_size * embed_dim
        print("✓ Weight Tying: token_output shares weights with token_embedding")
        print(f"  → Saved ~{saved_params / 1e6:.2f}M parameters")

        # Initialize embeddings (if not pretrained)
        if not use_pretrained_embeddings:
            self._init_weights()

    def _load_pretrained_embeddings(self) -> None:
        """Load GPT-2 pretrained embeddings"""
        try:
            from transformers import GPT2Model
            print("Loading GPT-2 pretrained embeddings...")

            gpt2 = GPT2Model.from_pretrained('gpt2')
            pretrained_embeddings = gpt2.wte.weight.data

            self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            self.token_embedding.weight.data.copy_(pretrained_embeddings)
            self.token_embedding.weight.requires_grad = False

            print(f"✓ Loaded GPT-2 embeddings: {pretrained_embeddings.shape}")

        except Exception as e:
            print(f"Warning: Failed to load GPT-2 embeddings: {e}")
            print("Falling back to random initialization...")
            self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    def _init_weights(self) -> None:
        """Initialize embedding weights"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    def forward_context(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        """
        ContextBlock forward pass (Phase 1用)

        Args:
            context: [batch, context_dim]
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            new_context: [batch, context_dim]
        """
        result: torch.Tensor = self.context_block(context, token_embeds)
        return result

    def forward_context_with_intermediates(
        self, context: torch.Tensor, token_embeds: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        ContextBlock forward pass with intermediate outputs (E案用)

        Args:
            context: [batch, context_dim] (初期コンテキスト)
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            context_outputs: List of context outputs [context_1, ..., context_N]
        """
        return self.context_block.forward_with_intermediates(context, token_embeds)

    def forward_token_e(
        self, context_list: List[torch.Tensor], token_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        TokenBlock forward pass with layer-specific contexts (E案用)

        TokenBlock Layer i は context_list[i] を参照する。

        Args:
            context_list: List of context outputs from ContextBlock
                          [context_1, context_2, ..., context_N]
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            token_out: [batch, embed_dim]
        """
        return self.token_block.forward_with_contexts(context_list, token_embeds)

    def forward_token_a(
        self, final_context: torch.Tensor, token_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        TokenBlock forward pass with final context only (A案用)

        全TokenBlockレイヤーが同じfinal_contextを使用する。

        Args:
            final_context: Final context from ContextBlock [batch, context_dim]
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            token_out: [batch, embed_dim]
        """
        return self.token_block(final_context, token_embeds, context_list=None)

    def forward_token_f(
        self, final_context: torch.Tensor, token_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        TokenBlock forward pass with first layer context only (F案用)

        1層目のみfinal_contextを使用し、2層目以降はcontextなし。

        Args:
            final_context: Final context from ContextBlock [batch, context_dim]
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            token_out: [batch, embed_dim]
        """
        return self.token_block(final_context, token_embeds, context_list=None)

    def forward_token_g(
        self,
        prev_context: torch.Tensor,
        current_context: torch.Tensor,
        token_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        TokenBlock forward pass with prev and current context (G案用)

        1層目に前のcontext、最終層に現在のcontextを使用する。

        Args:
            prev_context: Previous context [batch, context_dim]
            current_context: Current context [batch, context_dim]
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            token_out: [batch, embed_dim]
        """
        return self.token_block.forward_with_prev_and_current(
            prev_context, current_context, token_embeds
        )

    def freeze_context_block(self) -> None:
        """ContextBlockのパラメータをfreezeする（Phase 2用）"""
        for param in self.context_block.parameters():
            param.requires_grad = False
        print("✓ ContextBlock frozen")

    def unfreeze_token_output(self, freeze_embedding: bool = False) -> None:
        """
        token_output層の勾配を有効化する（Phase 2用）

        Args:
            freeze_embedding: Trueの場合、Embeddingを凍結したまま維持
                             Weight Tying時はOutput Headも凍結される
                             TokenBlockのみ学習（パラメータ大幅削減）
        """
        if self.use_weight_tying:
            if freeze_embedding:
                # Embedding凍結 → Weight TyingによりOutput Headも凍結
                # TokenBlockのみ学習
                self.token_embedding.weight.requires_grad = False
                print("✓ Embedding frozen (Weight Tying: Output Head also frozen)")
                print("  → Only TokenBlock will be trained")
            else:
                # Embedding学習 → Output Headも学習
                self.token_embedding.weight.requires_grad = True
                print("✓ token_output (weight-tied with embedding) unfrozen")
                print("  Note: token_embedding will also be updated during Phase 2")
        else:
            if freeze_embedding:
                # Weight Tyingなし: Output Headのみ学習、Embedding凍結
                self.token_output.weight.requires_grad = True
                self.token_output.bias.requires_grad = True
                self.token_embedding.weight.requires_grad = False
                print("✓ token_output unfrozen, Embedding frozen")
            else:
                self.token_output.weight.requires_grad = True
                self.token_output.bias.requires_grad = True
                print("✓ token_output layer unfrozen")

    def _update_context_one_step(
        self, token_embeds: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Update context for one token step (診断用)

        Args:
            token_embeds: Token embeddings [batch, embed_dim * num_input_tokens]
            context: Current context [batch, context_dim]

        Returns:
            new_context: Updated context [batch, context_dim]
        """
        result: torch.Tensor = self.context_block(context, token_embeds)
        return result

    def num_params(self) -> Dict[str, int]:
        """
        モデル全体のパラメータ数を返す

        Returns:
            パラメータ数の詳細辞書
        """
        embedding_params = self.token_embedding.weight.numel()
        embed_norm_params = sum(p.numel() for p in self.embed_norm.parameters())
        context_block_params = self.context_block.num_params()
        token_block_params = self.token_block.num_params()
        # Weight Tyingにより output_head は追加パラメータなし
        output_head_params = 0

        total = embedding_params + embed_norm_params + context_block_params + token_block_params

        return {
            'embedding': embedding_params,
            'embed_norm': embed_norm_params,
            'context_block': context_block_params,
            'token_block': token_block_params,
            'output_head': output_head_params,
            'total': total,
            'trainable_phase1': context_block_params,
            'trainable_phase2': token_block_params,  # Embedding凍結時
        }
