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

import torch
import torch.nn as nn


class ContextLayer(nn.Module):
    """
    Context Layer - 文脈処理専用レイヤー

    入力: [context, token_embed]
    出力: context_out（contextのみ、tokenは出力しない）

    Args:
        context_dim: Context vector dimension
        embed_dim: Token embedding dimension
    """

    def __init__(self, context_dim, embed_dim):
        super().__init__()

        self.context_dim = context_dim
        self.embed_dim = embed_dim

        # FNN: [context + token] -> context_dim
        self.fnn = nn.Sequential(
            nn.Linear(context_dim + embed_dim, context_dim),
            nn.ReLU()
        )

        # LayerNorm（必須：数値安定性のため）
        self.context_norm = nn.LayerNorm(context_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                if module.bias is not None:
                    nn.init.normal_(module.bias, mean=0.0, std=0.01)

    def forward(self, context, token_embed):
        """
        Forward pass: Update context only

        Args:
            context: Current context [batch, context_dim]
            token_embed: Token embedding [batch, embed_dim] (入力のみ)

        Returns:
            new_context: Updated context [batch, context_dim]
        """
        # Concatenate inputs
        fnn_input = torch.cat([context, token_embed], dim=-1)

        # FNN forward -> delta_context
        delta_context = self.fnn(fnn_input)

        # Residual connection + LayerNorm
        new_context = self.context_norm(context + delta_context)

        return new_context


class TokenLayer(nn.Module):
    """
    Token Layer - トークン処理専用レイヤー

    入力: [context, token]
    出力: token_out（tokenのみ更新、contextは参照のみ）

    Args:
        context_dim: Context vector dimension
        embed_dim: Token embedding dimension
    """

    def __init__(self, context_dim, embed_dim):
        super().__init__()

        self.context_dim = context_dim
        self.embed_dim = embed_dim

        # FNN: [context + token] -> embed_dim
        self.fnn = nn.Sequential(
            nn.Linear(context_dim + embed_dim, embed_dim),
            nn.ReLU()
        )

        # LayerNorm（必須：数値安定性のため）
        self.token_norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                if module.bias is not None:
                    nn.init.normal_(module.bias, mean=0.0, std=0.01)

    def forward(self, context, token):
        """
        Forward pass: Update token only

        Args:
            context: Context vector [batch, context_dim] (参照のみ)
            token: Current token [batch, embed_dim]

        Returns:
            new_token: Updated token [batch, embed_dim]
        """
        # Concatenate inputs
        fnn_input = torch.cat([context, token], dim=-1)

        # FNN forward -> delta_token
        delta_token = self.fnn(fnn_input)

        # Residual connection + LayerNorm
        new_token = self.token_norm(token + delta_token)

        return new_token


class ContextBlock(nn.Module):
    """
    Context Block - 文脈処理ブロック（複数レイヤー）

    Phase 1で学習、Phase 2でfreeze

    Args:
        num_layers: Number of context layers
        context_dim: Context vector dimension
        embed_dim: Token embedding dimension
    """

    def __init__(self, num_layers, context_dim, embed_dim):
        super().__init__()

        self.num_layers = num_layers

        # Stack of Context layers
        self.layers = nn.ModuleList([
            ContextLayer(
                context_dim=context_dim,
                embed_dim=embed_dim
            )
            for _ in range(num_layers)
        ])

    def forward(self, context, token_embed):
        """
        Execute all context layers sequentially

        Args:
            context: [batch, context_dim]
            token_embed: [batch, embed_dim] (参照のみ、更新されない)

        Returns:
            context: Updated context [batch, context_dim]
        """
        for layer in self.layers:
            context = layer(context, token_embed)

        return context

    def forward_with_intermediates(self, context, token_embed):
        """
        Execute all context layers and return intermediate outputs (E案用)

        各レイヤーの出力を返す。TokenBlockの各レイヤーが対応する
        ContextBlockレイヤーの出力を参照するために使用。

        Args:
            context: [batch, context_dim] (初期コンテキスト)
            token_embed: [batch, embed_dim] (参照のみ、更新されない)

        Returns:
            outputs: List of context outputs [context_1, context_2, ..., context_N]
                     len(outputs) == num_layers
        """
        outputs = []
        for layer in self.layers:
            context = layer(context, token_embed)
            outputs.append(context)
        return outputs


class TokenBlock(nn.Module):
    """
    Token Block - トークン処理ブロック（複数レイヤー）

    Phase 2で学習

    Args:
        num_layers: Number of token layers
        context_dim: Context vector dimension
        embed_dim: Token embedding dimension
    """

    def __init__(self, num_layers, context_dim, embed_dim):
        super().__init__()

        self.num_layers = num_layers

        # Stack of Token layers
        self.layers = nn.ModuleList([
            TokenLayer(
                context_dim=context_dim,
                embed_dim=embed_dim
            )
            for _ in range(num_layers)
        ])

    def forward(self, context, token):
        """
        Execute all token layers sequentially (A案: 全レイヤーで同じcontext)

        Args:
            context: [batch, context_dim] (参照のみ、更新されない)
            token: [batch, embed_dim]

        Returns:
            token: Updated token [batch, embed_dim]
        """
        for layer in self.layers:
            token = layer(context, token)

        return token

    def forward_with_contexts(self, context_list, token):
        """
        Execute all token layers with layer-specific contexts (E案用)

        各レイヤーが対応するContextBlockレイヤーの出力を使用する。
        TokenBlock Layer i は context_list[i] を参照。

        Args:
            context_list: List of context outputs from ContextBlock
                          [context_1, context_2, ..., context_N]
                          len(context_list) == num_layers
            token: [batch, embed_dim] (初期トークン = token_embed)

        Returns:
            token: Updated token [batch, embed_dim]

        Raises:
            ValueError: if len(context_list) != num_layers
        """
        if len(context_list) != self.num_layers:
            raise ValueError(
                f"context_list length ({len(context_list)}) must equal "
                f"num_layers ({self.num_layers})"
            )

        for i, layer in enumerate(self.layers):
            token = layer(context_list[i], token)

        return token


class LLM(nn.Module):
    """
    New-LLM with Separated Context/Token Architecture (E案)

    分離アーキテクチャ:
    - ContextBlock: 文脈処理専用（Phase 1で学習、Phase 2でfreeze）
    - TokenBlock: トークン処理専用（Phase 2で学習）

    E案: TokenBlock Layer i は ContextBlock Layer i の出力を参照

    Args:
        vocab_size: Vocabulary size
        embed_dim: Token embedding dimension
        context_dim: Context vector dimension
        context_layers: Number of layers in ContextBlock
        token_layers: Number of layers in TokenBlock (must equal context_layers)
        use_pretrained_embeddings: Whether to use GPT-2 pretrained embeddings
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        context_dim,
        context_layers=3,
        token_layers=3,
        use_pretrained_embeddings=True
    ):
        super().__init__()

        # Validate layer counts match (E案 requirement)
        if context_layers != token_layers:
            raise ValueError(
                f"E案 requires context_layers ({context_layers}) == token_layers ({token_layers})"
            )

        # Save configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.context_layers = context_layers
        self.token_layers = token_layers
        self.use_separated_architecture = True  # Always true now

        # ========== Token Embeddings ==========
        if use_pretrained_embeddings:
            self._load_pretrained_embeddings()
        else:
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.embed_norm = nn.LayerNorm(embed_dim)

        # ========== Separated Architecture ==========
        print(f"Using E案 architecture: ContextBlock({context_layers} layers) + TokenBlock({token_layers} layers)")

        # ContextBlock: 文脈処理専用
        self.context_block = ContextBlock(
            num_layers=context_layers,
            context_dim=context_dim,
            embed_dim=embed_dim
        )

        # TokenBlock: トークン処理専用
        self.token_block = TokenBlock(
            num_layers=token_layers,
            context_dim=context_dim,
            embed_dim=embed_dim
        )

        # ========== Output Head ==========
        self.token_output = nn.Linear(embed_dim, vocab_size)

        # Phase 1用: ゼロ初期化 + 勾配無効化
        with torch.no_grad():
            self.token_output.weight.fill_(0)
            self.token_output.bias.fill_(0)
        self.token_output.weight.requires_grad = False
        self.token_output.bias.requires_grad = False

        # Initialize embeddings (if not pretrained)
        if not use_pretrained_embeddings:
            self._init_weights()

    def _load_pretrained_embeddings(self):
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

    def _init_weights(self):
        """Initialize embedding weights"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

    def forward_context(self, context, token_embed):
        """
        ContextBlock forward pass (Phase 1用)

        Args:
            context: [batch, context_dim]
            token_embed: [batch, embed_dim]

        Returns:
            new_context: [batch, context_dim]
        """
        return self.context_block(context, token_embed)

    def forward_context_with_intermediates(self, context, token_embed):
        """
        ContextBlock forward pass with intermediate outputs (E案用)

        Args:
            context: [batch, context_dim] (初期コンテキスト)
            token_embed: [batch, embed_dim]

        Returns:
            context_outputs: List of context outputs [context_1, ..., context_N]
        """
        return self.context_block.forward_with_intermediates(context, token_embed)

    def forward_token_e(self, context_list, token_embed):
        """
        TokenBlock forward pass with layer-specific contexts (E案用)

        TokenBlock Layer i は context_list[i] を参照する。

        Args:
            context_list: List of context outputs from ContextBlock
                          [context_1, context_2, ..., context_N]
            token_embed: [batch, embed_dim]

        Returns:
            token_out: [batch, embed_dim]
        """
        return self.token_block.forward_with_contexts(context_list, token_embed)

    def freeze_context_block(self):
        """ContextBlockのパラメータをfreezeする（Phase 2用）"""
        for param in self.context_block.parameters():
            param.requires_grad = False
        print("✓ ContextBlock frozen")

    def unfreeze_token_output(self):
        """token_output層の勾配を有効化する（Phase 2用）"""
        self.token_output.weight.requires_grad = True
        self.token_output.bias.requires_grad = True
        print("✓ token_output layer unfrozen")

    def _update_context_one_step(self, token_vec, context):
        """
        Update context for one token step (診断用)

        Args:
            token_vec: Token vector [batch, embed_dim]
            context: Current context [batch, context_dim]

        Returns:
            new_context: Updated context [batch, context_dim]
        """
        return self.context_block(context, token_vec)
