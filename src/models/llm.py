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

    入力: [context] または [context, token_embeds]（最初のレイヤーのみ）
    出力: context_out（contextのみ、tokenは出力しない）

    等差減少設計: 入力context次元から出力context次元へ段階的に縮小

    Args:
        context_input_dim: Input context dimension
        context_output_dim: Output context dimension
        token_input_dim: Token input dimension (0 = no token input)
    """

    def __init__(self, context_input_dim, context_output_dim, token_input_dim=0):
        super().__init__()

        self.context_input_dim = context_input_dim
        self.context_output_dim = context_output_dim
        self.token_input_dim = token_input_dim

        # FNN: [context (+ token_embeds)] -> context_output_dim
        input_dim = context_input_dim + token_input_dim
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, context_output_dim),
            nn.ReLU()
        )

        # LayerNorm（必須：数値安定性のため）
        self.context_norm = nn.LayerNorm(context_output_dim)

        # 残差接続用の射影レイヤー（次元が異なる場合のみ）
        if context_input_dim != context_output_dim:
            self.residual_proj = nn.Linear(context_input_dim, context_output_dim)
        else:
            self.residual_proj = None

        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                if module.bias is not None:
                    nn.init.normal_(module.bias, mean=0.0, std=0.01)

    def forward(self, context, token_embeds=None):
        """
        Forward pass: Update context only

        Args:
            context: Current context [batch, context_input_dim]
            token_embeds: Token embeddings [batch, token_input_dim] (optional)
                          最初のレイヤーのみ使用、それ以外はNone

        Returns:
            new_context: Updated context [batch, context_output_dim]
        """
        # Concatenate inputs
        if token_embeds is not None and self.token_input_dim > 0:
            fnn_input = torch.cat([context, token_embeds], dim=-1)
        else:
            fnn_input = context

        # FNN forward -> delta_context
        delta_context = self.fnn(fnn_input)

        # 残差接続（次元が異なる場合は射影）
        if self.residual_proj is not None:
            residual = self.residual_proj(context)
        else:
            residual = context

        # Residual connection + LayerNorm
        new_context = self.context_norm(residual + delta_context)

        return new_context


class TokenLayer(nn.Module):
    """
    Token Layer - トークン処理専用レイヤー

    入力: [context, token_embeds]
    出力: token_out（tokenのみ更新、contextは参照のみ）

    等差減少設計: 入力トークン次元から出力トークン次元へ段階的に縮小

    Args:
        context_dim: Context vector dimension
        token_input_dim: Input token dimension (前のレイヤーからの出力次元)
        token_output_dim: Output token dimension (このレイヤーの出力次元)
    """

    def __init__(self, context_dim, token_input_dim, token_output_dim):
        super().__init__()

        self.context_dim = context_dim
        self.token_input_dim = token_input_dim
        self.token_output_dim = token_output_dim

        # FNN: [context + token_embeds] -> token_output_dim
        input_dim = context_dim + token_input_dim
        self.fnn = nn.Sequential(
            nn.Linear(input_dim, token_output_dim),
            nn.ReLU()
        )

        # LayerNorm（必須：数値安定性のため）
        self.token_norm = nn.LayerNorm(token_output_dim)

        # 残差接続用の射影レイヤー（次元が異なる場合のみ）
        if token_input_dim != token_output_dim:
            self.residual_proj = nn.Linear(token_input_dim, token_output_dim)
        else:
            self.residual_proj = None

        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                if module.bias is not None:
                    nn.init.normal_(module.bias, mean=0.0, std=0.01)

    def forward(self, context, token_embeds):
        """
        Forward pass: Update token only

        Args:
            context: Context vector [batch, context_dim] (参照のみ)
            token_embeds: Token embeddings [batch, token_input_dim]

        Returns:
            new_token: Updated token [batch, token_output_dim]
        """
        # Concatenate inputs
        fnn_input = torch.cat([context, token_embeds], dim=-1)

        # FNN forward -> delta_token
        delta_token = self.fnn(fnn_input)

        # 残差接続（次元が異なる場合は射影）
        if self.residual_proj is not None:
            residual = self.residual_proj(token_embeds)
        else:
            residual = token_embeds

        # Residual connection + LayerNorm
        new_token = self.token_norm(residual + delta_token)

        return new_token


class ContextBlock(nn.Module):
    """
    Context Block - 文脈処理ブロック（複数レイヤー）

    Phase 1で学習、Phase 2でfreeze

    等差減少設計:
        入力: context_dim + embed_dim * num_input_tokens（最初のレイヤーのみ）
        出力: context_dim
        各レイヤーで等差的に次元を減少させる
        2番目以降のレイヤーはtoken入力なし

    例: num_input_tokens=2, num_layers=6, context_dim=768, embed_dim=768
        Layer 0: context(768) + token(1536) → 2176  (token入力あり)
        Layer 1: 2176 → 1894                         (token入力なし)
        Layer 2: 1894 → 1613
        Layer 3: 1613 → 1331
        Layer 4: 1331 → 1050
        Layer 5: 1050 → 768

    Args:
        num_layers: Number of context layers
        context_dim: Final context vector dimension
        embed_dim: Token embedding dimension (単一トークンの次元)
        num_input_tokens: Number of input tokens (1 = current only, 2+ = with history)
    """

    def __init__(self, num_layers, context_dim, embed_dim, num_input_tokens=1):
        super().__init__()

        self.num_layers = num_layers
        self.num_input_tokens = num_input_tokens
        self.context_dim = context_dim
        self.embed_dim = embed_dim

        # 等差減少の次元計算
        # 最初のレイヤー入力: context_dim + token_dim
        # 最終出力: context_dim
        token_input_dim = embed_dim * num_input_tokens
        input_context_dim = context_dim + token_input_dim  # 最初のレイヤー後の次元
        output_context_dim = context_dim
        total_reduction = input_context_dim - output_context_dim

        # 各レイヤーの入出力次元を計算
        self.context_dims = []
        for i in range(num_layers + 1):
            # 線形補間: input_dim から output_dim へ
            dim = input_context_dim - (total_reduction * i) // num_layers
            self.context_dims.append(dim)

        # Stack of Context layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # 最初のレイヤーのみtoken入力あり
            layer_token_dim = token_input_dim if i == 0 else 0
            # 最初のレイヤーはcontext_dimから開始
            layer_input_dim = context_dim if i == 0 else self.context_dims[i]

            self.layers.append(
                ContextLayer(
                    context_input_dim=layer_input_dim,
                    context_output_dim=self.context_dims[i + 1],
                    token_input_dim=layer_token_dim
                )
            )

    def forward(self, context, token_embeds):
        """
        Execute all context layers sequentially

        Args:
            context: [batch, context_dim]
            token_embeds: [batch, embed_dim * num_input_tokens] (最初のレイヤーのみ使用)

        Returns:
            context: Updated context [batch, context_dim]
        """
        for i, layer in enumerate(self.layers):
            if i == 0:
                context = layer(context, token_embeds)
            else:
                context = layer(context)

        return context

    def forward_with_intermediates(self, context, token_embeds):
        """
        Execute all context layers and return intermediate outputs (E案用)

        各レイヤーの出力を返す。TokenBlockの各レイヤーが対応する
        ContextBlockレイヤーの出力を参照するために使用。

        Args:
            context: [batch, context_dim] (初期コンテキスト)
            token_embeds: [batch, embed_dim * num_input_tokens] (最初のレイヤーのみ使用)

        Returns:
            outputs: List of context outputs [context_1, context_2, ..., context_N]
                     len(outputs) == num_layers
        """
        outputs = []
        for i, layer in enumerate(self.layers):
            if i == 0:
                context = layer(context, token_embeds)
            else:
                context = layer(context)
            outputs.append(context)
        return outputs


class TokenBlock(nn.Module):
    """
    Token Block - トークン処理ブロック（複数レイヤー）

    Phase 2で学習

    等差減少設計:
        入力: embed_dim * num_input_tokens
        出力: embed_dim
        各レイヤーで等差的に次元を減少させる

    E案対応:
        各レイヤーはContextBlockの対応するレイヤーの出力を参照
        ContextBlockの出力次元もレイヤーごとに異なる

    例: num_input_tokens=2, num_layers=6, embed_dim=768
        Layer 0: token(1536) + context(2048) → 1408
        Layer 1: token(1408) + context(1792) → 1280
        Layer 2: token(1280) + context(1536) → 1152
        Layer 3: token(1152) + context(1280) → 1024
        Layer 4: token(1024) + context(1024) → 896
        Layer 5: token(896)  + context(768)  → 768

    Args:
        num_layers: Number of token layers
        context_dim: Final context vector dimension
        embed_dim: Token embedding dimension (最終出力次元)
        num_input_tokens: Number of input tokens (1 = current only, 2+ = with history)
        context_dims_list: List of context dimensions from ContextBlock (for E案)
    """

    def __init__(self, num_layers, context_dim, embed_dim, num_input_tokens=1,
                 context_dims_list=None):
        super().__init__()

        self.num_layers = num_layers
        self.num_input_tokens = num_input_tokens
        self.embed_dim = embed_dim

        # 等差減少の次元計算（トークン側）
        # 入力次元: embed_dim * num_input_tokens
        # 出力次元: embed_dim
        input_token_dim = embed_dim * num_input_tokens
        output_token_dim = embed_dim
        total_reduction = input_token_dim - output_token_dim

        # 各レイヤーの入出力次元を計算
        self.token_dims = []
        for i in range(num_layers + 1):
            # 線形補間: input_dim から output_dim へ
            dim = input_token_dim - (total_reduction * i) // num_layers
            self.token_dims.append(dim)

        # ContextBlockからの次元リスト（E案用）
        # context_dims_listはContextBlockのcontext_dims[1:]に相当
        if context_dims_list is None:
            # 後方互換性: 固定次元
            self.context_dims_list = [context_dim] * num_layers
        else:
            self.context_dims_list = context_dims_list

        # Stack of Token layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TokenLayer(
                    context_dim=self.context_dims_list[i],
                    token_input_dim=self.token_dims[i],
                    token_output_dim=self.token_dims[i + 1]
                )
            )

    def forward(self, context, token_embeds):
        """
        Execute all token layers sequentially (A案: 全レイヤーで同じcontext)

        Args:
            context: [batch, context_dim] (参照のみ、更新されない)
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            token: Updated token [batch, embed_dim]
        """
        for layer in self.layers:
            # 各レイヤーの出力は [batch, embed_dim]
            # 次のレイヤーへの入力は最後のトークン + 履歴として再構成
            token_embeds = layer(context, token_embeds)

        return token_embeds

    def forward_with_contexts(self, context_list, token_embeds):
        """
        Execute all token layers with layer-specific contexts (E案用)

        各レイヤーが対応するContextBlockレイヤーの出力を使用する。
        TokenBlock Layer i は context_list[i] を参照。

        Args:
            context_list: List of context outputs from ContextBlock
                          [context_1, context_2, ..., context_N]
                          len(context_list) == num_layers
            token_embeds: [batch, embed_dim * num_input_tokens] (初期トークン)

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
            token_embeds = layer(context_list[i], token_embeds)

        return token_embeds


class LLM(nn.Module):
    """
    New-LLM with Separated Context/Token Architecture (E案)

    分離アーキテクチャ:
    - ContextBlock: 文脈処理専用（Phase 1で学習、Phase 2でfreeze）
    - TokenBlock: トークン処理専用（Phase 2で学習）

    E案: TokenBlock Layer i は ContextBlock Layer i の出力を参照

    Args:
        vocab_size: Vocabulary size
        embed_dim: Token embedding dimension (単一トークンの次元)
        context_dim: Context vector dimension
        num_layers: Number of layers in both ContextBlock and TokenBlock
        num_input_tokens: Number of input tokens (1 = current only, 2+ = with history)
        use_pretrained_embeddings: Whether to use GPT-2 pretrained embeddings
        use_weight_tying: Whether to tie token_embedding and token_output weights
                          (reduces parameters by ~38M, GPT-2 style)
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        context_dim,
        num_layers=6,
        num_input_tokens=1,
        use_pretrained_embeddings=True,
        use_weight_tying=False
    ):
        super().__init__()

        # Save configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.num_input_tokens = num_input_tokens
        self.use_separated_architecture = True  # Always true now
        self.use_weight_tying = use_weight_tying

        # ========== Token Embeddings ==========
        if use_pretrained_embeddings:
            self._load_pretrained_embeddings()
        else:
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.embed_norm = nn.LayerNorm(embed_dim)

        # ========== Separated Architecture ==========
        print(f"Using E案 architecture: ContextBlock({num_layers} layers) + TokenBlock({num_layers} layers)")
        print(f"  num_input_tokens: {num_input_tokens}")

        # ContextBlock: 文脈処理専用
        self.context_block = ContextBlock(
            num_layers=num_layers,
            context_dim=context_dim,
            embed_dim=embed_dim,
            num_input_tokens=num_input_tokens
        )

        # TokenBlock: トークン処理専用
        # E案: ContextBlockの各レイヤー出力次元をTokenBlockに渡す
        # context_dims[1:]は各レイヤーの出力次元
        context_dims_for_token = self.context_block.context_dims[1:]
        self.token_block = TokenBlock(
            num_layers=num_layers,
            context_dim=context_dim,
            embed_dim=embed_dim,
            num_input_tokens=num_input_tokens,
            context_dims_list=context_dims_for_token
        )

        # ========== Output Head ==========
        if use_weight_tying:
            # Weight Tying: Token EmbeddingとOutput Headで重みを共有
            # GPT-2と同じ手法、パラメータを約38M削減
            self.token_output = nn.Linear(embed_dim, vocab_size, bias=False)
            # 重み共有（転置の関係: embedding [vocab, embed] → output [embed, vocab].T）
            self.token_output.weight = self.token_embedding.weight
            print("✓ Weight Tying enabled: token_output shares weights with token_embedding")
            print(f"  → Saved ~{vocab_size * embed_dim / 1e6:.2f}M parameters")
        else:
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

    def forward_context(self, context, token_embeds):
        """
        ContextBlock forward pass (Phase 1用)

        Args:
            context: [batch, context_dim]
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            new_context: [batch, context_dim]
        """
        return self.context_block(context, token_embeds)

    def forward_context_with_intermediates(self, context, token_embeds):
        """
        ContextBlock forward pass with intermediate outputs (E案用)

        Args:
            context: [batch, context_dim] (初期コンテキスト)
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            context_outputs: List of context outputs [context_1, ..., context_N]
        """
        return self.context_block.forward_with_intermediates(context, token_embeds)

    def forward_token_e(self, context_list, token_embeds):
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

    def freeze_context_block(self):
        """ContextBlockのパラメータをfreezeする（Phase 2用）"""
        for param in self.context_block.parameters():
            param.requires_grad = False
        print("✓ ContextBlock frozen")

    def unfreeze_token_output(self, freeze_embedding: bool = False):
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

    def _update_context_one_step(self, token_embeds, context):
        """
        Update context for one token step (診断用)

        Args:
            token_embeds: Token embeddings [batch, embed_dim * num_input_tokens]
            context: Current context [batch, context_dim]

        Returns:
            new_context: Updated context [batch, context_dim]
        """
        return self.context_block(context, token_embeds)
