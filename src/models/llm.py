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

    入力: [context, token_embeds]（全レイヤーでtoken継ぎ足し）
    出力: context_out

    Args:
        context_input_dim: Input context dimension
        context_output_dim: Output context dimension
        token_input_dim: Token input dimension
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

    Args:
        context_dim: Context vector dimension
        token_input_dim: Input token dimension
        token_output_dim: Output token dimension
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

    token継ぎ足し方式（2025-11-29に一本化）:
    - 全レイヤーでtoken入力
    - 次元: context_dim → context_dim（全レイヤー同じ）
    - PPL 334 vs 536（38%改善）
    - Acc 18.9% vs 15.4%（23%向上）

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

        token_input_dim = embed_dim * num_input_tokens

        # token継ぎ足し方式: 全レイヤーでtoken入力、次元は固定
        self.context_dims = [context_dim] * (num_layers + 1)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                ContextLayer(
                    context_input_dim=context_dim,
                    context_output_dim=context_dim,
                    token_input_dim=token_input_dim  # 全レイヤーでtoken入力
                )
            )

    def forward(self, context, token_embeds):
        """
        Execute all context layers sequentially

        Args:
            context: [batch, context_dim]
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            context: Updated context [batch, context_dim]
        """
        # token継ぎ足し方式: 全レイヤーでtoken入力
        for layer in self.layers:
            context = layer(context, token_embeds)
        return context

    def forward_with_intermediates(self, context, token_embeds):
        """
        Execute all context layers and return intermediate outputs (E案用)

        各レイヤーの出力を返す。TokenBlockの各レイヤーが対応する
        ContextBlockレイヤーの出力を参照するために使用。

        Args:
            context: [batch, context_dim] (初期コンテキスト)
            token_embeds: [batch, embed_dim * num_input_tokens]

        Returns:
            outputs: List of context outputs [context_1, context_2, ..., context_N]
                     len(outputs) == num_layers
        """
        outputs = []
        # token継ぎ足し方式: 全レイヤーでtoken入力
        for layer in self.layers:
            context = layer(context, token_embeds)
            outputs.append(context)
        return outputs

    def forward_with_intermediates_batch(self, contexts, token_embeds):
        """
        バッチ並列で全レイヤーの中間出力を計算

        Phase 1で確定したcontextsを入力として、各レイヤーの出力を並列計算。
        シーケンシャル処理と異なり、全トークンを同時に処理できる。

        Args:
            contexts: [num_tokens, context_dim] - 確定済みのcontext（Phase 1の出力）
            token_embeds: [num_tokens, embed_dim * num_input_tokens]

        Returns:
            outputs: [num_layers, num_tokens, context_dim] - 各レイヤーの出力
        """
        num_tokens = contexts.shape[0]
        device = contexts.device

        # 結果を格納するテンソル
        outputs = torch.zeros(
            self.num_layers, num_tokens, self.context_dim,
            device=device, dtype=contexts.dtype
        )

        # token継ぎ足し方式: 全レイヤーでtoken入力
        current_context = contexts
        for layer_idx, layer in enumerate(self.layers):
            current_context = layer(current_context, token_embeds)
            outputs[layer_idx] = current_context

        return outputs


class SplitContextBlock(nn.Module):
    """
    Split Context Block - 分割されたContextBlockのコンテナ

    N分割されたContextBlockを管理し、推論時に出力を連結する。
    各ブロックは異なるサンプルで訓練され、推論時は全ブロックを実行して
    出力を連結することで、元のcontext_dimと同じ次元の出力を生成。

    効果:
        - 計算量: 約 1/N に削減 (context_dim² → (context_dim/N)² × N)
        - パラメータ: 約 1/N に削減

    Args:
        num_splits: Number of splits
        num_layers: Number of layers per split block
        context_dim: Total context dimension (will be split into context_dim/N per block)
        embed_dim: Token embedding dimension (not split, full size to each block)
        num_input_tokens: Number of input tokens
    """

    def __init__(self, num_splits, num_layers, context_dim, embed_dim, num_input_tokens=1):
        super().__init__()

        self.num_splits = num_splits
        self.num_layers = num_layers
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.num_input_tokens = num_input_tokens

        # 分割後の各ブロックのcontext_dim
        if context_dim % num_splits != 0:
            raise ValueError(
                f"context_dim ({context_dim}) must be divisible by "
                f"num_splits ({num_splits})"
            )
        self.split_context_dim = context_dim // num_splits

        # 各分割ブロックを作成
        self.blocks = nn.ModuleList([
            ContextBlock(
                num_layers=num_layers,
                context_dim=self.split_context_dim,
                embed_dim=embed_dim,
                num_input_tokens=num_input_tokens
            )
            for _ in range(num_splits)
        ])

        # E案用: 各レイヤーの出力次元（結合後）
        # 各ブロックのcontext_dims[1:]を結合
        self.context_dims = self._compute_merged_context_dims()

    def _compute_merged_context_dims(self):
        """各レイヤーの結合後の出力次元を計算"""
        # 全ブロックの最初のブロックから次元情報を取得
        # 各ブロックは同じ構造なので、最初のブロックの次元 × num_splits
        base_dims = self.blocks[0].context_dims
        merged_dims = [dim * self.num_splits for dim in base_dims]
        return merged_dims

    def forward(self, context, token_embeds, split_id=None):
        """
        Forward pass

        Args:
            context: [batch, context_dim] or [batch, split_context_dim] (split_id指定時)
            token_embeds: [batch, embed_dim * num_input_tokens]
            split_id: None = 全ブロック実行して連結（推論用）
                      int = 指定ブロックのみ実行（訓練用）

        Returns:
            context: [batch, context_dim] (split_id=None)
                     [batch, split_context_dim] (split_id指定時)
        """
        if split_id is not None:
            # 訓練: 特定の分割のみ実行
            return self.blocks[split_id](context, token_embeds)
        else:
            # 推論: 全分割を実行して連結
            outputs = []
            for i, block in enumerate(self.blocks):
                start = i * self.split_context_dim
                end = (i + 1) * self.split_context_dim
                split_context = context[:, start:end]
                outputs.append(block(split_context, token_embeds))
            return torch.cat(outputs, dim=-1)

    def forward_with_intermediates(self, context, token_embeds, split_id=None):
        """
        Forward pass with intermediate outputs (E案用)

        Args:
            context: [batch, context_dim] or [batch, split_context_dim]
            token_embeds: [batch, embed_dim * num_input_tokens]
            split_id: None = 全ブロックの出力を連結
                      int = 指定ブロックのみ

        Returns:
            outputs: List of context outputs [context_1, ..., context_N]
                     split_id=None: 各要素は結合された次元
                     split_id指定: 各要素は分割された次元
        """
        if split_id is not None:
            # 訓練: 特定の分割のみ
            return self.blocks[split_id].forward_with_intermediates(context, token_embeds)
        else:
            # 推論: 全分割の出力を連結
            all_intermediates = []
            for i, block in enumerate(self.blocks):
                start = i * self.split_context_dim
                end = (i + 1) * self.split_context_dim
                split_context = context[:, start:end]
                intermediates = block.forward_with_intermediates(split_context, token_embeds)
                all_intermediates.append(intermediates)

            # レイヤーごとに連結
            num_layers = len(all_intermediates[0])
            merged = []
            for layer_idx in range(num_layers):
                layer_outputs = [all_intermediates[s][layer_idx] for s in range(self.num_splits)]
                merged.append(torch.cat(layer_outputs, dim=-1))
            return merged


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

    次元計算（動的）:
        入力: embed_dim * num_input_tokens
        出力: embed_dim
        各レイヤーで等差的に次元を減少

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

    token継ぎ足し方式（2025-11-29に一本化）:
    - 全レイヤーでtoken入力
    - PPL 334 vs 536（38%改善）

    ContextBlock分割機能:
    - num_context_splits > 1 の場合、ContextBlockを分割
    - 各ブロックは異なるサンプルで訓練
    - 推論時は出力を連結

    Args:
        vocab_size: Vocabulary size
        embed_dim: Token embedding dimension (単一トークンの次元)
        context_dim: Context vector dimension
        num_layers: Number of layers in both ContextBlock and TokenBlock
        num_input_tokens: Number of input tokens (1 = current only, 2+ = with history)
        num_context_splits: Number of ContextBlock splits (1 = no split)
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
        num_context_splits=1,
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
        self.num_context_splits = num_context_splits
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
        print("  token継ぎ足し方式: 全レイヤーでtoken入力")

        # ContextBlock: 文脈処理専用
        if num_context_splits > 1:
            # 分割モード: SplitContextBlockを使用
            print(f"  num_context_splits: {num_context_splits} (split mode)")
            print(f"    → Each split: context_dim={context_dim // num_context_splits}")
            self.context_block = SplitContextBlock(
                num_splits=num_context_splits,
                num_layers=num_layers,
                context_dim=context_dim,
                embed_dim=embed_dim,
                num_input_tokens=num_input_tokens
            )
        else:
            # 通常モード: 従来のContextBlockを使用
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
            saved_params = vocab_size * embed_dim
            print("✓ Weight Tying enabled: token_output shares weights with token_embedding")
            print(f"  → Saved ~{saved_params / 1e6:.2f}M parameters")
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
