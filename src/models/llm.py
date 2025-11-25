"""
New-LLM: Residual Connection Architecture

Clean implementation with inline CVFP layers.
Main features:
1. Context vector updates with residual connections
2. LayerNorm for stable training
3. GPT-2 pretrained embeddings support
"""

import torch
import torch.nn as nn


class CVFPLayer(nn.Module):
    """
    Context Vector Fixed-Point Layer - Basic computation unit

    Encapsulates:
    1. FNN-based context updates
    2. Token embedding integration
    3. Residual connections
    4. Optional LayerNorm

    Args:
        context_dim: Context vector dimension
        embed_dim: Token embedding dimension
        hidden_dim: Hidden layer dimension (must equal context_dim + embed_dim)
        layernorm_mix: LayerNorm mixing ratio (0.0 = disabled, 1.0 = full)
    """

    def __init__(self, context_dim, embed_dim, hidden_dim, layernorm_mix=0.0):
        super().__init__()

        if hidden_dim != context_dim + embed_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal "
                f"context_dim ({context_dim}) + embed_dim ({embed_dim})"
            )

        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.layernorm_mix = layernorm_mix

        # FNN: [context + token] -> [hidden_dim]
        self.fnn = nn.Sequential(
            nn.Linear(context_dim + embed_dim, hidden_dim),
            nn.ReLU()
        )

        # Optional LayerNorm
        if layernorm_mix > 0:
            self.context_norm = nn.LayerNorm(context_dim)
            self.token_norm = nn.LayerNorm(embed_dim)

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
        Forward pass: Update context and token

        Args:
            context: Current context [batch, context_dim]
            token_embed: Token embedding [batch, embed_dim]

        Returns:
            new_context: Updated context [batch, context_dim]
            new_token: Updated token embedding [batch, embed_dim]
        """
        # Concatenate inputs
        fnn_input = torch.cat([context, token_embed], dim=-1)

        # FNN forward
        fnn_output = self.fnn(fnn_input)

        # Split output
        delta_context = fnn_output[:, :self.context_dim]
        delta_token = fnn_output[:, self.context_dim:]

        # Residual connections
        new_context = context + delta_context
        new_token = token_embed + delta_token

        # Optional LayerNorm mixing
        if self.layernorm_mix > 0:
            context_normed = self.context_norm(new_context)
            token_normed = self.token_norm(new_token)

            mix = self.layernorm_mix
            new_context = (1 - mix) * new_context + mix * context_normed
            new_token = (1 - mix) * new_token + mix * token_normed

        return new_context, new_token


class CVFPBlock(nn.Module):
    """
    CVFP Block - Grouping of multiple layers

    Sequentially executes multiple CVFPLayer instances.

    Args:
        num_layers: Number of CVFP layers in this block
        context_dim: Context vector dimension
        embed_dim: Token embedding dimension
        hidden_dim: Hidden layer dimension
        layernorm_mix: LayerNorm mixing ratio
    """

    def __init__(self, num_layers, context_dim, embed_dim, hidden_dim, layernorm_mix=0.0):
        super().__init__()

        self.num_layers = num_layers

        # Stack of CVFP layers
        self.layers = nn.ModuleList([
            CVFPLayer(
                context_dim=context_dim,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                layernorm_mix=layernorm_mix
            )
            for _ in range(num_layers)
        ])

    def forward(self, context, token_embed):
        """
        Execute all layers in the block sequentially

        Args:
            context: [batch, context_dim]
            token_embed: [batch, embed_dim]

        Returns:
            context: Updated context [batch, context_dim]
            token_embed: Updated token [batch, embed_dim]
        """
        for layer in self.layers:
            context, token_embed = layer(context, token_embed)

        return context, token_embed


class LLM(nn.Module):
    """
    New-LLM with ResNet-style Residual connections

    Uses CVFPLayer to cleanly encapsulate:
    - Context updates
    - Token updates

    Args:
        vocab_size: Vocabulary size
        embed_dim: Token embedding dimension
        context_dim: Context vector dimension
        hidden_dim: Hidden layer dimension (must equal embed_dim + context_dim)
        layer_structure: List specifying number of layers per block
        layernorm_mix: LayerNorm mixing ratio, 0.0=disabled (default: 0.0)
        use_pretrained_embeddings: Whether to use GPT-2 pretrained embeddings
    """

    def __init__(
        self,
        vocab_size,
        embed_dim,
        context_dim,
        hidden_dim,
        layer_structure,
        layernorm_mix=0.0,
        use_pretrained_embeddings=False
    ):
        super().__init__()

        # Validate dimensions
        if hidden_dim != embed_dim + context_dim:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must equal "
                f"embed_dim ({embed_dim}) + context_dim ({context_dim})"
            )

        # Save configuration
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.layer_structure = layer_structure
        self.use_pretrained_embeddings = use_pretrained_embeddings

        # ========== Token Embeddings ==========
        if use_pretrained_embeddings:
            self._load_pretrained_embeddings()
        else:
            self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.embed_norm = nn.LayerNorm(embed_dim)

        # ========== CVFP Blocks ==========
        self.blocks = nn.ModuleList([
            CVFPBlock(
                num_layers=num_layers,
                context_dim=context_dim,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                layernorm_mix=layernorm_mix
            )
            for num_layers in layer_structure
        ])

        # ========== Output Head ==========
        # Prediction from concatenated context + token_embed
        self.token_output = nn.Linear(context_dim + embed_dim, vocab_size)

        # Phase 1用: ゼロ初期化 + 勾配無効化
        # Phase 2開始時に有効化される
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

    def _update_context_one_step(self, token_vec, context, return_token=False):
        """
        Update context for one token step

        Args:
            token_vec: Token vector [batch, embed_dim]
            context: Current context [batch, context_dim]
            return_token: If True, also return updated token

        Returns:
            new_context: Updated context [batch, context_dim]
            new_token: Updated token (if return_token=True)
        """
        current_context = context
        current_token = token_vec

        # Process through all blocks
        for block in self.blocks:
            current_context, current_token = block(current_context, current_token)

        if return_token:
            return current_context, current_token
        return current_context

    def forward(self, input_ids, return_context_trajectory=False):
        """
        Model forward pass

        Args:
            input_ids: Input token IDs [batch, seq_len]
            return_context_trajectory: If True, return all intermediate contexts

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            context_trajectory: (Optional) All contexts [batch, seq_len, context_dim]
        """
        batch_size, seq_len = input_ids.shape

        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)
        token_embeds = self.embed_norm(token_embeds)

        # Initialize context
        context = torch.zeros(
            batch_size, self.context_dim,
            device=input_ids.device,
            dtype=token_embeds.dtype
        )

        # Process sequence
        contexts = []
        for t in range(seq_len):
            token_vec = token_embeds[:, t, :]
            context = self._update_context_one_step(token_vec, context)
            contexts.append(context)

        # Stack contexts
        all_contexts = torch.stack(contexts, dim=1)

        # Predict next token
        logits = self.token_output(all_contexts)

        if return_context_trajectory:
            return logits, all_contexts
        return logits
