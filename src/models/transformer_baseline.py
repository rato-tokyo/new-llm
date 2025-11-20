"""Transformer-based Language Model with Multi-Head Self-Attention

This is the standard baseline to compare against New-LLM.
Uses self-attention mechanism like GPT.
"""

import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Q, K, V projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, embed_dim]
            mask: [batch, seq_len, seq_len] optional causal mask

        Returns:
            output: [batch, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # [batch, seq_len, 3 * embed_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # [batch, num_heads, seq_len, seq_len]

        # Apply causal mask (for autoregressive generation)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # [batch, num_heads, seq_len, head_dim]

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, embed_dim)

        # Final projection
        output = self.out_proj(attn_output)

        return output


class TransformerBlock(nn.Module):
    """Single Transformer block with attention + FFN"""

    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.1):
        super().__init__()

        # Multi-head self-attention
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, embed_dim]
            mask: causal mask

        Returns:
            output: [batch, seq_len, embed_dim]
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # FFN with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


class TransformerLM(nn.Module):
    """
    Transformer-based Language Model (GPT-like)

    Standard architecture with:
    - Token embeddings
    - Positional embeddings
    - Multi-head self-attention layers
    - Feed-forward layers
    - Layer normalization
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # Positional embedding
        self.position_embedding = nn.Embedding(config.max_seq_length, config.embed_dim)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.embed_dim,
                config.num_heads,
                config.hidden_dim,
                config.dropout
            )
            for _ in range(config.num_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.embed_dim)

        # Output projection to vocabulary
        self.output_projection = nn.Linear(config.embed_dim, config.vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def _create_causal_mask(self, seq_len, device):
        """Create causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, 0).masked_fill(mask == 0, 1)
        return mask.bool()

    def forward(self, input_ids):
        """
        Forward pass

        Args:
            input_ids: Token indices [batch_size, seq_len]

        Returns:
            logits: Next token predictions [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch, seq, embed]

        # Add positional embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)  # [1, seq, embed]
        x = self.dropout(token_embeds + pos_embeds)

        # Create causal mask
        mask = self._create_causal_mask(seq_len, device)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary
        logits = self.output_projection(x)  # [batch, seq, vocab_size]

        return logits

    def generate(self, input_ids, max_new_tokens=20, temperature=1.0):
        """
        Generate new tokens autoregressively

        Args:
            input_ids: Starting tokens [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature

        Returns:
            generated: Generated sequence [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Truncate BEFORE forward pass to avoid position embedding errors
                if input_ids.size(1) > self.config.max_seq_length:
                    input_ids = input_ids[:, -self.config.max_seq_length:]

                # Forward pass (with causal masking)
                logits = self.forward(input_ids)

                # Get logits for last position
                next_token_logits = logits[:, -1, :] / temperature

                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
