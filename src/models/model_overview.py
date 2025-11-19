"""
New-LLM Model Overview - Processing Flow
========================================

This file provides a simplified, readable overview of what the model does.
For actual implementation, see context_vector_llm.py or context_vector_llm_gated.py
"""

import torch
import torch.nn as nn


class NewLLMOverview:
    """
    Simplified overview of New-LLM processing flow.

    Key Components:
    1. Token Embedding + LayerNorm
    2. Context Vector (fixed size, 256 dims)
    3. FNN Processing
    4. Token Prediction
    5. Context Update (Simple or Gated)
    6. Reconstruction Learning
    """

    def __init__(self, config):
        # === 1. Token Embedding ===
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.token_norm = nn.LayerNorm(config.embed_dim)  # Normalize token embeddings

        # === 2. FNN Layers ===
        # Input: [token_embed (256) + context (256)] = 512 dims
        # Output: hidden (512 dims)
        self.fnn_layers = nn.Sequential(
            nn.Linear(config.embed_dim + config.context_vector_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # === 3. Output Heads ===
        # 3a. Token prediction
        self.token_output = nn.Linear(config.hidden_dim, config.vocab_size)

        # 3b. Context update
        self.context_update = nn.Linear(config.hidden_dim, config.context_vector_dim)

        # 3c. Context normalization (CRITICAL for stability)
        self.context_norm = nn.LayerNorm(config.context_vector_dim)

        # === 4. Context Decoder (Reconstruction Learning) ===
        # Reconstructs [prev_context (256) + current_token (256)] = 512 dims
        self.context_decoder = nn.Sequential(
            nn.Linear(config.context_vector_dim, config.embed_dim + config.context_vector_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim + config.context_vector_dim, config.embed_dim + config.context_vector_dim),
        )

    def forward_step_simple(self, token_id, prev_context):
        """
        SIMPLE VERSION: Direct overwrite of context vector

        Processing flow for one timestep:

        1. Embed token
        2. Normalize token embedding
        3. Concatenate with previous context
        4. FNN processing
        5. Predict next token
        6. Generate new context (direct overwrite)
        7. Normalize context
        8. Reconstruct for training

        Args:
            token_id: Current token [batch]
            prev_context: Previous context vector [batch, 256]

        Returns:
            token_logits: Next token prediction [batch, vocab_size]
            new_context: Updated context vector [batch, 256]
            reconstruction_target: Target for reconstruction loss [batch, 512]
        """
        # === STEP 1-2: Token Embedding + Normalization ===
        token_embed = self.token_embedding(token_id)  # [batch, 256]
        token_embed = self.token_norm(token_embed)    # LayerNorm

        # === STEP 3: Store reconstruction target ===
        # What should the context vector compress?
        # Answer: [previous_context + current_token_embedding]
        reconstruction_target = torch.cat([prev_context, token_embed], dim=-1)  # [batch, 512]

        # === STEP 4: Concatenate token + context ===
        fnn_input = torch.cat([token_embed, prev_context], dim=-1)  # [batch, 512]

        # === STEP 5: FNN Processing ===
        hidden = self.fnn_layers(fnn_input)  # [batch, 512]

        # === STEP 6: Token Prediction ===
        token_logits = self.token_output(hidden)  # [batch, vocab_size]

        # === STEP 7: Context Update (SIMPLE VERSION - Direct overwrite) ===
        new_context = torch.tanh(self.context_update(hidden))  # [batch, 256], bounded to [-1, 1]

        # === STEP 8: Normalize Context ===
        new_context = self.context_norm(new_context)           # LayerNorm
        new_context = torch.clamp(new_context, min=-10.0, max=10.0)  # Clipping

        # === STEP 9: Reconstruction (for training) ===
        # Context decoder tries to reconstruct [prev_context + token_embed] from new_context
        reconstructed = self.context_decoder(new_context)  # [batch, 512]

        return token_logits, new_context, reconstruction_target, reconstructed

    def forward_step_gated(self, token_id, prev_context):
        """
        GATED VERSION: LSTM-style gates for context update

        Difference from Simple version:
        - Uses forget_gate and input_gate to blend old and new context
        - More stable but more parameters

        Args:
            token_id: Current token [batch]
            prev_context: Previous context vector [batch, 256]

        Returns:
            token_logits: Next token prediction [batch, vocab_size]
            new_context: Updated context vector [batch, 256]
            reconstruction_target: Target for reconstruction loss [batch, 512]
        """
        # === STEP 1-6: Same as Simple version ===
        token_embed = self.token_embedding(token_id)
        token_embed = self.token_norm(token_embed)

        reconstruction_target = torch.cat([prev_context, token_embed], dim=-1)

        fnn_input = torch.cat([token_embed, prev_context], dim=-1)
        hidden = self.fnn_layers(fnn_input)

        token_logits = self.token_output(hidden)

        # === STEP 7: Context Update (GATED VERSION - LSTM-style) ===
        # 7a. Generate context delta
        context_delta = torch.tanh(self.context_update(hidden))  # [batch, 256]

        # 7b. Compute gates
        forget_gate = torch.sigmoid(self.forget_gate(hidden))  # [batch, 256] - How much to keep
        input_gate = torch.sigmoid(self.input_gate(hidden))    # [batch, 256] - How much to add

        # 7c. Blend old and new context
        new_context = forget_gate * prev_context + input_gate * context_delta

        # === STEP 8: Normalize Context (same as Simple) ===
        new_context = self.context_norm(new_context)
        new_context = torch.clamp(new_context, min=-10.0, max=10.0)

        # === STEP 9: Reconstruction (same as Simple) ===
        reconstructed = self.context_decoder(new_context)

        return token_logits, new_context, reconstruction_target, reconstructed


# ============================================================================
# Processing Flow Comparison
# ============================================================================

"""
SIMPLE VERSION vs GATED VERSION

┌─────────────────────────────────────────────────────────────────────────┐
│                        SIMPLE VERSION                                   │
├─────────────────────────────────────────────────────────────────────────┤
│ context_update:  new_context = tanh(W @ hidden)                         │
│                  ↓                                                       │
│                  context[t] = new_context                               │
│                  (Complete overwrite)                                   │
├─────────────────────────────────────────────────────────────────────────┤
│ Parameters:      2,209,591                                              │
│ Context Change:  ~1.35 (large changes)                                  │
│ PPL (10 epochs): 568.67                                                 │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        GATED VERSION                                    │
├─────────────────────────────────────────────────────────────────────────┤
│ context_delta:   delta = tanh(W_c @ hidden)                             │
│ forget_gate:     f_g = sigmoid(W_f @ hidden)                            │
│ input_gate:      i_g = sigmoid(W_i @ hidden)                            │
│                  ↓                                                       │
│                  context[t] = f_g * context[t-1] + i_g * delta          │
│                  (Blended update)                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ Parameters:      2,472,247                                              │
│ Context Change:  ~0.59 (smaller, more stable)                           │
│ PPL (10 epochs): 576.28                                                 │
└─────────────────────────────────────────────────────────────────────────┘


ACTIVATION FUNCTIONS & NORMALIZATION
=====================================

1. Token Embedding
   - Embedding lookup: vocab_size × embed_dim
   - LayerNorm(embed_dim)  ← NEW (standard in modern LLMs)

2. FNN Layers
   - Linear → ReLU → Dropout
   - Activation: ReLU (rectified linear unit)

3. Context Update
   - tanh activation (bounds to [-1, 1])
   - LayerNorm (mean=0, var=1)
   - torch.clamp (hard clip to [-10, 10])

4. Gates (Gated version only)
   - sigmoid activation (bounds to [0, 1])
   - Used for forget_gate and input_gate

5. Reconstruction Decoder
   - Linear → ReLU → Linear
   - Activation: ReLU


LOSS FUNCTIONS
==============

1. Token Prediction Loss (Cross-Entropy)
   token_loss = F.cross_entropy(token_logits, target_token_ids)

2. Reconstruction Loss (MSE)
   recon_loss = F.mse_loss(reconstructed, reconstruction_target)

3. Total Loss
   total_loss = token_loss + context_loss_weight * recon_loss
   (context_loss_weight = 1.0 by default)


TRAINING METRICS
================

- Train Loss:          Combined loss (token + reconstruction)
- Token Loss:          Next-token prediction accuracy
- Reconstruction Loss: How well context compresses information
- Context Change:      L2 norm of ||context[t] - context[t-1]||
- Perplexity (PPL):    exp(token_loss) - lower is better
- Accuracy:            % of correct token predictions


PARAMETERS BREAKDOWN (Simple Version, Layer 1)
==============================================

Token Embedding:     vocab_size × embed_dim = 1847 × 256 = 472,832
Token Norm:          embed_dim × 2 = 256 × 2 = 512
FNN Layer 1:         (512 × 512) + 512 = 262,656
Token Output:        (512 × vocab_size) + vocab_size = (512 × 1847) + 1847 = 947,711
Context Update:      (512 × 256) + 256 = 131,328
Context Norm:        256 × 2 = 512
Context Decoder:     (256 × 512) + 512 + (512 × 512) + 512 = 394,240
─────────────────────────────────────────────────────────────────
TOTAL:               2,209,591 parameters
"""
