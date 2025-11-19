"""New-LLM: FNN-based Language Model with Context Vector Propagation"""

import torch
import torch.nn as nn


class ContextVectorLLM(nn.Module):
    """
    New architecture: FNN-based LM with context vector propagation.

    Key ideas:
    1. No attention mechanism
    2. Instead, concatenate a "context vector" to each token embedding
    3. FNN outputs both: next token prediction + updated context vector
    4. Context vector is additively updated and passed to next position
    5. First token has zero context vector

    During training:
    - Only the token prediction loss is used (not context vector loss)
    - Context vector updates emerge from optimizing token predictions
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.context_dim = config.context_vector_dim

        # Token embedding
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # NOTE: No positional embedding - position information is implicitly learned
        # through sequential context vector updates, similar to RNN/LSTM.
        # This maintains New-LLM's core principle: fixed memory usage regardless of sequence length.

        # FNN input size: embed_dim + context_vector_dim
        fnn_input_dim = config.embed_dim + self.context_dim

        # Feedforward layers
        layers = []
        for i in range(config.num_layers):
            if i == 0:
                # First layer takes concatenated input
                layers.extend([
                    nn.Linear(fnn_input_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ])
            else:
                layers.extend([
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ])

        self.fnn_layers = nn.Sequential(*layers)

        # Output heads
        # 1. Token prediction head
        self.token_output = nn.Linear(config.hidden_dim, config.vocab_size)

        # 2. Context vector update head (outputs delta to add to context)
        self.context_update = nn.Linear(config.hidden_dim, self.context_dim)

        # 3. Gated context update (LSTM-style gates)
        self.forget_gate = nn.Linear(config.hidden_dim, self.context_dim)
        self.input_gate = nn.Linear(config.hidden_dim, self.context_dim)

        # 4. Layer normalization for context vector (CRITICAL for stability)
        self.context_norm = nn.LayerNorm(self.context_dim)

        # 5. Context Decoder for reconstruction learning (NEW)
        # Reconstructs [prev_context + current_token_embedding] from context vector
        self.context_decoder = nn.Sequential(
            nn.Linear(self.context_dim, config.embed_dim + self.context_dim),
            nn.ReLU(),
            nn.Linear(config.embed_dim + self.context_dim, config.embed_dim + self.context_dim),
        )

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

    def forward(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with context vector propagation

        Args:
            input_ids: Token indices [batch_size, seq_len]

        Returns:
            logits: Next token predictions [batch_size, seq_len, vocab_size]
            context_trajectory: Context vectors at each position [batch_size, seq_len, context_dim]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch, seq, embed]

        # No positional embeddings - position information emerges from sequential processing
        # This allows New-LLM to handle arbitrary sequence lengths without memory growth

        # Initialize context vector (zeros for first token)
        context = torch.zeros(batch_size, self.context_dim, device=device)

        logits_list = []
        context_list = []
        reconstruction_targets = []  # NEW: Store [prev_context + current_token] for reconstruction

        # Process each position sequentially (context propagation)
        for t in range(seq_len):
            # Get current token embedding
            current_token = token_embeds[:, t, :]  # [batch, embed]

            # Store reconstruction target: [prev_context, current_token]
            # This is what the context_decoder should reconstruct
            reconstruction_target = torch.cat([context, current_token], dim=-1)  # [batch, context_dim + embed]
            reconstruction_targets.append(reconstruction_target)

            # Concatenate token embedding with context vector
            fnn_input = torch.cat([current_token, context], dim=-1)  # [batch, embed + context_dim]

            # Pass through FNN
            hidden = self.fnn_layers(fnn_input)  # [batch, hidden_dim]

            # Get token prediction
            token_logits = self.token_output(hidden)  # [batch, vocab_size]
            logits_list.append(token_logits)

            # Get context update (delta to add)
            context_delta = self.context_update(hidden)  # [batch, context_dim]

            # Gated context update (LSTM-style)
            forget_g = torch.sigmoid(self.forget_gate(hidden))  # [batch, context_dim]
            input_g = torch.sigmoid(self.input_gate(hidden))    # [batch, context_dim]

            # Store current context before update
            context_list.append(context.clone())

            # Update context with gates: forget old + accept new
            context = forget_g * context + input_g * context_delta  # [batch, context_dim]

            # Normalize context vector (CRITICAL: prevents unbounded growth)
            context = self.context_norm(context)

            # Clip context values to prevent extreme values
            context = torch.clamp(context, min=-10.0, max=10.0)

        # Stack results
        logits = torch.stack(logits_list, dim=1)  # [batch, seq, vocab_size]
        context_trajectory = torch.stack(context_list, dim=1)  # [batch, seq, context_dim]
        reconstruction_targets_tensor = torch.stack(reconstruction_targets, dim=1)  # [batch, seq, context_dim + embed]

        # Store for reconstruction learning
        self.context_history = context_list  # List of (batch, context_dim) tensors
        self.reconstruction_targets = reconstruction_targets_tensor

        return logits, context_trajectory

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 20, temperature: float = 1.0) -> torch.Tensor:
        """
        Generate new tokens autoregressively with context propagation

        Args:
            input_ids: Starting tokens [batch_size, seq_len]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature

        Returns:
            generated: Generated sequence [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device

        # Initialize context from input sequence
        with torch.no_grad():
            _, context_traj = self.forward(input_ids)
            context = context_traj[:, -1, :]  # Last context vector

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get last token
                last_token_id = input_ids[:, -1:]  # [batch, 1]

                # Get token embedding
                token_embed = self.token_embedding(last_token_id).squeeze(1)  # [batch, embed]

                # No positional embedding - position information is implicitly maintained
                # in the context vector through sequential updates

                # Concatenate with context
                fnn_input = torch.cat([token_embed, context], dim=-1)

                # FNN forward
                hidden = self.fnn_layers(fnn_input)

                # Get next token prediction
                next_token_logits = self.token_output(hidden) / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Update context with gating
                context_delta = self.context_update(hidden)
                forget_g = torch.sigmoid(self.forget_gate(hidden))
                input_g = torch.sigmoid(self.input_gate(hidden))
                context = forget_g * context + input_g * context_delta

                # Normalize context vector
                context = self.context_norm(context)

                # Clip context values to prevent extreme values
                context = torch.clamp(context, min=-10.0, max=10.0)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Truncate if needed
                if input_ids.size(1) > self.config.max_seq_length * 2:
                    input_ids = input_ids[:, -self.config.max_seq_length:]

        return input_ids
