"""Text generation module for New-LLM

Provides text generation capabilities for trained models.
Supports various sampling strategies and chat-style interaction.
"""

import torch
import torch.nn.functional as F


class TextGenerator:
    """Text generator for New-LLM models

    Supports:
    - Greedy decoding
    - Temperature sampling
    - Top-k sampling
    - Top-p (nucleus) sampling
    """

    def __init__(self, model, tokenizer, device='cuda'):
        """
        Args:
            model: Trained ContextVectorLLM model
            tokenizer: SimpleTokenizer instance
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Move model to device and set to eval mode
        self.model.to(device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        stop_tokens: list = None,
        repetition_penalty: float = 1.0
    ) -> str:
        """Generate text from a prompt

        Args:
            prompt: Input text prompt
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
                        1.0 = normal, <1.0 = more focused, >1.0 = more random
            top_k: If > 0, only sample from top k tokens
            top_p: If < 1.0, nucleus sampling (sample from top tokens with cumulative prob >= p)
            stop_tokens: List of token IDs to stop generation
            repetition_penalty: Penalty for repeating tokens (>1.0 = discourage repetition)

        Returns:
            Generated text string
        """
        if stop_tokens is None:
            stop_tokens = []

        # Encode prompt
        tokens = self.tokenizer.encode(prompt)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)

        # Track full sequence for context management
        full_sequence = tokens.copy()

        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input (last N tokens to fit model's max_seq_length)
                # New-LLM uses context vector, so we can use recent tokens
                if len(full_sequence) > self.model.config.max_seq_length:
                    # Use last max_seq_length-1 tokens (leave room for next token)
                    input_tokens = full_sequence[-(self.model.config.max_seq_length-1):]
                else:
                    input_tokens = full_sequence

                input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(self.device)

                # Get model prediction
                output = self.model(input_tensor)

                # output can be tuple (logits, context) or just logits
                if isinstance(output, tuple):
                    logits, _ = output
                else:
                    logits = output

                # Get logits for next token (last position)
                next_token_logits = logits[0, -1, :].detach().clone()

                # Apply repetition penalty (discourage recently generated tokens)
                if repetition_penalty != 1.0:
                    # Penalize tokens that appear in the generated sequence
                    # Look at last 50 tokens to avoid repetition
                    lookback = min(50, len(full_sequence))
                    for prev_token in full_sequence[-lookback:]:
                        # Standard repetition penalty: always divide to reduce probability
                        next_token_logits[prev_token] /= repetition_penalty

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k > 0:
                    next_token_logits = self._top_k_filtering(next_token_logits, top_k)

                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    next_token_logits = self._top_p_filtering(next_token_logits, top_p)

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()

                # Check stop conditions
                if next_token == 0:  # PAD token
                    break
                if next_token in stop_tokens:
                    break

                # Add to sequence
                full_sequence.append(next_token)

        # Decode generated sequence
        generated_text = self.tokenizer.decode(full_sequence)

        return generated_text

    def chat(
        self,
        user_input: str,
        context: str = "",
        max_length: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2
    ) -> tuple:
        """Generate chat response

        Args:
            user_input: User's message
            context: Previous conversation context
            max_length: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens (default: 1.2)

        Returns:
            (response, updated_context) tuple
        """
        # Build prompt in chat format
        if context:
            prompt = f"{context}\nHuman: {user_input}\n\nAssistant:"
        else:
            prompt = f"Human: {user_input}\n\nAssistant:"

        # Generate response
        full_text = self.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

        # Extract assistant's response
        # full_text contains: prompt + generated_response
        if "Assistant:" in full_text:
            # Split by last "Assistant:" to get the response
            parts = full_text.split("Assistant:")
            response = parts[-1].strip()

            # Clean up response (stop at next "Human:" if present)
            if "Human:" in response:
                response = response.split("Human:")[0].strip()
        else:
            # Fallback: use everything after prompt
            response = full_text[len(prompt):].strip()

        # Update context
        updated_context = f"{prompt} {response}"

        return response, updated_context

    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Filter logits to only keep top k tokens"""
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        return logits

    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Filter logits using nucleus (top-p) sampling"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Scatter sorted tensors back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            0, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits
