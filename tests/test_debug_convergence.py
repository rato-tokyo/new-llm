"""Debug convergence behavior

This script shows detailed information about what happens during
fixed-point iteration for a single simple token.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.models.new_llm_flexible import NewLLMFlexible
from src.utils.dialogue_config import TinyDialogueConfig
from transformers import AutoTokenizer


def debug_single_token():
    """Debug convergence for a single simple token"""

    print("=" * 60)
    print("DEBUG: Single Token Convergence")
    print("=" * 60)

    # Initialize model
    config = TinyDialogueConfig()
    config.device = "cpu"
    model = NewLLMFlexible(config).to(config.device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/DialoGPT-small",
        cache_dir="cache/tokenizer"
    )

    # Test single word
    sentence = "hello"
    tokens = tokenizer.encode(sentence)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=config.device)

    print(f"\nTesting: \"{sentence}\"")
    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")

    # Manual fixed-point iteration with debug output
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Token embeddings
    token_embeds = model.token_embedding(input_ids)
    token_embeds = model.embed_norm(token_embeds)

    with torch.no_grad():
        for t in range(seq_len):
            current_token = token_embeds[:, t, :]
            print(f"\n--- Token {t} (id={tokens[t]}) ---")

            # Initialize context
            context = torch.zeros(batch_size, model.context_dim, device=device)

            print(f"Initial context norm: {torch.norm(context).item():.6f}")

            # Fixed-point iteration with detailed logging
            warmup_iterations = 10
            max_iterations = 50
            tolerance = 1e-2  # Relaxed threshold

            distances = []

            for iteration in range(max_iterations):
                fnn_input = torch.cat([current_token, context], dim=-1)
                hidden = model.fnn(fnn_input)

                # Update context
                context_delta = torch.tanh(model.context_delta_proj(hidden))
                forget = torch.sigmoid(model.forget_gate(hidden))
                input_g = torch.sigmoid(model.input_gate(hidden))

                context_new = forget * context + input_g * context_delta
                context_new = model.context_norm(context_new)
                context_new = torch.clamp(context_new, min=-10.0, max=10.0)

                # Calculate distance
                delta = torch.norm(context_new - context, dim=-1).item()
                distances.append(delta)

                # Log first 10 and last 10 iterations
                if iteration < 10 or iteration >= max_iterations - 10:
                    print(f"  Iter {iteration:2d}: distance={delta:.6f} | forget={forget.mean().item():.4f} | input_g={input_g.mean().item():.4f}")

                # Check convergence after warmup
                if iteration >= warmup_iterations:
                    if delta < tolerance:
                        print(f"\n✓ CONVERGED at iteration {iteration}")
                        print(f"  Final distance: {delta:.6f}")
                        print(f"  Final context norm: {torch.norm(context_new).item():.6f}")
                        break

                context = context_new
            else:
                print(f"\n✗ DID NOT CONVERGE after {max_iterations} iterations")
                print(f"  Final distance: {delta:.6f}")
                print(f"  Final context norm: {torch.norm(context).item():.6f}")

            # Summary statistics
            print(f"\nDistance statistics:")
            print(f"  Min: {min(distances):.6f}")
            print(f"  Max: {max(distances):.6f}")
            print(f"  Mean: {sum(distances)/len(distances):.6f}")
            print(f"  Last 10 mean: {sum(distances[-10:])/10:.6f}")

            # Check if distance is decreasing
            if len(distances) >= 20:
                early_mean = sum(distances[10:20]) / 10
                late_mean = sum(distances[-10:]) / 10
                print(f"  Early (iter 10-20) mean: {early_mean:.6f}")
                print(f"  Late (last 10) mean: {late_mean:.6f}")
                if late_mean < early_mean:
                    print(f"  → Distance IS decreasing (good)")
                else:
                    print(f"  → Distance NOT decreasing (problem!)")


if __name__ == "__main__":
    if not os.path.exists("cache/tokenizer"):
        print("❌ ERROR: Tokenizer not found")
        print("   Please run train_dialogue.py first")
    else:
        debug_single_token()
