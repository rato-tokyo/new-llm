"""Test Phase 1 training with 10 tokens

Compare Sequential vs Layer-wise with proper training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from src.models.new_llm_sequential import NewLLMSequential
from src.models.new_llm_layerwise import NewLLMLayerwise
from src.utils.dialogue_config import Small2LayerSequentialConfig, Small2LayerLayerwiseConfig


def train_10tokens(model, config, max_iterations=50, num_epochs=3):
    """Train model on 10 tokens to reach fixed points"""

    print(f"\n{'='*60}")
    print(f"Architecture: {config.architecture.upper()}")
    print(f"Epochs: {num_epochs}, Max iterations per token: {max_iterations}")
    print(f"{'='*60}")

    # Token IDs
    token_ids = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.phase1_learning_rate)

    # Get token embeddings
    model.train()
    input_ids = torch.tensor([token_ids])
    token_embeds = model.token_embedding(input_ids)
    if hasattr(model, 'embed_norm'):
        token_embeds = model.embed_norm(token_embeds)
    token_embeds = token_embeds.squeeze(0)  # [10, embed_dim]

    # Training epochs
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_converged = 0

        for t, token_embed in enumerate(token_embeds):
            # Initialize context
            context = torch.zeros(1, config.context_dim)

            # Fixed-point iteration with training
            for iteration in range(max_iterations):
                optimizer.zero_grad()

                # Forward pass
                token_detached = token_embed.unsqueeze(0).detach()

                if hasattr(model, 'fnn_layers'):
                    # Layer-wise
                    context_temp = context.detach()
                    for layer_idx in range(model.num_layers):
                        fnn_input = torch.cat([token_detached, context_temp], dim=-1)
                        hidden = model.fnn_layers[layer_idx](fnn_input)

                        context_delta = torch.tanh(model.context_delta_projs[layer_idx](hidden))
                        forget = torch.sigmoid(model.forget_gates[layer_idx](hidden))
                        input_g = torch.sigmoid(model.input_gates[layer_idx](hidden))

                        context_temp = forget * context_temp + input_g * context_delta
                        context_temp = model.context_norms[layer_idx](context_temp)

                    context_new = context_temp
                else:
                    # Sequential
                    context_detached = context.detach()
                    fnn_input = torch.cat([token_detached, context_detached], dim=-1)
                    hidden = model.fnn(fnn_input)

                    context_delta = torch.tanh(model.context_delta_proj(hidden))
                    forget = torch.sigmoid(model.forget_gate(hidden))
                    input_g = torch.sigmoid(model.input_gate(hidden))

                    context_new = forget * context_detached + input_g * context_delta
                    context_new = model.context_norm(context_new)

                # Fixed-point loss
                loss = nn.functional.mse_loss(context_new, context.detach())

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Update context
                context = context_new.detach()
                epoch_loss += loss.item()

                # Check convergence (no warmup needed!)
                if loss.item() < config.phase1_convergence_threshold:
                    epoch_converged += 1
                    break

        avg_loss = epoch_loss / (len(token_ids) * max_iterations)
        converged_ratio = epoch_converged / len(token_ids)
        print(f"  Epoch {epoch+1}/{num_epochs} | Converged: {epoch_converged}/{len(token_ids)} ({converged_ratio:.1%}) | Avg Loss: {avg_loss:.6f}")

    # Final evaluation
    print("\nFinal Evaluation:")
    model.eval()
    with torch.no_grad():
        converged_count = 0
        total_iterations = 0

        for t, token_embed in enumerate(token_embeds):
            context = torch.zeros(1, config.context_dim)
            token_input = token_embed.unsqueeze(0)

            for iteration in range(max_iterations):
                if hasattr(model, 'fnn_layers'):
                    # Layer-wise
                    for layer_idx in range(model.num_layers):
                        fnn_input = torch.cat([token_input, context], dim=-1)
                        hidden = model.fnn_layers[layer_idx](fnn_input)

                        context_delta = torch.tanh(model.context_delta_projs[layer_idx](hidden))
                        forget = torch.sigmoid(model.forget_gates[layer_idx](hidden))
                        input_g = torch.sigmoid(model.input_gates[layer_idx](hidden))

                        context = forget * context + input_g * context_delta
                        context = model.context_norms[layer_idx](context)
                else:
                    # Sequential
                    fnn_input = torch.cat([token_input, context], dim=-1)
                    hidden = model.fnn(fnn_input)

                    context_delta = torch.tanh(model.context_delta_proj(hidden))
                    forget = torch.sigmoid(model.forget_gate(hidden))
                    input_g = torch.sigmoid(model.input_gate(hidden))

                    context_new = forget * context + input_g * context_delta
                    context_new = model.context_norm(context_new)

                    loss = nn.functional.mse_loss(context_new, context)
                    context = context_new

                if loss.item() < config.phase1_convergence_threshold:
                    converged_count += 1
                    total_iterations += (iteration + 1)
                    break

        avg_iters = total_iterations / converged_count if converged_count > 0 else max_iterations
        print(f"  Converged: {converged_count}/{len(token_ids)} ({converged_count/len(token_ids):.1%})")
        print(f"  Avg iterations: {avg_iters:.1f}")

        return converged_count, len(token_ids), avg_iters


def main():
    """Run 10-token training test for both architectures"""
    print("\n" + "="*60)
    print("10-Token Phase 1 Training Test")
    print("Sequential vs Layer-wise (WITH TRAINING)")
    print("="*60)

    # Test Sequential
    config_seq = Small2LayerSequentialConfig()
    config_seq.vocab_size = 50259
    config_seq.phase1_warmup_iterations = 0  # No warmup needed!
    model_seq = NewLLMSequential(config_seq)

    converged_seq, total_seq, avg_iters_seq = train_10tokens(model_seq, config_seq)

    # Test Layer-wise
    config_layer = Small2LayerLayerwiseConfig()
    config_layer.vocab_size = 50259
    config_layer.phase1_warmup_iterations = 0  # No warmup needed!
    model_layer = NewLLMLayerwise(config_layer)

    converged_layer, total_layer, avg_iters_layer = train_10tokens(model_layer, config_layer)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Sequential:")
    print(f"  Converged: {converged_seq}/{total_seq} ({converged_seq/total_seq:.1%})")
    print(f"  Avg iterations: {avg_iters_seq:.1f}")
    print(f"\nLayer-wise:")
    print(f"  Converged: {converged_layer}/{total_layer} ({converged_layer/total_layer:.1%})")
    print(f"  Avg iterations: {avg_iters_layer:.1f}")
    print("="*60)


if __name__ == "__main__":
    main()
