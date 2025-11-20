"""Test Phase 1 training with 1 token

Compare Sequential vs Layer-wise with proper training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from src.models.new_llm_sequential import NewLLMSequential
from src.models.new_llm_layerwise import NewLLMLayerwise
from src.utils.dialogue_config import Small2LayerSequentialConfig, Small2LayerLayerwiseConfig


def train_single_token(model, config, token_id=100, max_iterations=50, num_epochs=3):
    """Train model on single token to reach fixed point"""

    print(f"\n{'='*60}")
    print(f"Architecture: {config.architecture.upper()}")
    print(f"Token ID: {token_id}")
    print(f"Epochs: {num_epochs}, Max iterations: {max_iterations}")
    print(f"{'='*60}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.phase1_learning_rate)

    # Get token embedding
    model.train()
    token_ids = torch.tensor([[token_id]])
    token_embed = model.token_embedding(token_ids)
    if hasattr(model, 'embed_norm'):
        token_embed = model.embed_norm(token_embed)
    token_embed = token_embed.squeeze(0)  # [1, embed_dim]

    # Training epochs
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # Initialize context
        context = torch.zeros(1, config.context_dim)

        # Fixed-point iteration with training
        for iteration in range(max_iterations):
            optimizer.zero_grad()

            # Forward pass
            token_detached = token_embed.detach()

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
                print(f"  Epoch {epoch+1}/{num_epochs} | Converged at iteration {iteration+1} | Loss: {loss.item():.6f}")
                break

        if loss.item() >= config.phase1_convergence_threshold:
            avg_loss = epoch_loss / max_iterations
            print(f"  Epoch {epoch+1}/{num_epochs} | Did not converge | Avg Loss: {avg_loss:.6f} | Final: {loss.item():.6f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        context = torch.zeros(1, config.context_dim)

        for iteration in range(max_iterations):
            if hasattr(model, 'fnn_layers'):
                # Layer-wise
                for layer_idx in range(model.num_layers):
                    fnn_input = torch.cat([token_embed, context], dim=-1)
                    hidden = model.fnn_layers[layer_idx](fnn_input)

                    context_delta = torch.tanh(model.context_delta_projs[layer_idx](hidden))
                    forget = torch.sigmoid(model.forget_gates[layer_idx](hidden))
                    input_g = torch.sigmoid(model.input_gates[layer_idx](hidden))

                    context = forget * context + input_g * context_delta
                    context = model.context_norms[layer_idx](context)
            else:
                # Sequential
                fnn_input = torch.cat([token_embed, context], dim=-1)
                hidden = model.fnn(fnn_input)

                context_delta = torch.tanh(model.context_delta_proj(hidden))
                forget = torch.sigmoid(model.forget_gate(hidden))
                input_g = torch.sigmoid(model.input_gate(hidden))

                context_new = forget * context + input_g * context_delta
                context_new = model.context_norm(context_new)

                loss = nn.functional.mse_loss(context_new, context)
                context = context_new

            if loss.item() < config.phase1_convergence_threshold:
                print(f"\nFinal Evaluation: Converged at iteration {iteration+1} | Loss: {loss.item():.6f}")
                return True, iteration + 1, loss.item()

        print(f"\nFinal Evaluation: Did not converge | Loss: {loss.item():.6f}")
        return False, max_iterations, loss.item()


def main():
    """Run 1-token training test for both architectures"""
    print("\n" + "="*60)
    print("1-Token Phase 1 Training Test")
    print("Sequential vs Layer-wise (WITH TRAINING)")
    print("="*60)

    # Test Sequential
    config_seq = Small2LayerSequentialConfig()
    config_seq.vocab_size = 50259
    config_seq.phase1_warmup_iterations = 0  # No warmup needed!
    model_seq = NewLLMSequential(config_seq)

    converged_seq, iters_seq, loss_seq = train_single_token(model_seq, config_seq)

    # Test Layer-wise
    config_layer = Small2LayerLayerwiseConfig()
    config_layer.vocab_size = 50259
    config_layer.phase1_warmup_iterations = 0  # No warmup needed!
    model_layer = NewLLMLayerwise(config_layer)

    converged_layer, iters_layer, loss_layer = train_single_token(model_layer, config_layer)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Sequential:")
    print(f"  Converged: {converged_seq}")
    print(f"  Iterations: {iters_seq}")
    print(f"  Final Loss: {loss_seq:.6f}")
    print(f"\nLayer-wise:")
    print(f"  Converged: {converged_layer}")
    print(f"  Iterations: {iters_layer}")
    print(f"  Final Loss: {loss_layer:.6f}")
    print("="*60)


if __name__ == "__main__":
    main()
