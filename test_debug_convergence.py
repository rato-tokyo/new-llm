"""Debug convergence test - check what's happening"""

import torch
import torch.optim as optim
from src.models.new_llm_layerwise import NewLLMLayerwise
from src.utils.dialogue_config import Small2LayerLayerwiseConfig


def train_model(model, config, num_tokens=10, num_epochs=3):
    """Train model"""
    optimizer = optim.Adam(model.parameters(), lr=config.phase1_learning_rate)
    token_ids = torch.arange(num_tokens)

    model.train()
    input_ids = torch.tensor([token_ids.tolist()])
    token_embeds = model.token_embedding(input_ids)
    if hasattr(model, 'embed_norm'):
        token_embeds = model.embed_norm(token_embeds)
    token_embeds = token_embeds.squeeze(0)

    print(f"Training with {num_tokens} tokens, {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_converged = 0

        for t, token_embed in enumerate(token_embeds):
            context = torch.zeros(1, config.context_dim)

            for iteration in range(50):
                optimizer.zero_grad()
                token_detached = token_embed.unsqueeze(0).detach()
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
                loss = torch.nn.functional.mse_loss(context_new, context.detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                context = context_new.detach()

                if loss.item() < config.phase1_convergence_threshold:
                    epoch_converged += 1
                    break

        print(f"  Epoch {epoch+1}: {epoch_converged}/{num_tokens} converged during training")


def test_convergence_detailed(model, config, num_tokens):
    """Test with detailed debugging"""
    print(f"\nTesting convergence with {num_tokens} tokens...")
    print(f"Tolerance: {config.phase1_convergence_threshold}")
    print(f"Warmup iterations: {config.phase1_warmup_iterations}")

    input_ids = torch.arange(num_tokens).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        # Manual implementation to debug
        batch_size = 1
        seq_len = num_tokens
        device = input_ids.device

        token_embeds = model.token_embedding(input_ids)
        token_embeds = model.embed_norm(token_embeds)

        converged_list = []
        iter_list = []
        final_deltas = []

        for t in range(seq_len):
            current_token = token_embeds[:, t, :]
            context = torch.zeros(batch_size, model.context_dim, device=device)

            converged_this = False

            for iteration in range(50):
                context_old = context.clone()

                for layer_idx in range(model.num_layers):
                    fnn_input = torch.cat([current_token, context], dim=-1)
                    hidden = model.fnn_layers[layer_idx](fnn_input)
                    context_delta = torch.tanh(model.context_delta_projs[layer_idx](hidden))
                    forget = torch.sigmoid(model.forget_gates[layer_idx](hidden))
                    input_g = torch.sigmoid(model.input_gates[layer_idx](hidden))
                    context = forget * context + input_g * context_delta
                    context = model.context_norms[layer_idx](context)

                if iteration >= config.phase1_warmup_iterations:
                    delta = torch.norm(context - context_old, dim=-1).item()

                    if delta < config.phase1_convergence_threshold:
                        converged_this = True
                        converged_list.append(True)
                        iter_list.append(iteration + 1)
                        final_deltas.append(delta)
                        break

            if not converged_this:
                converged_list.append(False)
                iter_list.append(50)
                final_deltas.append(delta)

        print(f"\nResults:")
        print(f"  Converged: {sum(converged_list)}/{num_tokens} ({sum(converged_list)/num_tokens*100:.1f}%)")
        print(f"  Avg iterations (converged only): {sum([iter_list[i] for i in range(num_tokens) if converged_list[i]]) / max(sum(converged_list), 1):.1f}")
        print(f"  Min delta: {min(final_deltas):.6f}")
        print(f"  Max delta: {max(final_deltas):.6f}")
        print(f"  Avg delta: {sum(final_deltas)/len(final_deltas):.6f}")

        print(f"\nFirst 5 tokens detail:")
        for i in range(min(5, num_tokens)):
            status = "✅" if converged_list[i] else "❌"
            print(f"  Token {i}: {status} | Iters: {iter_list[i]} | Delta: {final_deltas[i]:.6f}")


def main():
    print("="*70)
    print("Debug Convergence Test - Layer-wise")
    print("="*70)

    config = Small2LayerLayerwiseConfig()
    config.vocab_size = 50259
    config.phase1_warmup_iterations = 0

    model = NewLLMLayerwise(config)

    # Train
    train_model(model, config, num_tokens=10, num_epochs=3)

    # Test
    test_convergence_detailed(model, config, num_tokens=10)

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
