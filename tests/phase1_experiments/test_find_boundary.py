"""Find Token Count Boundary

Binary search to find the exact token count where architectures start to fail.
We know: 10 tokens = success, 20 tokens = failure
"""

import torch
import torch.optim as optim
from src.models.new_llm_sequential import NewLLMSequential
from src.models.new_llm_layerwise import NewLLMLayerwise
from src.utils.dialogue_config import Small2LayerSequentialConfig, Small2LayerLayerwiseConfig


def train_model(model, config, num_tokens, max_iterations=50, num_epochs=3):
    """Train model on random tokens"""
    optimizer = optim.Adam(model.parameters(), lr=config.phase1_learning_rate)
    token_ids = torch.arange(num_tokens)

    model.train()
    input_ids = torch.tensor([token_ids.tolist()])
    token_embeds = model.token_embedding(input_ids)
    if hasattr(model, 'embed_norm'):
        token_embeds = model.embed_norm(token_embeds)
    token_embeds = token_embeds.squeeze(0)

    for epoch in range(num_epochs):
        for t, token_embed in enumerate(token_embeds):
            context = torch.zeros(1, config.context_dim)

            for iteration in range(max_iterations):
                optimizer.zero_grad()
                token_detached = token_embed.unsqueeze(0).detach()

                if hasattr(model, 'fnn_layers'):
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
                    context_detached = context.detach()
                    fnn_input = torch.cat([token_detached, context_detached], dim=-1)
                    hidden = model.fnn(fnn_input)

                    context_delta = torch.tanh(model.context_delta_proj(hidden))
                    forget = torch.sigmoid(model.forget_gate(hidden))
                    input_g = torch.sigmoid(model.input_gate(hidden))

                    context_new = forget * context_detached + input_g * context_delta
                    context_new = model.context_norm(context_new)

                loss = torch.nn.functional.mse_loss(context_new, context.detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                context = context_new.detach()

                if loss.item() < config.phase1_convergence_threshold:
                    break


def test_convergence(model, config, num_tokens):
    """Test if model converges for given number of tokens"""
    input_ids = torch.arange(num_tokens).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        contexts, converged, num_iters = model.get_fixed_point_context(
            input_ids,
            max_iterations=50,
            tolerance=config.phase1_convergence_threshold,
            warmup_iterations=0
        )

    converged_count = converged.sum().item()
    convergence_rate = converged_count / num_tokens
    return convergence_rate >= 0.95  # Success if 95%+ converge


def main():
    """Find boundary for both architectures"""
    print("="*70)
    print("Token Count Boundary Search")
    print("Sequential vs Layer-wise (WITH TRAINING)")
    print("="*70)

    # Test specific token counts: 12, 14, 16, 18
    token_counts = [12, 14, 16, 18]

    results_seq = []
    results_layer = []

    for num_tokens in token_counts:
        print(f"\n{'='*70}")
        print(f"Testing with {num_tokens} tokens")
        print(f"{'='*70}")

        # Test Sequential
        print(f"\n[1/2] Sequential with {num_tokens} tokens")
        config_seq = Small2LayerSequentialConfig()
        config_seq.vocab_size = 50259
        config_seq.phase1_warmup_iterations = 0
        model_seq = NewLLMSequential(config_seq)

        print(f"  Training...")
        train_model(model_seq, config_seq, num_tokens=num_tokens, num_epochs=3)
        print(f"  Testing...")
        seq_passed = test_convergence(model_seq, config_seq, num_tokens)
        results_seq.append((num_tokens, seq_passed))
        print(f"  Result: {'✅ PASS' if seq_passed else '❌ FAIL'}")

        # Test Layer-wise
        print(f"\n[2/2] Layer-wise with {num_tokens} tokens")
        config_layer = Small2LayerLayerwiseConfig()
        config_layer.vocab_size = 50259
        config_layer.phase1_warmup_iterations = 0
        model_layer = NewLLMLayerwise(config_layer)

        print(f"  Training...")
        train_model(model_layer, config_layer, num_tokens=num_tokens, num_epochs=3)
        print(f"  Testing...")
        layer_passed = test_convergence(model_layer, config_layer, num_tokens)
        results_layer.append((num_tokens, layer_passed))
        print(f"  Result: {'✅ PASS' if layer_passed else '❌ FAIL'}")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Token Count':<15} {'Sequential':<15} {'Layer-wise':<15}")
    print("-" * 70)
    print(f"{'10 (known)':<15} {'✅ PASS':<15} {'✅ PASS':<15}")
    for i in range(len(token_counts)):
        tokens = token_counts[i]
        seq_str = "✅ PASS" if results_seq[i][1] else "❌ FAIL"
        layer_str = "✅ PASS" if results_layer[i][1] else "❌ FAIL"
        print(f"{tokens:<15} {seq_str:<15} {layer_str:<15}")
    print(f"{'20 (known)':<15} {'❌ FAIL':<15} {'❌ FAIL':<15}")

    print(f"\n{'='*70}")

    # Find boundaries
    seq_boundary = None
    layer_boundary = None

    for tokens, passed in results_seq:
        if not passed:
            seq_boundary = tokens
            break

    for tokens, passed in results_layer:
        if not passed:
            layer_boundary = tokens
            break

    print(f"\nSequential: Fails at {seq_boundary} tokens" if seq_boundary else "\nSequential: Passes all tested token counts")
    print(f"Layer-wise: Fails at {layer_boundary} tokens" if layer_boundary else "Layer-wise: Passes all tested token counts")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
