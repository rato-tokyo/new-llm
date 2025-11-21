"""Test 19 tokens - find exact boundary"""

import torch
import torch.optim as optim
from src.models.new_llm_sequential import NewLLMSequential
from src.models.new_llm_layerwise import NewLLMLayerwise
from src.utils.dialogue_config import Small2LayerSequentialConfig, Small2LayerLayerwiseConfig


def train_model(model, config, num_tokens, num_epochs=3):
    """Train model"""
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

            for iteration in range(50):
                optimizer.zero_grad()
                token_detached = token_embed.unsqueeze(0).detach()
                context_detached = context.detach()

                context_new = model._update_context_one_step(token_detached, context_detached)

                loss = torch.nn.functional.mse_loss(context_new, context.detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                context = context_new.detach()

                if loss.item() < config.phase1_convergence_threshold:
                    break


def test_convergence(model, config, num_tokens, model_name):
    """Test convergence"""
    print(f"\nTesting: {model_name} with {num_tokens} tokens")

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
    avg_iters = num_iters.float().mean().item()
    convergence_rate = converged_count / num_tokens

    print(f"  Converged: {converged_count}/{num_tokens} ({convergence_rate:.1%})")
    print(f"  Avg iterations: {avg_iters:.1f}")

    return convergence_rate >= 0.95


def main():
    print("="*70)
    print("Test 19 Tokens - Finding Exact Boundary")
    print("="*70)

    # Sequential
    print("\n[1/2] Sequential with 19 tokens")
    config_seq = Small2LayerSequentialConfig()
    config_seq.vocab_size = 50259
    config_seq.phase1_warmup_iterations = 0
    model_seq = NewLLMSequential(config_seq)

    print("  Training...")
    train_model(model_seq, config_seq, num_tokens=19, num_epochs=3)
    seq_passed = test_convergence(model_seq, config_seq, 19, "Sequential (trained)")
    print(f"  Result: {'✅ PASS' if seq_passed else '❌ FAIL'}")

    # Layer-wise
    print("\n[2/2] Layer-wise with 19 tokens")
    config_layer = Small2LayerLayerwiseConfig()
    config_layer.vocab_size = 50259
    config_layer.phase1_warmup_iterations = 0
    model_layer = NewLLMLayerwise(config_layer)

    print("  Training...")
    train_model(model_layer, config_layer, num_tokens=19, num_epochs=3)
    layer_passed = test_convergence(model_layer, config_layer, 19, "Layer-wise (trained)")
    print(f"  Result: {'✅ PASS' if layer_passed else '❌ FAIL'}")

    print("\n" + "="*70)
    print("Summary:")
    print(f"  18 tokens: Both PASS")
    print(f"  19 tokens: Sequential {'PASS' if seq_passed else 'FAIL'}, Layer-wise {'PASS' if layer_passed else 'FAIL'}")
    print(f"  20 tokens: Both FAIL")
    print("="*70)


if __name__ == "__main__":
    main()
