"""Progressive Token Scaling Test

Test both Sequential and Layer-wise architectures with gradually increasing
token counts (20, 30, 40, ...) until one architecture fails to converge.
"""

import torch
import torch.optim as optim
from src.models.new_llm_sequential import NewLLMSequential
from src.models.new_llm_layerwise import NewLLMLayerwise
from src.utils.dialogue_config import Small2LayerSequentialConfig, Small2LayerLayerwiseConfig


def train_model(model, config, num_tokens=20, max_iterations=50, num_epochs=3):
    """Train model on random tokens"""
    print(f"  Training {config.architecture} model with {num_tokens} tokens...", flush=True)

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
                context_detached = context.detach()

                context_new = model._update_context_one_step(token_detached, context_detached)

                loss = torch.nn.functional.mse_loss(context_new, context.detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                context = context_new.detach()

                if loss.item() < config.phase1_convergence_threshold:
                    break

    print(f"  Training complete", flush=True)


def test_convergence(model, config, num_tokens, model_name):
    """Test if model converges for given number of tokens"""
    print(f"  Testing {model_name}...", flush=True)

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

    print(f"  Result: {converged_count}/{num_tokens} ({convergence_rate:.1%}), avg iters: {avg_iters:.1f}", flush=True)

    return convergence_rate >= 0.95  # Success if 95%+ converge


def main():
    """Progressive token scaling test"""
    print("="*70)
    print("Progressive Token Scaling Test")
    print("Sequential vs Layer-wise (WITH TRAINING)")
    print("="*70)

    # Test parameters - 100 token increments up to 2000
    token_counts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
                    1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    seq_active = True
    layer_active = True

    results = []

    for num_tokens in token_counts:
        print(f"\n{'='*70}", flush=True)
        print(f"Testing with {num_tokens} tokens", flush=True)
        print(f"{'='*70}", flush=True)

        seq_passed = False
        layer_passed = False

        # Test Sequential (if still active)
        if seq_active:
            print(f"\n[Sequential]", flush=True)
            config_seq = Small2LayerSequentialConfig()
            config_seq.vocab_size = 50259
            config_seq.phase1_warmup_iterations = 0
            model_seq = NewLLMSequential(config_seq)

            train_model(model_seq, config_seq, num_tokens=num_tokens, num_epochs=3)
            seq_passed = test_convergence(model_seq, config_seq, num_tokens, "Sequential")

            if not seq_passed:
                print(f"❌ Sequential FAILED at {num_tokens} tokens", flush=True)
                seq_active = False
            else:
                print(f"✅ Sequential PASSED", flush=True)

        # Test Layer-wise (if still active)
        if layer_active:
            print(f"\n[Layer-wise]", flush=True)
            config_layer = Small2LayerLayerwiseConfig()
            config_layer.vocab_size = 50259
            config_layer.phase1_warmup_iterations = 0
            model_layer = NewLLMLayerwise(config_layer)

            train_model(model_layer, config_layer, num_tokens=num_tokens, num_epochs=3)
            layer_passed = test_convergence(model_layer, config_layer, num_tokens, "Layer-wise")

            if not layer_passed:
                print(f"❌ Layer-wise FAILED at {num_tokens} tokens", flush=True)
                layer_active = False
            else:
                print(f"✅ Layer-wise PASSED", flush=True)

        results.append({
            'tokens': num_tokens,
            'seq_passed': seq_passed if seq_active else None,
            'layer_passed': layer_passed if layer_active else None
        })

        # Stop if both failed
        if not seq_active and not layer_active:
            print(f"\n{'='*70}")
            print("Both architectures failed. Stopping test.")
            print(f"{'='*70}")
            break

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Token Count':<12} {'Sequential':<15} {'Layer-wise':<15}")
    print("-" * 70)
    for r in results:
        seq_str = "✅ PASS" if r['seq_passed'] else ("❌ FAIL" if r['seq_passed'] is not None else "N/A")
        layer_str = "✅ PASS" if r['layer_passed'] else ("❌ FAIL" if r['layer_passed'] is not None else "N/A")
        print(f"{r['tokens']:<12} {seq_str:<15} {layer_str:<15}")

    print(f"\n{'='*70}")

    # Determine winner
    seq_max = max([r['tokens'] for r in results if r['seq_passed']], default=0)
    layer_max = max([r['tokens'] for r in results if r['layer_passed']], default=0)

    print(f"\nSequential: Maximum {seq_max} tokens")
    print(f"Layer-wise: Maximum {layer_max} tokens")

    if layer_max > seq_max:
        print(f"\n🏆 Winner: Layer-wise architecture")
    elif seq_max > layer_max:
        print(f"\n🏆 Winner: Sequential architecture")
    else:
        print(f"\n🤝 Tie: Both architectures reached {seq_max} tokens")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
