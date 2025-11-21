"""Test trained models for Global Attractor Problem

Test both Sequential and Layer-wise trained models to ensure they
produce diverse context vectors (not degenerate solution).
"""

import torch
import torch.optim as optim
from src.models.new_llm_sequential import NewLLMSequential
from src.models.new_llm_layerwise import NewLLMLayerwise
from src.utils.dialogue_config import Small2LayerSequentialConfig, Small2LayerLayerwiseConfig


def train_model(model, config, num_tokens=20, max_iterations=50, num_epochs=3):
    """Train model on random tokens"""
    print(f"Training {config.architecture} model...")

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

    print(f"  Training complete ({num_epochs} epochs)")


def test_global_attractor(model, config, model_name):
    """Test if model has global attractor problem"""
    print(f"\n{'='*70}")
    print(f"Testing: {model_name}")
    print(f"{'='*70}")

    # Test 1: Different tokens should produce different contexts
    num_tokens = 20
    input_ids = torch.arange(num_tokens).unsqueeze(0)

    print(f"\nTEST 1: Different Tokens → Different Contexts")
    print(f"Input: {num_tokens} different tokens (0-19)")

    model.eval()
    with torch.no_grad():
        contexts, converged, num_iters = model.get_fixed_point_context(
            input_ids,
            max_iterations=50,
            tolerance=config.phase1_convergence_threshold,
            warmup_iterations=0
        )

    contexts = contexts.squeeze(0)  # [num_tokens, context_dim]

    # Compute pairwise distances
    distances = []
    cosine_sims = []

    for i in range(num_tokens):
        for j in range(i + 1, num_tokens):
            c1 = contexts[i]
            c2 = contexts[j]

            # L2 distance
            dist = torch.norm(c1 - c2).item()
            distances.append(dist)

            # Cosine similarity
            cos_sim = torch.dot(c1, c2) / (torch.norm(c1) * torch.norm(c2) + 1e-8)
            cosine_sims.append(cos_sim.item())

    avg_distance = sum(distances) / len(distances)
    min_distance = min(distances)
    max_distance = max(distances)
    avg_cosine = sum(cosine_sims) / len(cosine_sims)
    max_cosine = max(cosine_sims)

    # Get context norms
    norms = torch.norm(contexts, dim=-1)
    avg_norm = norms.mean().item()

    print(f"\nResults:")
    print(f"  Average L2 distance: {avg_distance:.6f}")
    print(f"  Min L2 distance: {min_distance:.6f}")
    print(f"  Max L2 distance: {max_distance:.6f}")
    print(f"  Average cosine similarity: {avg_cosine:.6f}")
    print(f"  Max cosine similarity: {max_cosine:.6f}")
    print(f"  Average context norm: {avg_norm:.4f}")

    # Convergence info
    converged_count = converged.sum().item()
    avg_iters = num_iters.float().mean().item()
    print(f"\nConvergence:")
    print(f"  Converged: {converged_count}/{num_tokens}")
    print(f"  Avg iterations: {avg_iters:.1f}")

    # Diagnosis
    print(f"\nDiagnosis:")

    has_problem = False

    if avg_distance < 0.001:
        print(f"  ❌ GLOBAL ATTRACTOR DETECTED!")
        print(f"     All tokens converge to same point (L2={avg_distance:.6f})")
        print(f"     This is a DEGENERATE SOLUTION")
        has_problem = True
    elif avg_distance < 0.1 and avg_cosine > 0.99:
        print(f"  ⚠️  SUSPICIOUS: Very similar contexts")
        print(f"     L2={avg_distance:.6f}, cosine={avg_cosine:.6f}")
        has_problem = True
    else:
        print(f"  ✅ HEALTHY: Different tokens have different contexts")
        print(f"     L2={avg_distance:.6f} (good diversity)")
        print(f"     Cosine={avg_cosine:.6f} (not identical)")

    # Test 2: Same token should produce consistent context
    print(f"\nTEST 2: Same Token → Consistent Context")
    token_id = 5
    num_repeats = 10
    input_ids_repeat = torch.full((1, num_repeats), token_id)

    print(f"Input: Token {token_id} repeated {num_repeats} times")

    with torch.no_grad():
        contexts_repeat, _, _ = model.get_fixed_point_context(
            input_ids_repeat,
            max_iterations=50,
            tolerance=config.phase1_convergence_threshold,
            warmup_iterations=0
        )

    contexts_repeat = contexts_repeat.squeeze(0)

    # Check consistency
    repeat_distances = []
    for i in range(num_repeats):
        for j in range(i + 1, num_repeats):
            dist = torch.norm(contexts_repeat[i] - contexts_repeat[j]).item()
            repeat_distances.append(dist)

    avg_repeat_dist = sum(repeat_distances) / len(repeat_distances) if repeat_distances else 0.0

    print(f"\nResults:")
    print(f"  Average L2 distance: {avg_repeat_dist:.6f}")

    print(f"\nDiagnosis:")
    if avg_repeat_dist < 0.0001:
        print(f"  ✅ EXPECTED: Same token → Same fixed point (dist={avg_repeat_dist:.6f})")
        print(f"     Fixed points are consistent")
    else:
        print(f"  ⚠️  UNEXPECTED: Same token → Different contexts (dist={avg_repeat_dist:.6f})")
        print(f"     Fixed points should be identical")
        has_problem = True

    # Summary
    print(f"\n{'='*70}")
    if has_problem:
        print(f"❌ {model_name}: FAILED - Degenerate solution detected")
    else:
        print(f"✅ {model_name}: PASSED - No global attractor problem")
    print(f"{'='*70}")

    return not has_problem


def main():
    """Test both architectures"""
    print("="*70)
    print("Global Attractor Test for Trained Models")
    print("="*70)

    # Test Sequential
    print("\n" + "="*70)
    print("SEQUENTIAL ARCHITECTURE")
    print("="*70)

    config_seq = Small2LayerSequentialConfig()
    config_seq.vocab_size = 50259
    config_seq.phase1_warmup_iterations = 0
    model_seq = NewLLMSequential(config_seq)

    train_model(model_seq, config_seq, num_tokens=20, num_epochs=3)
    seq_passed = test_global_attractor(model_seq, config_seq, "Sequential (trained)")

    # Test Layer-wise
    print("\n" + "="*70)
    print("LAYER-WISE ARCHITECTURE")
    print("="*70)

    config_layer = Small2LayerLayerwiseConfig()
    config_layer.vocab_size = 50259
    config_layer.phase1_warmup_iterations = 0
    model_layer = NewLLMLayerwise(config_layer)

    train_model(model_layer, config_layer, num_tokens=20, num_epochs=3)
    layer_passed = test_global_attractor(model_layer, config_layer, "Layer-wise (trained)")

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Sequential: {'✅ PASSED' if seq_passed else '❌ FAILED'}")
    print(f"Layer-wise: {'✅ PASSED' if layer_passed else '❌ FAILED'}")
    print("="*70)

    if seq_passed and layer_passed:
        print("\n✅ All tests passed! No global attractor problem detected.")
        return 0
    else:
        print("\n❌ Global attractor problem detected in one or more models!")
        return 1


if __name__ == "__main__":
    exit(main())
