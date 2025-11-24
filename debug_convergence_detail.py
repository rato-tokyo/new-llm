"""
Debug convergence detail - check actual MSE values between iterations
"""

import torch
import numpy as np
from config import ResidualConfig
from src.models.new_llm_residual import NewLLMResidual
from src.data.loader import load_data

def debug_convergence_detail():
    config = ResidualConfig()
    device = torch.device("cpu")

    # Load data
    print("Loading data...")
    train_token_ids, _ = load_data(config)

    # Use only first 100 tokens for debugging
    train_token_ids = train_token_ids[:100]

    # Create model
    layer_structure = [1] * config.num_layers
    model = NewLLMResidual(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        hidden_dim=config.hidden_dim,
        layer_structure=layer_structure,
        layernorm_mix=1.0,
        use_pretrained_embeddings=True
    )
    model.to(device)
    model.eval()

    # Get token embeddings
    token_embeds = model.token_embedding(train_token_ids.unsqueeze(0).to(device))
    token_embeds = model.embed_norm(token_embeds).squeeze(0)

    # Track contexts for 3 iterations
    all_contexts = []

    for iter_num in range(3):
        contexts = []
        if iter_num == 0:
            context = torch.zeros(1, config.context_dim, device=device)
        else:
            # Use last context from previous iteration
            context = all_contexts[-1][-1].unsqueeze(0)

        with torch.no_grad():
            for token_embed in token_embeds:
                context = model._update_context_one_step(
                    token_embed.unsqueeze(0),
                    context
                )
                contexts.append(context.clone())

        contexts = torch.cat(contexts, dim=0)
        all_contexts.append(contexts)

    # Calculate MSE between iterations
    print("\n" + "="*70)
    print("MSE Analysis (Per-Token)")
    print("="*70)

    # Iter 1 -> 2
    mse_1_2 = ((all_contexts[1] - all_contexts[0]) ** 2).mean(dim=1)
    print(f"\nIteration 1 → 2:")
    print(f"  MSE Mean: {mse_1_2.mean().item():.6f}")
    print(f"  MSE Std:  {mse_1_2.std().item():.6f}")
    print(f"  MSE Min:  {mse_1_2.min().item():.6f}")
    print(f"  MSE Max:  {mse_1_2.max().item():.6f}")

    # Check convergence with different thresholds
    for threshold in [0.05, 0.1, 0.5, 1.0, 2.0]:
        converged = (mse_1_2 < threshold).sum().item()
        print(f"  Converged with threshold {threshold:4.2f}: {converged}/{len(mse_1_2)} ({converged/len(mse_1_2)*100:.1f}%)")

    # Iter 2 -> 3
    mse_2_3 = ((all_contexts[2] - all_contexts[1]) ** 2).mean(dim=1)
    print(f"\nIteration 2 → 3:")
    print(f"  MSE Mean: {mse_2_3.mean().item():.6f}")
    print(f"  MSE Std:  {mse_2_3.std().item():.6f}")
    print(f"  MSE Min:  {mse_2_3.min().item():.6f}")
    print(f"  MSE Max:  {mse_2_3.max().item():.6f}")

    for threshold in [0.05, 0.1, 0.5, 1.0, 2.0]:
        converged = (mse_2_3 < threshold).sum().item()
        print(f"  Converged with threshold {threshold:4.2f}: {converged}/{len(mse_2_3)} ({converged/len(mse_2_3)*100:.1f}%)")

    # Show actual convergence behavior
    print("\n" + "="*70)
    print("Context Change Analysis")
    print("="*70)

    # Check if contexts are actually changing
    print("\nContext norms:")
    for i, contexts in enumerate(all_contexts):
        norm = contexts.norm(dim=1).mean().item()
        print(f"  Iteration {i+1}: {norm:.4f}")

    # Check if model is learning (contexts should be different from zero init)
    print("\nDistance from zero initialization:")
    zero_context = torch.zeros_like(all_contexts[0])
    for i, contexts in enumerate(all_contexts):
        dist = (contexts - zero_context).norm(dim=1).mean().item()
        print(f"  Iteration {i+1}: {dist:.4f}")

if __name__ == "__main__":
    debug_convergence_detail()