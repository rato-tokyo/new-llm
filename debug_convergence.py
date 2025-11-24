"""
Debug convergence issue - why is the change so large (32)?
"""

import torch
import numpy as np
from config import ResidualConfig
from src.models.new_llm_residual import NewLLMResidual
from src.data.loader import load_data

def debug_convergence():
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

    print(f"\nToken embedding stats:")
    print(f"  Shape: {token_embeds.shape}")
    print(f"  Norm mean: {token_embeds.norm(dim=1).mean().item():.4f}")
    print(f"  Norm std: {token_embeds.norm(dim=1).std().item():.4f}")

    # Initialize contexts (iteration 1)
    contexts_iter1 = []
    context = torch.zeros(1, config.context_dim, device=device)

    with torch.no_grad():
        for token_embed in token_embeds:
            context = model._update_context_one_step(
                token_embed.unsqueeze(0),
                context
            )
            contexts_iter1.append(context.clone())

    contexts_iter1 = torch.cat(contexts_iter1, dim=0)

    print(f"\nIteration 1 context stats:")
    print(f"  Shape: {contexts_iter1.shape}")
    print(f"  Norm mean: {contexts_iter1.norm(dim=1).mean().item():.4f}")
    print(f"  Norm std: {contexts_iter1.norm(dim=1).std().item():.4f}")
    print(f"  Min norm: {contexts_iter1.norm(dim=1).min().item():.4f}")
    print(f"  Max norm: {contexts_iter1.norm(dim=1).max().item():.4f}")

    # Get context from last token for iteration 2
    initial_context_iter2 = contexts_iter1[-1].unsqueeze(0)

    # Run iteration 2
    contexts_iter2 = []
    context = initial_context_iter2

    with torch.no_grad():
        for token_embed in token_embeds:
            context = model._update_context_one_step(
                token_embed.unsqueeze(0),
                context
            )
            contexts_iter2.append(context.clone())

    contexts_iter2 = torch.cat(contexts_iter2, dim=0)

    print(f"\nIteration 2 context stats:")
    print(f"  Norm mean: {contexts_iter2.norm(dim=1).mean().item():.4f}")
    print(f"  Norm std: {contexts_iter2.norm(dim=1).std().item():.4f}")

    # Calculate change between iterations
    change = (contexts_iter2 - contexts_iter1)
    mse_per_token = (change ** 2).mean(dim=1)

    print(f"\nChange between iterations:")
    print(f"  MSE per token - mean: {mse_per_token.mean().item():.4f}")
    print(f"  MSE per token - std: {mse_per_token.std().item():.4f}")
    print(f"  MSE per token - min: {mse_per_token.min().item():.4f}")
    print(f"  MSE per token - max: {mse_per_token.max().item():.4f}")

    # Average change (used in CVFP convergence check)
    avg_change = torch.norm(contexts_iter2 - contexts_iter1, p=2, dim=1).mean().item()
    print(f"\nAverage L2 change: {avg_change:.4f}")
    print(f"Convergence threshold: {config.phase1_convergence_threshold}")
    print(f"Ratio (change/threshold): {avg_change / config.phase1_convergence_threshold:.1f}x")

    # Check how many tokens would be considered converged
    converged = mse_per_token < config.phase1_convergence_threshold
    print(f"\nConverged tokens: {converged.sum().item()}/{len(mse_per_token)} ({converged.sum().item()/len(mse_per_token)*100:.1f}%)")

    # Check individual layer contributions
    print(f"\n--- Analyzing layer contributions ---")
    context = torch.zeros(1, config.context_dim, device=device)
    token_embed = token_embeds[0].unsqueeze(0)

    print(f"Initial context norm: {context.norm().item():.4f}")
    print(f"Token embed norm: {token_embed.norm().item():.4f}")

    for i, block in enumerate(model.blocks):
        context_before = context.clone()
        context, _ = block(token_embed, context)
        change = (context - context_before).norm().item()
        print(f"After block {i}: context norm = {context.norm().item():.4f}, change = {change:.4f}")

if __name__ == "__main__":
    debug_convergence()