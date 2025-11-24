"""
Debug script to analyze token embeddings and their fixed-point contexts
"""

import torch
import numpy as np
import random
from config import ResidualConfig
from src.models.new_llm_residual import NewLLMResidual
from src.data.loader import load_data
from src.training.phase1_trainer import Phase1Trainer

# Fix seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Configuration
config = ResidualConfig()
config.num_samples = 50
config.max_seq_length = 128

device = torch.device("cpu")

# Load data
train_token_ids, val_token_ids = load_data(config)

# Create and train model
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

# Create trainer and train
trainer = Phase1Trainer(
    model=model,
    max_iterations=config.phase1_max_iterations,
    convergence_threshold=config.phase1_convergence_threshold,
    min_converged_ratio=config.phase1_min_converged_ratio,
    learning_rate=config.phase1_learning_rate,
    dist_reg_weight=config.dist_reg_weight
)

print("\nTraining model...")
train_contexts = trainer.train(train_token_ids[:100], device, label="Debug")  # Use only 100 tokens for debug

print("\n" + "="*70)
print("TOKEN ANALYSIS - First 5 Tokens")
print("="*70)

# Analyze first 5 tokens
with torch.no_grad():
    # Get token embeddings
    first_5_tokens = train_token_ids[:5]
    token_embeds = model.token_embedding(first_5_tokens.unsqueeze(0).to(device))
    token_embeds = model.embed_norm(token_embeds).squeeze(0)

    # Get corresponding fixed-point contexts
    first_5_contexts = train_contexts[:5]

    print("\nToken Embeddings (after LayerNorm):")
    for i in range(5):
        token_id = first_5_tokens[i].item()
        embed = token_embeds[i]
        context = first_5_contexts[i]

        print(f"\nToken {i+1} (ID: {token_id}):")
        print(f"  Embedding norm: {torch.norm(embed).item():.6f}")
        print(f"  Embedding first 5 dims: {embed[:5].tolist()}")
        print(f"  Context norm: {torch.norm(context).item():.6f}")
        print(f"  Context first 5 dims: {context[:5].tolist()}")

        # Check if context is similar to embedding
        if embed.shape[0] == context.shape[0]:
            cosine_sim = torch.cosine_similarity(embed.unsqueeze(0), context.unsqueeze(0)).item()
            l2_distance = torch.norm(embed - context).item()
            print(f"  Embed vs Context - Cosine similarity: {cosine_sim:.6f}")
            print(f"  Embed vs Context - L2 distance: {l2_distance:.6f}")

    # Check pairwise distances between contexts
    print("\n" + "-"*70)
    print("Pairwise Context Distances (L2):")
    print("-"*70)
    for i in range(5):
        for j in range(i+1, 5):
            dist = torch.norm(first_5_contexts[i] - first_5_contexts[j]).item()
            print(f"  Context {i+1} <-> Context {j+1}: {dist:.6f}")

    # Check if all contexts are similar
    print("\n" + "-"*70)
    print("Context Statistics:")
    print("-"*70)
    all_5_norms = torch.norm(first_5_contexts, dim=1)
    print(f"  Norms: {all_5_norms.tolist()}")
    print(f"  Mean norm: {all_5_norms.mean().item():.6f}")
    print(f"  Std norm: {all_5_norms.std().item():.6f}")

    # Check if contexts are on a sphere
    mean_context = first_5_contexts.mean(dim=0)
    deviations = first_5_contexts - mean_context.unsqueeze(0)
    deviation_norms = torch.norm(deviations, dim=1)
    print(f"\n  Deviations from mean: {deviation_norms.tolist()}")
    print(f"  Mean deviation: {deviation_norms.mean().item():.6f}")

print("\n" + "="*70)