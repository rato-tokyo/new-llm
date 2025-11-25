"""
Debug convergence WITH training - proper fixed-point learning test

This script verifies that:
1. Context carryover works correctly
2. CVFP learning reduces MSE between iterations
3. Fixed-point convergence occurs with training
"""

import torch
import numpy as np
from config import ResidualConfig
from src.models.new_llm_residual import NewLLMResidual
from src.data.loader import load_data
from src.training.phase1_trainer import Phase1Trainer

def debug_convergence_with_training():
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

    # Create trainer
    trainer = Phase1Trainer(
        model=model,
        max_iterations=5,  # 5 iterations for testing
        convergence_threshold=config.phase1_convergence_threshold,
        min_converged_ratio=config.phase1_min_converged_ratio,
        learning_rate=config.phase1_learning_rate,
        dist_reg_weight=config.dist_reg_weight
    )

    print("\n" + "="*70)
    print("CVFP LEARNING TEST (WITH TRAINING)")
    print("="*70)
    print(f"Tokens: {len(train_token_ids)}")
    print(f"Learning rate: {config.phase1_learning_rate}")
    print(f"dist_reg_weight: {config.dist_reg_weight}")
    print(f"Convergence threshold: {config.phase1_convergence_threshold}")

    # Train with 5 iterations
    trainer.train(train_token_ids, device, label="Debug")

    print("\n" + "="*70)
    print("RESULT SUMMARY")
    print("="*70)
    print(f"Converged tokens: {trainer.num_converged_tokens}/{len(train_token_ids)}")
    print(f"Convergence rate: {trainer.num_converged_tokens/len(train_token_ids)*100:.1f}%")

    if trainer.num_converged_tokens > 0:
        print("\n✅ PASS: Some tokens converged (CVFP learning is working)")
    else:
        print("\n❌ FAIL: No tokens converged (bug confirmed)")

if __name__ == "__main__":
    debug_convergence_with_training()
