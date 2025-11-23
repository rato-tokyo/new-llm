"""
Test refactored CVFP implementation with minimal parameters
"""

import os
import sys
import torch
import time

# Add project root to path (go up one level from tests/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.new_llm_residual import NewLLMResidual
from src.training.phase1 import train_phase1
from src.evaluation.diagnostics import check_identity_mapping, print_identity_mapping_warning


def print_flush(msg):
    print(msg)
    sys.stdout.flush()


class MinimalConfig:
    """Minimal configuration for quick testing"""
    # Model
    num_layers = 2
    context_dim = 16
    embed_dim = 16
    hidden_dim = 32
    vocab_size = 50257

    # Phase 1
    phase1_max_iterations = 5  # Reduced for quick test
    phase1_convergence_threshold = 0.02
    phase1_min_converged_ratio = 0.95
    phase1_lr_warmup = 0.002
    phase1_lr_medium = 0.0005
    phase1_lr_finetune = 0.0001

    # Distribution Regularization
    use_distribution_reg = True
    dist_reg_weight = 0.5  # Increased to 50% for stronger effect
    ema_momentum = 0.99    # EMA momentum for running statistics

    # Diagnostics
    identity_mapping_threshold = 0.95
    identity_check_samples = 50  # Reduced for quick test

    # Device
    device = "cpu"


def test_refactored_model():
    """Test the refactored model with minimal data"""

    config = MinimalConfig()
    device = torch.device(config.device)

    print_flush("="*70)
    print_flush("TESTING REFACTORED CVFP IMPLEMENTATION")
    print_flush("="*70)
    print_flush(f"\nConfiguration:")
    print_flush(f"  Layers: {config.num_layers}")
    print_flush(f"  Context dim: {config.context_dim}")
    print_flush(f"  Distribution reg weight: {config.dist_reg_weight}")
    print_flush(f"  Max iterations: {config.phase1_max_iterations}")

    # Create model with new architecture
    print_flush("\n1. Creating model...")
    layer_structure = [1] * config.num_layers
    model = NewLLMResidual(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        hidden_dim=config.hidden_dim,
        layer_structure=layer_structure,
        use_dist_reg=config.use_distribution_reg,
        ema_momentum=config.ema_momentum,
        layernorm_mix=0.0,  # Disabled
        enable_cvfp_learning=False  # Phase 1では無効（外部でCVFP損失計算）
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print_flush(f"   Total parameters: {total_params:,}")

    # Create minimal test data (10 tokens only)
    print_flush("\n2. Creating test data (10 tokens)...")
    test_token_ids = torch.randint(0, 1000, (10,), device=device)

    # Test Phase 1
    print_flush("\n3. Running Phase 1...")
    start_time = time.time()

    contexts = train_phase1(
        model=model,
        token_ids=test_token_ids,
        config=config,
        device=device,
        is_training=True,
        label="Test"
    )

    elapsed = time.time() - start_time
    print_flush(f"   Completed in {elapsed:.2f}s")

    # Analyze results
    print_flush("\n4. Analyzing results...")
    print_flush(f"   Context shape: {contexts.shape}")
    print_flush(f"   Mean: {contexts.mean().item():.4f}")
    print_flush(f"   Std: {contexts.std().item():.4f}")
    print_flush(f"   Min: {contexts.min().item():.4f}")
    print_flush(f"   Max: {contexts.max().item():.4f}")

    # Check running statistics
    print_flush("\n5. Checking running statistics...")
    for i, block in enumerate(model.blocks):
        for j, layer in enumerate(block.layers):
            if layer.use_dist_reg:
                print_flush(f"   Block {i}, Layer {j}:")
                print_flush(f"     Running mean: {layer.running_mean.mean().item():.4f}")
                print_flush(f"     Running var: {layer.running_var.mean().item():.4f}")
                print_flush(f"     Batches tracked: {layer.num_batches_tracked.item()}")

    # Check distribution loss
    dist_loss = model.get_distribution_loss()
    print_flush(f"\n6. Distribution loss: {dist_loss.item():.6f}")

    # Test identity mapping with new function
    print_flush("\n7. Identity mapping check...")
    identity_check = check_identity_mapping(
        model=model,
        context_dim=config.context_dim,
        device=device,
        num_samples=config.identity_check_samples,
        threshold=config.identity_mapping_threshold
    )
    is_identity = print_identity_mapping_warning(identity_check)

    # Test context diversity
    print_flush("8. Testing context diversity...")
    distances = torch.cdist(contexts, contexts, p=2)
    mask = ~torch.eye(len(contexts), dtype=bool, device=device)
    avg_distance = distances[mask].mean().item()
    print_flush(f"   Average pairwise distance: {avg_distance:.4f}")

    if avg_distance < 1.0:
        print_flush("   ⚠️ WARNING: Contexts are too similar!")
    else:
        print_flush("   ✅ Good: Contexts are diverse")

    print_flush("\n" + "="*70)
    print_flush("TEST COMPLETE")
    print_flush("="*70)

    # Summary
    print_flush("\n📊 Summary:")
    print_flush(f"   ✓ Model created successfully")
    print_flush(f"   ✓ Phase 1 completed in {elapsed:.2f}s")
    print_flush(f"   ✓ Distribution loss: {dist_loss.item():.6f}")
    print_flush(f"   {'⚠️' if is_identity else '✓'} Identity mapping: {identity_check['avg_similarity']:.4f}")
    print_flush(f"   ✓ Context diversity: {avg_distance:.4f}")

    return {
        'contexts': contexts,
        'dist_loss': dist_loss.item(),
        'identity_check': identity_check,
        'diversity': avg_distance
    }


if __name__ == "__main__":
    results = test_refactored_model()
