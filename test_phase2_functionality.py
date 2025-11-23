"""
Test Phase 2 Multi-Output Functionality

Quick test to verify Phase 2 architecture and training work correctly.
NOT testing accuracy - only testing that the code runs without errors.
"""

import torch
import sys
sys.path.insert(0, '.')

from config import ResidualConfig
from src.models.new_llm_residual import NewLLMResidual
from src.models.new_llm_phase2 import expand_to_phase2
from src.training.phase2_multioutput import train_phase2_multioutput

def test_phase2_functionality():
    """Test Phase 2 with minimal data - functionality check only"""

    print("="*70)
    print("Phase 2 Multi-Output Functionality Test")
    print("="*70)
    print()

    config = ResidualConfig()
    device = torch.device(config.device)

    # Create Phase 1 model (simulating trained model)
    print("Step 1: Creating Phase 1 model...")
    layer_structure = [1] * config.num_layers
    phase1_model = NewLLMResidual(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        hidden_dim=config.hidden_dim,
        layer_structure=layer_structure,
        layernorm_mix=1.0,
        use_pretrained_embeddings=config.use_pretrained_embeddings
    )
    phase1_model.to(device)
    print(f"✓ Phase 1 model created")
    print()

    # Expand to Phase 2
    print("Step 2: Expanding to Phase 2 (multi-output)...")
    phase2_model = expand_to_phase2(phase1_model)
    phase2_model.to(device)
    print(f"✓ Phase 2 model created")
    print()

    # Test forward pass
    print("Step 3: Testing forward pass...")
    test_input = torch.randint(0, 1000, (2, 10), device=device)  # [batch=2, seq=10]
    with torch.no_grad():
        block_logits = phase2_model(test_input, return_all_logits=True)

    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {block_logits.shape}")
    print(f"  Expected: [num_blocks=6, batch=2, seq=10, vocab={config.vocab_size}]")

    assert block_logits.shape == (6, 2, 10, config.vocab_size), "Incorrect output shape!"
    print(f"✓ Forward pass works correctly")
    print()

    # Test training (1 epoch, tiny data)
    print("Step 4: Testing training loop (1 epoch, 50 tokens)...")
    config.phase2_epochs = 1
    config.skip_phase2 = False

    # Create tiny dataset
    train_tokens = torch.randint(0, 1000, (50,), device=device)
    val_tokens = torch.randint(0, 1000, (30,), device=device)

    # Run training
    try:
        train_phase2_multioutput(
            phase2_model=phase2_model,
            train_token_ids=train_tokens,
            val_token_ids=val_tokens,
            config=config,
            device=device
        )
        print(f"✓ Training loop completed successfully")
    except Exception as e:
        print(f"✗ Training failed: {e}")
        raise
    print()

    # Test inference
    print("Step 5: Testing inference...")
    phase2_model.eval()
    with torch.no_grad():
        test_input = torch.randint(0, 1000, (1, 5), device=device)
        final_logits = phase2_model(test_input, return_all_logits=False)

    print(f"  Input shape: {test_input.shape}")
    print(f"  Final block output: {final_logits.shape}")
    assert final_logits.shape == (1, 5, config.vocab_size), "Incorrect inference shape!"
    print(f"✓ Inference works correctly")
    print()

    print("="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)
    print()
    print("Phase 2 multi-output architecture is working correctly!")
    print("Ready for actual training with real data.")

if __name__ == "__main__":
    test_phase2_functionality()
