"""Compare baseline and new-llm performance"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt

from src.models.baseline_llm import BaselineLLM
from src.models.context_vector_llm import ContextVectorLLM
from src.utils.config import BaseConfig, NewLLMConfig


def load_model_checkpoint(model, checkpoint_path):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint.get('train_losses', []), checkpoint.get('val_losses', [])


def plot_comparison(baseline_train, baseline_val, newllm_train, newllm_val):
    """Plot training curves comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training loss
    axes[0].plot(baseline_train, label='Baseline', linewidth=2)
    axes[0].plot(newllm_train, label='New-LLM', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Validation loss
    axes[1].plot(baseline_val, label='Baseline', linewidth=2)
    axes[1].plot(newllm_val, label='New-LLM', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('checkpoints/model_comparison.png', dpi=150)
    print("\nSaved comparison plot to checkpoints/model_comparison.png")
    plt.show()


def main():
    print("="*60)
    print("Model Comparison: Baseline vs New-LLM")
    print("="*60)

    # Load models
    baseline_config = BaseConfig()
    newllm_config = NewLLMConfig()

    baseline_model = BaselineLLM(baseline_config)
    newllm_model = ContextVectorLLM(newllm_config)

    # Load checkpoints
    print("\nLoading checkpoints...")
    baseline_path = "checkpoints/best_baseline_llm.pt"
    newllm_path = "checkpoints/best_new_llm.pt"

    if not os.path.exists(baseline_path):
        print(f"❌ Baseline checkpoint not found: {baseline_path}")
        print("   Please run train_baseline.py first")
        return

    if not os.path.exists(newllm_path):
        print(f"❌ New-LLM checkpoint not found: {newllm_path}")
        print("   Please run train_new_llm.py first")
        return

    baseline_train, baseline_val = load_model_checkpoint(baseline_model, baseline_path)
    newllm_train, newllm_val = load_model_checkpoint(newllm_model, newllm_path)

    print("✓ Loaded both model checkpoints")

    # Count parameters
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    newllm_params = sum(p.numel() for p in newllm_model.parameters())

    print("\n" + "="*60)
    print("Model Statistics:")
    print("="*60)
    print(f"\nBaseline Model:")
    print(f"  Parameters: {baseline_params:,}")
    print(f"  Best train loss: {min(baseline_train):.4f}")
    print(f"  Best val loss: {min(baseline_val):.4f}")

    print(f"\nNew-LLM Model:")
    print(f"  Parameters: {newllm_params:,}")
    print(f"  Context vector dim: {newllm_config.context_vector_dim}")
    print(f"  Best train loss: {min(newllm_train):.4f}")
    print(f"  Best val loss: {min(newllm_val):.4f}")

    # Performance comparison
    print("\n" + "="*60)
    print("Performance Comparison:")
    print("="*60)

    baseline_final_val = baseline_val[-1] if baseline_val else float('inf')
    newllm_final_val = newllm_val[-1] if newllm_val else float('inf')

    print(f"\nFinal Validation Loss:")
    print(f"  Baseline: {baseline_final_val:.4f}")
    print(f"  New-LLM:  {newllm_final_val:.4f}")

    improvement = ((baseline_final_val - newllm_final_val) / baseline_final_val) * 100
    if improvement > 0:
        print(f"\n✓ New-LLM is {improvement:.2f}% better")
    else:
        print(f"\n✗ New-LLM is {abs(improvement):.2f}% worse")

    # Plot comparison
    if baseline_train and newllm_train:
        plot_comparison(baseline_train, baseline_val, newllm_train, newllm_val)

    print("\n" + "="*60)
    print("Analysis:")
    print("="*60)
    print("""
The comparison evaluates whether the context vector propagation
mechanism can learn meaningful representations through indirect
supervision (only optimizing token prediction loss).

Key questions:
1. Does New-LLM achieve comparable or better performance?
2. Are context vectors learning useful information?
3. Is the additive update mechanism effective?

If New-LLM performs well, it suggests that:
- Context information can be learned without direct supervision
- Additive propagation is viable for sequence modeling
- The architecture captures dependencies without attention
    """)

    print("="*60)


if __name__ == "__main__":
    main()
