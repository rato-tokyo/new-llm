#!/usr/bin/env python3
"""
Compare context vectors between 1-layer and 2-layer New-LLM models

This script loads trained models and compares their context vector behavior:
1. Same input → different context vectors (1-layer vs 2-layer)
2. Fixed-point convergence comparison
3. Context vector representation quality
"""

import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.new_llm import NewLLM
from src.models.new_llm_2layer import NewLLM2Layer
from src.utils.config import NewLLMConfig


def load_model(checkpoint_path: str, model_class, device: torch.device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']

    model = model_class(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def extract_context_vectors(model, input_ids: torch.Tensor, device: torch.device):
    """Extract context vectors from model"""
    with torch.no_grad():
        input_ids = input_ids.to(device)
        _, context_trajectory = model(input_ids)

    return context_trajectory.cpu().numpy()


def analyze_differences(contexts_1layer, contexts_2layer, tokens):
    """Analyze differences between 1-layer and 2-layer context vectors

    Args:
        contexts_1layer: [num_tokens, context_dim]
        contexts_2layer: [num_tokens, context_dim]
        tokens: List of token strings

    Returns:
        metrics: Dictionary of analysis metrics
    """
    # L2 distance
    l2_distances = np.linalg.norm(contexts_1layer - contexts_2layer, axis=1)

    # Cosine similarity
    dot_products = np.sum(contexts_1layer * contexts_2layer, axis=1)
    norms_1layer = np.linalg.norm(contexts_1layer, axis=1)
    norms_2layer = np.linalg.norm(contexts_2layer, axis=1)
    cosine_similarities = dot_products / (norms_1layer * norms_2layer + 1e-8)

    # Angular difference
    angles_degrees = np.arccos(np.clip(cosine_similarities, -1, 1)) * 180 / np.pi

    # Magnitude analysis
    mag_diff = np.abs(norms_1layer - norms_2layer)

    # Per-dimension analysis
    dim_diffs = np.abs(contexts_1layer - contexts_2layer)
    mean_dim_diff = np.mean(dim_diffs, axis=0)

    # L2 decomposition
    mag_component = mag_diff
    perp_component = np.sqrt(np.maximum(0, l2_distances**2 - mag_component**2))

    mag_contribution = np.mean(mag_component**2) / (np.mean(l2_distances**2) + 1e-8)
    dir_contribution = np.mean(perp_component**2) / (np.mean(l2_distances**2) + 1e-8)

    metrics = {
        'tokens': tokens,
        'l2_distances': l2_distances,
        'cosine_similarities': cosine_similarities,
        'angles_degrees': angles_degrees,
        'norms_1layer': norms_1layer,
        'norms_2layer': norms_2layer,
        'mag_diff': mag_diff,
        'mean_dim_diff': mean_dim_diff,
        'mag_contribution': mag_contribution,
        'dir_contribution': dir_contribution,
        'contexts_1layer': contexts_1layer,
        'contexts_2layer': contexts_2layer,
    }

    return metrics


def print_analysis(metrics):
    """Print detailed analysis"""
    print("=" * 80)
    print("1-Layer vs 2-Layer Context Vector Analysis")
    print("=" * 80)

    # L2 Distance
    print("\n1. L2 Distance (1-layer vs 2-layer)")
    print("-" * 80)
    print(f"Mean:   {np.mean(metrics['l2_distances']):.4f}")
    print(f"Std:    {np.std(metrics['l2_distances']):.4f}")
    print(f"Min:    {np.min(metrics['l2_distances']):.4f}")
    print(f"Max:    {np.max(metrics['l2_distances']):.4f}")
    print(f"Median: {np.median(metrics['l2_distances']):.4f}")

    # Relative to norm
    relative_distance = np.mean(metrics['l2_distances']) / np.mean(metrics['norms_1layer'])
    print(f"\nRelative Distance (L2 / Norm): {relative_distance:.2%}")
    print(f"Interpretation: 2-layer differs from 1-layer by {relative_distance:.1%} of vector magnitude")

    # Cosine Similarity
    print("\n2. Cosine Similarity (Direction alignment)")
    print("-" * 80)
    print(f"Mean:   {np.mean(metrics['cosine_similarities']):.6f}")
    print(f"Std:    {np.std(metrics['cosine_similarities']):.6f}")
    print(f"Min:    {np.min(metrics['cosine_similarities']):.6f}")
    print(f"Max:    {np.max(metrics['cosine_similarities']):.6f}")

    # Angular difference
    print(f"\nAngular Difference:")
    print(f"Mean:   {np.mean(metrics['angles_degrees']):.2f}°")
    print(f"Std:    {np.std(metrics['angles_degrees']):.2f}°")
    print(f"Min:    {np.min(metrics['angles_degrees']):.2f}°")
    print(f"Max:    {np.max(metrics['angles_degrees']):.2f}°")

    # Magnitude Analysis
    print("\n3. Vector Magnitude Analysis")
    print("-" * 80)
    print(f"1-layer norm: {np.mean(metrics['norms_1layer']):.4f} ± {np.std(metrics['norms_1layer']):.4f}")
    print(f"2-layer norm: {np.mean(metrics['norms_2layer']):.4f} ± {np.std(metrics['norms_2layer']):.4f}")
    print(f"\nNorm difference:  {np.mean(metrics['mag_diff']):.4f} ± {np.std(metrics['mag_diff']):.4f}")
    print(f"Relative norm diff: {np.mean(metrics['mag_diff']) / np.mean(metrics['norms_1layer']):.4%}")

    # L2 Decomposition
    print("\n4. L2 Distance Decomposition")
    print("-" * 80)
    print(f"Magnitude contribution: {metrics['mag_contribution']:.1%}")
    print(f"Direction contribution: {metrics['dir_contribution']:.1%}")

    # Interpretation
    print("\n" + "=" * 80)
    print("PRACTICAL INTERPRETATION")
    print("=" * 80)
    print(f"""
L2 Distance = {np.mean(metrics['l2_distances']):.2f} (out of norm ~{np.mean(metrics['norms_1layer']):.2f})
  → 2-layer differs by {relative_distance:.1%} from 1-layer

Cosine Similarity = {np.mean(metrics['cosine_similarities']):.4f}
  → Vectors point in nearly same direction (angle ~{np.mean(metrics['angles_degrees']):.1f}°)

This means:
1. **Direction is {'very' if np.mean(metrics['cosine_similarities']) > 0.95 else 'moderately'} similar**: {np.mean(metrics['cosine_similarities'])*100:.1f}% aligned
2. **Magnitude is similar**: Within {np.mean(metrics['mag_diff']) / np.mean(metrics['norms_1layer']):.1%} of each other
3. **Main difference**: {'Direction' if metrics['dir_contribution'] > 0.5 else 'Magnitude'}

Conclusion:
{'2-layer captures similar representations as 1-layer with minor refinements.' if np.mean(metrics['cosine_similarities']) > 0.9 else '2-layer learns significantly different representations than 1-layer.'}
""")


def visualize(metrics, output_path='context_comparison_1layer_2layer.png'):
    """Create visualization of 1-layer vs 2-layer comparison"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot 1: L2 Distance distribution
    ax = axes[0, 0]
    ax.hist(metrics['l2_distances'], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(metrics['l2_distances']), color='red', linestyle='--',
               label=f'Mean: {np.mean(metrics["l2_distances"]):.2f}')
    ax.set_xlabel('L2 Distance')
    ax.set_ylabel('Frequency')
    ax.set_title('L2 Distance Distribution (1-layer vs 2-layer)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Cosine Similarity distribution
    ax = axes[0, 1]
    ax.hist(metrics['cosine_similarities'], bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(np.mean(metrics['cosine_similarities']), color='red', linestyle='--',
               label=f'Mean: {np.mean(metrics["cosine_similarities"]):.4f}')
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_title('Cosine Similarity Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Angular difference
    ax = axes[0, 2]
    ax.hist(metrics['angles_degrees'], bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(np.mean(metrics['angles_degrees']), color='red', linestyle='--',
               label=f'Mean: {np.mean(metrics["angles_degrees"]):.1f}°')
    ax.set_xlabel('Angular Difference (degrees)')
    ax.set_ylabel('Frequency')
    ax.set_title('Angular Difference Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Norm comparison
    ax = axes[1, 0]
    ax.scatter(metrics['norms_1layer'], metrics['norms_2layer'], alpha=0.5)
    min_norm = min(np.min(metrics['norms_1layer']), np.min(metrics['norms_2layer']))
    max_norm = max(np.max(metrics['norms_1layer']), np.max(metrics['norms_2layer']))
    ax.plot([min_norm, max_norm], [min_norm, max_norm], 'r--', label='y=x')
    ax.set_xlabel('1-Layer Norm')
    ax.set_ylabel('2-Layer Norm')
    ax.set_title('Norm Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 5: Per-dimension difference
    ax = axes[1, 1]
    ax.plot(metrics['mean_dim_diff'])
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Mean Absolute Difference')
    ax.set_title('Per-Dimension Difference')
    ax.grid(True, alpha=0.3)

    # Plot 6: L2 decomposition
    ax = axes[1, 2]
    contributions = [metrics['mag_contribution'] * 100, metrics['dir_contribution'] * 100]
    labels = ['Magnitude\nDifference', 'Direction\nDifference']
    colors = ['#ff9999', '#66b3ff']
    ax.bar(labels, contributions, color=colors, edgecolor='black')
    ax.set_ylabel('Contribution to L2 Distance (%)')
    ax.set_title('L2 Distance Decomposition')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare 1-layer and 2-layer context vectors")

    parser.add_argument('--checkpoint-1layer', type=str,
                       default='checkpoints/new_llm_repetition_final.pt',
                       help='1-layer model checkpoint')
    parser.add_argument('--checkpoint-2layer', type=str,
                       default='checkpoints/new_llm_2layer_repetition_final.pt',
                       help='2-layer model checkpoint')
    parser.add_argument('--num-tokens', type=int, default=100,
                       help='Number of tokens to analyze')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cuda/cpu)')
    parser.add_argument('--output-dir', type=str, default='experiments/layer_comparison',
                       help='Output directory')

    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    print("\n📚 Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Load models
    print("\n🔍 Loading 1-layer model...")
    model_1layer, config_1layer = load_model(args.checkpoint_1layer, NewLLM, device)
    print(f"✓ 1-layer model loaded")

    print("\n🔍 Loading 2-layer model...")
    model_2layer, config_2layer = load_model(args.checkpoint_2layer, NewLLM2Layer, device)
    print(f"✓ 2-layer model loaded")

    # Generate test tokens (use vocabulary tokens)
    print(f"\n🎲 Generating {args.num_tokens} test tokens...")
    np.random.seed(42)
    token_ids = np.random.choice(len(tokenizer), size=args.num_tokens, replace=False)
    tokens = [tokenizer.decode([tid]) for tid in token_ids]

    # Create input (repeat each token to reach fixed point)
    repetitions = 10
    input_ids_list = []
    for tid in token_ids:
        input_ids_list.append([tid] * repetitions)

    # Extract context vectors
    print("\n🔬 Extracting context vectors from 1-layer model...")
    contexts_1layer_list = []
    for input_ids in input_ids_list:
        input_tensor = torch.tensor([input_ids])
        contexts = extract_context_vectors(model_1layer, input_tensor, device)
        # Use final context (fixed point)
        contexts_1layer_list.append(contexts[0, -1, :])
    contexts_1layer = np.array(contexts_1layer_list)

    print("🔬 Extracting context vectors from 2-layer model...")
    contexts_2layer_list = []
    for input_ids in input_ids_list:
        input_tensor = torch.tensor([input_ids])
        contexts = extract_context_vectors(model_2layer, input_tensor, device)
        # Use final context (fixed point)
        contexts_2layer_list.append(contexts[0, -1, :])
    contexts_2layer = np.array(contexts_2layer_list)

    print(f"✓ Extracted context vectors: {contexts_1layer.shape}")

    # Analyze differences
    print("\n📊 Analyzing differences...")
    metrics = analyze_differences(contexts_1layer, contexts_2layer, tokens)

    # Print analysis
    print_analysis(metrics)

    # Visualize
    output_path = os.path.join(args.output_dir, 'context_comparison_1layer_2layer.png')
    visualize(metrics, output_path)

    # Save data
    data_path = os.path.join(args.output_dir, 'comparison_data.npz')
    np.savez(
        data_path,
        tokens=metrics['tokens'],
        l2_distances=metrics['l2_distances'],
        cosine_similarities=metrics['cosine_similarities'],
        angles_degrees=metrics['angles_degrees'],
        norms_1layer=metrics['norms_1layer'],
        norms_2layer=metrics['norms_2layer'],
        contexts_1layer=metrics['contexts_1layer'],
        contexts_2layer=metrics['contexts_2layer']
    )
    print(f"✓ Data saved to {data_path}")

    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
