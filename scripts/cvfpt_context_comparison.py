#!/usr/bin/env python3
"""
CVFPT Context Vector Comparison Experiment

This script compares context vectors obtained through two different methods:
1. Fixed-point contexts: Context vectors after repetitive training (repeating tokens N times)
2. Single-pass contexts: Context vectors from a single forward pass (no repetition)

The experiment analyzes how repetition training affects the learned context representations.
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPT2Tokenizer
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.new_llm import NewLLM
from src.utils.config import NewLLMConfig


def load_trained_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"\n📥 Loading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config (already a NewLLMConfig object)
    config = checkpoint['config']

    # Detect actual updater strategy from state dict
    state_dict = checkpoint['model_state_dict']
    has_gates = any('forget_gate' in k or 'input_gate' in k for k in state_dict.keys())

    if has_gates:
        actual_strategy = 'gated'
    else:
        actual_strategy = 'simple'

    # Override config if mismatch
    if config.context_update_strategy != actual_strategy:
        print(f"⚠️  Config says '{config.context_update_strategy}' but checkpoint has '{actual_strategy}'")
        print(f"   Using actual strategy: {actual_strategy}")
        config.context_update_strategy = actual_strategy

    # Create and load model
    model = NewLLM(config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✓ Context updater: {actual_strategy}")
    return model, config


def get_fixed_point_context(model, token_id, num_repetitions=10, device='cpu'):
    """
    Get fixed-point context vector by repeating a token multiple times

    Args:
        model: Trained New-LLM model
        token_id: Single token ID
        num_repetitions: Number of repetitions
        device: Device to run on

    Returns:
        final_context: Context vector after num_repetitions [context_dim]
        trajectory: Full context trajectory [num_repetitions, context_dim]
    """
    model.eval()

    # Create input: repeat token N times
    input_ids = torch.tensor([[token_id] * num_repetitions], dtype=torch.long, device=device)

    with torch.no_grad():
        # Forward pass
        logits, context_trajectory = model(input_ids)

        # Get final context (after all repetitions)
        final_context = context_trajectory[0, -1, :]  # [context_dim]
        trajectory = context_trajectory[0, :, :]  # [num_repetitions, context_dim]

    return final_context, trajectory


def get_single_pass_context(model, token_id, device='cpu'):
    """
    Get context vector from a single forward pass (no repetition)

    Args:
        model: Trained New-LLM model
        token_id: Single token ID
        device: Device to run on

    Returns:
        context: Context vector after single pass [context_dim]
    """
    model.eval()

    # Create input: single token
    input_ids = torch.tensor([[token_id]], dtype=torch.long, device=device)

    with torch.no_grad():
        # Forward pass
        logits, context_trajectory = model(input_ids)

        # Get context after processing this single token
        context = context_trajectory[0, 0, :]  # [context_dim]

    return context


def compare_contexts(fixed_point_ctx, single_pass_ctx):
    """
    Compare two context vectors using multiple metrics

    Returns:
        dict with comparison metrics
    """
    # L2 distance
    l2_distance = torch.norm(fixed_point_ctx - single_pass_ctx).item()

    # Cosine similarity
    cosine_sim = F.cosine_similarity(
        fixed_point_ctx.unsqueeze(0),
        single_pass_ctx.unsqueeze(0),
        dim=1
    ).item()

    # Element-wise correlation
    correlation = np.corrcoef(
        fixed_point_ctx.cpu().numpy(),
        single_pass_ctx.cpu().numpy()
    )[0, 1]

    # L2 norms
    fixed_norm = torch.norm(fixed_point_ctx).item()
    single_norm = torch.norm(single_pass_ctx).item()

    return {
        'l2_distance': l2_distance,
        'cosine_similarity': cosine_sim,
        'correlation': correlation,
        'fixed_point_norm': fixed_norm,
        'single_pass_norm': single_norm,
    }


def analyze_convergence(trajectory):
    """
    Analyze how quickly context converges to fixed point

    Returns:
        convergence_steps: Number of steps to reach 95% convergence
        convergence_curve: L2 distance from final state at each step
    """
    final_context = trajectory[-1, :]  # Final state

    # Compute distance from final state at each step
    distances = []
    for t in range(len(trajectory)):
        dist = torch.norm(trajectory[t, :] - final_context).item()
        distances.append(dist)

    # Find when distance drops below 5% of initial distance
    initial_dist = distances[0] if len(distances) > 0 else 0
    threshold = 0.05 * initial_dist

    convergence_steps = len(distances)  # Default: never converges
    for t, dist in enumerate(distances):
        if dist < threshold:
            convergence_steps = t
            break

    return convergence_steps, distances


def run_experiment(model, tokenizer, num_tokens=100, num_repetitions=10, device='cpu'):
    """
    Run CVFPT comparison experiment on multiple tokens

    Args:
        model: Trained model
        tokenizer: Tokenizer
        num_tokens: Number of tokens to test
        num_repetitions: Repetitions for fixed-point training
        device: Device

    Returns:
        results: Dict with all experimental results
    """
    print(f"\n{'='*80}")
    print(f"🧪 Running CVFPT Context Comparison Experiment")
    print("=" * 80)
    print(f"📊 Testing {num_tokens} tokens")
    print(f"🔄 Fixed-point repetitions: {num_repetitions}")
    print(f"🖥️  Device: {device}")
    print("=" * 80)

    # Select random tokens from vocabulary
    vocab_size = len(tokenizer)
    # Avoid special tokens (0-100) and select from common tokens
    token_ids = np.random.randint(100, min(5000, vocab_size), size=num_tokens)

    results = {
        'token_ids': [],
        'tokens': [],
        'l2_distances': [],
        'cosine_similarities': [],
        'correlations': [],
        'fixed_point_norms': [],
        'single_pass_norms': [],
        'convergence_steps': [],
        'convergence_curves': [],
        'fixed_point_contexts': [],
        'single_pass_contexts': [],
    }

    print(f"\n🔬 Processing {num_tokens} tokens...")
    for token_id in tqdm(token_ids, desc="Analyzing tokens"):
        # Get token string
        token_str = tokenizer.decode([token_id])

        # Method 1: Fixed-point context (with repetition)
        fixed_ctx, trajectory = get_fixed_point_context(
            model, token_id, num_repetitions, device
        )

        # Method 2: Single-pass context (no repetition)
        single_ctx = get_single_pass_context(model, token_id, device)

        # Compare
        metrics = compare_contexts(fixed_ctx, single_ctx)

        # Analyze convergence
        conv_steps, conv_curve = analyze_convergence(trajectory)

        # Store results
        results['token_ids'].append(int(token_id))
        results['tokens'].append(token_str)
        results['l2_distances'].append(metrics['l2_distance'])
        results['cosine_similarities'].append(metrics['cosine_similarity'])
        results['correlations'].append(metrics['correlation'])
        results['fixed_point_norms'].append(metrics['fixed_point_norm'])
        results['single_pass_norms'].append(metrics['single_pass_norm'])
        results['convergence_steps'].append(conv_steps)
        results['convergence_curves'].append(conv_curve)
        results['fixed_point_contexts'].append(fixed_ctx.cpu().numpy())
        results['single_pass_contexts'].append(single_ctx.cpu().numpy())

    print(f"\n✓ Processed {num_tokens} tokens")

    return results


def generate_visualizations(results, output_dir):
    """Generate visualization plots"""
    print(f"\n📊 Generating visualizations...")

    os.makedirs(output_dir, exist_ok=True)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: L2 Distance Distribution
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(results['l2_distances'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('L2 Distance')
    ax1.set_ylabel('Frequency')
    ax1.set_title('L2 Distance: Fixed-Point vs Single-Pass')
    ax1.axvline(np.mean(results['l2_distances']), color='red', linestyle='--',
                label=f"Mean: {np.mean(results['l2_distances']):.3f}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Cosine Similarity Distribution
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(results['cosine_similarities'], bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Cosine Similarity')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Cosine Similarity: Fixed-Point vs Single-Pass')
    ax2.axvline(np.mean(results['cosine_similarities']), color='red', linestyle='--',
                label=f"Mean: {np.mean(results['cosine_similarities']):.3f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Correlation Distribution
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(results['correlations'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax3.set_xlabel('Correlation')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Correlation: Fixed-Point vs Single-Pass')
    ax3.axvline(np.mean(results['correlations']), color='red', linestyle='--',
                label=f"Mean: {np.mean(results['correlations']):.3f}")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Convergence Steps Distribution
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(results['convergence_steps'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax4.set_xlabel('Convergence Steps')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Steps to Reach Fixed Point (95% threshold)')
    ax4.axvline(np.mean(results['convergence_steps']), color='red', linestyle='--',
                label=f"Mean: {np.mean(results['convergence_steps']):.1f}")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Context Vector Norms Comparison
    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(results['fixed_point_norms'], results['single_pass_norms'], alpha=0.5)
    ax5.plot([0, max(results['fixed_point_norms'])],
             [0, max(results['fixed_point_norms'])], 'r--', label='y=x')
    ax5.set_xlabel('Fixed-Point Context Norm')
    ax5.set_ylabel('Single-Pass Context Norm')
    ax5.set_title('Context Vector Norms Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # Plot 6: Example Convergence Curves (first 10 tokens)
    ax6 = plt.subplot(3, 3, 6)
    for i in range(min(10, len(results['convergence_curves']))):
        ax6.plot(results['convergence_curves'][i], alpha=0.6)
    ax6.set_xlabel('Repetition Step')
    ax6.set_ylabel('L2 Distance from Final State')
    ax6.set_title('Convergence Curves (10 sample tokens)')
    ax6.grid(True, alpha=0.3)

    # Plot 7: L2 Distance vs Cosine Similarity
    ax7 = plt.subplot(3, 3, 7)
    scatter = ax7.scatter(results['l2_distances'], results['cosine_similarities'],
                          alpha=0.5, c=results['convergence_steps'], cmap='viridis')
    ax7.set_xlabel('L2 Distance')
    ax7.set_ylabel('Cosine Similarity')
    ax7.set_title('L2 Distance vs Cosine Similarity')
    plt.colorbar(scatter, ax=ax7, label='Convergence Steps')
    ax7.grid(True, alpha=0.3)

    # Plot 8: Context dimension analysis (first 50 dims of first token)
    ax8 = plt.subplot(3, 3, 8)
    first_fixed = results['fixed_point_contexts'][0][:50]
    first_single = results['single_pass_contexts'][0][:50]
    x = np.arange(len(first_fixed))
    ax8.plot(x, first_fixed, 'b-o', label='Fixed-Point', alpha=0.7, markersize=3)
    ax8.plot(x, first_single, 'r-s', label='Single-Pass', alpha=0.7, markersize=3)
    ax8.set_xlabel('Context Dimension')
    ax8.set_ylabel('Value')
    ax8.set_title(f'Context Vector Comparison (Token: "{results["tokens"][0]}")')
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    # Plot 9: Summary Statistics Table
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    stats_text = f"""
    Summary Statistics (N={len(results['token_ids'])})

    L2 Distance:
      Mean: {np.mean(results['l2_distances']):.3f}
      Std:  {np.std(results['l2_distances']):.3f}
      Min:  {np.min(results['l2_distances']):.3f}
      Max:  {np.max(results['l2_distances']):.3f}

    Cosine Similarity:
      Mean: {np.mean(results['cosine_similarities']):.3f}
      Std:  {np.std(results['cosine_similarities']):.3f}
      Min:  {np.min(results['cosine_similarities']):.3f}
      Max:  {np.max(results['cosine_similarities']):.3f}

    Convergence Steps:
      Mean: {np.mean(results['convergence_steps']):.1f}
      Std:  {np.std(results['convergence_steps']):.1f}
    """

    ax9.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center')

    plt.suptitle('CVFPT Context Vector Comparison Analysis',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    plot_path = f"{output_dir}/cvfpt_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {plot_path}")

    plt.close()


def generate_markdown_report(results, output_dir, num_repetitions):
    """Generate markdown report with analysis"""
    print(f"\n📝 Generating markdown report...")

    report_path = f"{output_dir}/CVFPT_COMPARISON_RESULTS.md"

    with open(report_path, 'w') as f:
        f.write("# CVFPT Context Vector Comparison Experiment\n\n")

        f.write("## Objective\n\n")
        f.write("Compare context vectors obtained through two different methods:\n\n")
        f.write("1. **Fixed-Point Contexts**: Context vectors after repetitive training ")
        f.write(f"(repeating each token {num_repetitions} times)\n")
        f.write("2. **Single-Pass Contexts**: Context vectors from a single forward pass ")
        f.write("(no repetition)\n\n")

        f.write("## Hypothesis\n\n")
        f.write("If CVFPT (Context Vector Fixed Point Training) is effective, we expect:\n\n")
        f.write("- **High similarity** between fixed-point and single-pass contexts\n")
        f.write("- **Fast convergence** to fixed points (few repetitions needed)\n")
        f.write("- **Consistent patterns** across different tokens\n\n")

        f.write("## Experimental Setup\n\n")
        f.write(f"- **Number of tokens tested**: {len(results['token_ids'])}\n")
        f.write(f"- **Repetitions for fixed-point**: {num_repetitions}\n")
        f.write(f"- **Context vector dimension**: {len(results['fixed_point_contexts'][0])}\n\n")

        f.write("## Results Summary\n\n")
        f.write("### Key Metrics\n\n")
        f.write("| Metric | Mean | Std | Min | Max |\n")
        f.write("|--------|------|-----|-----|-----|\n")

        metrics_to_report = [
            ('L2 Distance', results['l2_distances']),
            ('Cosine Similarity', results['cosine_similarities']),
            ('Correlation', results['correlations']),
            ('Convergence Steps', results['convergence_steps']),
            ('Fixed-Point Norm', results['fixed_point_norms']),
            ('Single-Pass Norm', results['single_pass_norms']),
        ]

        for name, values in metrics_to_report:
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            f.write(f"| {name} | {mean_val:.4f} | {std_val:.4f} | {min_val:.4f} | {max_val:.4f} |\n")

        f.write("\n### Interpretation\n\n")

        # L2 Distance analysis
        avg_l2 = np.mean(results['l2_distances'])
        f.write(f"**L2 Distance** (Average: {avg_l2:.3f})\n\n")
        if avg_l2 < 1.0:
            f.write("✅ **Very Low** - Fixed-point and single-pass contexts are nearly identical\n\n")
        elif avg_l2 < 3.0:
            f.write("⚠️ **Moderate** - Some difference between fixed-point and single-pass contexts\n\n")
        else:
            f.write("❌ **High** - Significant difference between the two methods\n\n")

        # Cosine similarity analysis
        avg_cos = np.mean(results['cosine_similarities'])
        f.write(f"**Cosine Similarity** (Average: {avg_cos:.3f})\n\n")
        if avg_cos > 0.95:
            f.write("✅ **Very High** - Context vectors point in nearly the same direction\n\n")
        elif avg_cos > 0.8:
            f.write("⚠️ **Moderate** - Context vectors are somewhat aligned\n\n")
        else:
            f.write("❌ **Low** - Context vectors differ significantly in direction\n\n")

        # Convergence analysis
        avg_conv = np.mean(results['convergence_steps'])
        f.write(f"**Convergence Steps** (Average: {avg_conv:.1f}/{num_repetitions})\n\n")
        if avg_conv < num_repetitions * 0.3:
            f.write("✅ **Fast Convergence** - Contexts reach fixed points quickly\n\n")
        elif avg_conv < num_repetitions * 0.7:
            f.write("⚠️ **Moderate Convergence** - Some tokens take longer to converge\n\n")
        else:
            f.write("❌ **Slow Convergence** - Many tokens do not reach fixed points\n\n")

        f.write("## Sample Tokens Analysis\n\n")
        f.write("Top 10 tokens with highest similarity:\n\n")

        # Sort by cosine similarity
        sorted_indices = np.argsort(results['cosine_similarities'])[::-1]

        f.write("| Rank | Token | L2 Dist | Cos Sim | Corr | Conv Steps |\n")
        f.write("|------|-------|---------|---------|------|------------|\n")

        for i, idx in enumerate(sorted_indices[:10], 1):
            token = results['tokens'][idx]
            l2 = results['l2_distances'][idx]
            cos = results['cosine_similarities'][idx]
            corr = results['correlations'][idx]
            conv = results['convergence_steps'][idx]
            # Escape pipe characters in token
            token_display = token.replace('|', '\\|')
            f.write(f"| {i} | `{token_display}` | {l2:.3f} | {cos:.3f} | {corr:.3f} | {conv} |\n")

        f.write("\n\nTop 10 tokens with lowest similarity:\n\n")
        f.write("| Rank | Token | L2 Dist | Cos Sim | Corr | Conv Steps |\n")
        f.write("|------|-------|---------|---------|------|------------|\n")

        for i, idx in enumerate(sorted_indices[-10:][::-1], 1):
            token = results['tokens'][idx]
            l2 = results['l2_distances'][idx]
            cos = results['cosine_similarities'][idx]
            corr = results['correlations'][idx]
            conv = results['convergence_steps'][idx]
            token_display = token.replace('|', '\\|')
            f.write(f"| {i} | `{token_display}` | {l2:.3f} | {cos:.3f} | {corr:.3f} | {conv} |\n")

        f.write("\n## Conclusions\n\n")

        # Generate conclusions based on metrics
        if avg_cos > 0.9 and avg_l2 < 2.0:
            f.write("✅ **CVFPT is Effective**\n\n")
            f.write("The high cosine similarity and low L2 distance indicate that:\n\n")
            f.write("- The model successfully learns fixed-point representations\n")
            f.write("- Repetitive training converges to stable context vectors\n")
            f.write("- Single-pass contexts are good approximations of fixed points\n")
        else:
            f.write("⚠️ **Mixed Results**\n\n")
            f.write("The metrics suggest that:\n\n")
            f.write("- Some tokens converge to stable fixed points\n")
            f.write("- Other tokens show significant variation\n")
            f.write("- Further training or architectural changes may be needed\n")

        f.write("\n## Visualizations\n\n")
        f.write("See `cvfpt_comparison.png` for detailed plots.\n\n")

        f.write("## Raw Data\n\n")
        f.write(f"Full experimental data saved to `cvfpt_comparison_data.npz`\n")

    print(f"✓ Report saved to {report_path}")


def save_raw_data(results, output_dir):
    """Save raw experimental data"""
    print(f"\n💾 Saving raw data...")

    data_path = f"{output_dir}/cvfpt_comparison_data.npz"

    np.savez(
        data_path,
        token_ids=np.array(results['token_ids']),
        tokens=np.array(results['tokens']),
        l2_distances=np.array(results['l2_distances']),
        cosine_similarities=np.array(results['cosine_similarities']),
        correlations=np.array(results['correlations']),
        fixed_point_norms=np.array(results['fixed_point_norms']),
        single_pass_norms=np.array(results['single_pass_norms']),
        convergence_steps=np.array(results['convergence_steps']),
        fixed_point_contexts=np.array(results['fixed_point_contexts']),
        single_pass_contexts=np.array(results['single_pass_contexts']),
    )

    print(f"✓ Raw data saved to {data_path}")


def main():
    parser = argparse.ArgumentParser(description='CVFPT Context Vector Comparison Experiment')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--num-tokens', type=int, default=100, help='Number of tokens to test')
    parser.add_argument('--num-repetitions', type=int, default=10, help='Repetitions for fixed-point training')
    parser.add_argument('--output-dir', type=str, default='./experiments/cvfpt_comparison', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')

    args = parser.parse_args()

    print("=" * 80)
    print("CVFPT Context Vector Comparison Experiment")
    print("=" * 80)

    # Setup
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    print("\n🔤 Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    print(f"✓ Tokenizer loaded: {len(tokenizer):,} tokens")

    # Load model
    model, config = load_trained_model(args.checkpoint, device)

    # Run experiment
    results = run_experiment(
        model=model,
        tokenizer=tokenizer,
        num_tokens=args.num_tokens,
        num_repetitions=args.num_repetitions,
        device=device
    )

    # Generate outputs
    generate_visualizations(results, args.output_dir)
    generate_markdown_report(results, args.output_dir, args.num_repetitions)
    save_raw_data(results, args.output_dir)

    print("\n" + "=" * 80)
    print("✅ Experiment Complete!")
    print("=" * 80)
    print(f"\n📁 Results saved to: {args.output_dir}/")
    print(f"  - CVFPT_COMPARISON_RESULTS.md  (Analysis report)")
    print(f"  - cvfpt_comparison.png         (Visualizations)")
    print(f"  - cvfpt_comparison_data.npz    (Raw data)")
    print()


if __name__ == "__main__":
    main()
