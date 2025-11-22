"""Fixed-Point Analysis: Degenerate Solution Detection and Statistics

Analyzes fixed-point contexts to detect:
1. Global Attractor (all tokens converge to same point)
2. Zero Solution (all contexts near zero)
3. Fixed-point distribution statistics

Usage:
    python3 tests/phase2_experiments/analyze_fixed_points.py \
        --model-path <model_checkpoint.pth> \
        --token-ids <token_ids.pth> \
        --output <analysis_results.txt>
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/Users/sakajiritomoyoshi/Desktop/git/new-llm')

from src.models.new_llm_residual import NewLLMResidual


def print_flush(*args, **kwargs):
    """Print with immediate flush"""
    print(*args, **kwargs)
    sys.stdout.flush()


def compute_fixed_contexts(model, token_ids, max_iters=50, threshold=0.01, device='cpu'):
    """Compute fixed-point contexts"""
    model.to(device)
    model.eval()

    with torch.no_grad():
        token_embeds = model.token_embedding(token_ids.unsqueeze(0).to(device))
        token_embeds = model.embed_norm(token_embeds).squeeze(0)

    fixed_contexts = torch.zeros(len(token_ids), model.context_dim).to(device)
    converged_tokens = torch.zeros(len(token_ids), dtype=torch.bool)

    for iteration in range(max_iters):
        context = torch.zeros(1, model.context_dim).to(device)

        for t, token_embed in enumerate(token_embeds):
            with torch.no_grad():
                context = model._update_context_one_step(
                    token_embed.unsqueeze(0),
                    context
                )

                if iteration > 0:
                    loss = torch.nn.functional.mse_loss(context, fixed_contexts[t].unsqueeze(0))
                    if loss.item() < threshold:
                        converged_tokens[t] = True

                fixed_contexts[t] = context.squeeze(0)

        convergence_rate = converged_tokens.float().mean().item()
        if iteration > 0 and convergence_rate >= 0.995:
            break

    return fixed_contexts.detach(), converged_tokens


def analyze_global_attractor(contexts, sample_size=100):
    """Check if all tokens converge to same global attractor

    Returns:
        is_degenerate: True if global attractor detected
        stats: Dictionary of statistics
    """
    print_flush("\n" + "="*70)
    print_flush("1. GLOBAL ATTRACTOR DETECTION")
    print_flush("="*70)

    # Sample random pairs
    n_tokens = len(contexts)
    if n_tokens < 2:
        return False, {}

    sample_size = min(sample_size, n_tokens * (n_tokens - 1) // 2)

    # Randomly sample pairs
    indices = torch.randperm(n_tokens)[:min(sample_size * 2, n_tokens)]

    l2_distances = []
    cosine_sims = []

    for i in range(0, len(indices) - 1, 2):
        idx1, idx2 = indices[i], indices[i + 1]
        ctx1, ctx2 = contexts[idx1], contexts[idx2]

        # L2 distance
        l2_dist = torch.norm(ctx1 - ctx2).item()
        l2_distances.append(l2_dist)

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            ctx1.unsqueeze(0), ctx2.unsqueeze(0)
        ).item()
        cosine_sims.append(cos_sim)

    avg_l2 = np.mean(l2_distances)
    avg_cosine = np.mean(cosine_sims)

    # Detection thresholds
    DEGENERATE_L2_THRESHOLD = 0.01
    DEGENERATE_COSINE_THRESHOLD = 0.999

    is_degenerate = (avg_l2 < DEGENERATE_L2_THRESHOLD and
                     avg_cosine > DEGENERATE_COSINE_THRESHOLD)

    print_flush(f"  Sampled {len(l2_distances)} random token pairs")
    print_flush(f"  Average L2 Distance: {avg_l2:.6f}")
    print_flush(f"  Average Cosine Similarity: {avg_cosine:.6f}")
    print_flush(f"  L2 Distance Range: [{min(l2_distances):.6f}, {max(l2_distances):.6f}]")
    print_flush(f"  Cosine Similarity Range: [{min(cosine_sims):.6f}, {max(cosine_sims):.6f}]")

    if is_degenerate:
        print_flush(f"\n  ⚠️  DEGENERATE SOLUTION DETECTED: Global Attractor")
        print_flush(f"  All tokens converge to nearly identical vectors")
    else:
        print_flush(f"\n  ✅ Token-specific fixed points detected")

    stats = {
        'avg_l2_distance': avg_l2,
        'avg_cosine_similarity': avg_cosine,
        'min_l2': min(l2_distances),
        'max_l2': max(l2_distances),
        'is_degenerate': is_degenerate
    }

    return is_degenerate, stats


def analyze_zero_solution(contexts):
    """Check if contexts are near zero (zero solution)

    Returns:
        is_zero: True if zero solution detected
        stats: Dictionary of statistics
    """
    print_flush("\n" + "="*70)
    print_flush("2. ZERO SOLUTION DETECTION")
    print_flush("="*70)

    norms = torch.norm(contexts, dim=1)
    avg_norm = norms.mean().item()
    std_norm = norms.std().item()

    ZERO_THRESHOLD = 0.1
    is_zero = avg_norm < ZERO_THRESHOLD

    print_flush(f"  Average Context Norm: {avg_norm:.6f}")
    print_flush(f"  Std Dev of Norms: {std_norm:.6f}")
    print_flush(f"  Norm Range: [{norms.min().item():.6f}, {norms.max().item():.6f}]")

    if is_zero:
        print_flush(f"\n  ⚠️  DEGENERATE SOLUTION DETECTED: Zero Solution")
        print_flush(f"  All contexts are near zero vector")
    else:
        print_flush(f"\n  ✅ Non-zero contexts detected")

    stats = {
        'avg_norm': avg_norm,
        'std_norm': std_norm,
        'min_norm': norms.min().item(),
        'max_norm': norms.max().item(),
        'is_zero': is_zero
    }

    return is_zero, stats


def analyze_distribution(contexts):
    """Analyze fixed-point distribution statistics

    Returns:
        stats: Dictionary of distribution statistics
    """
    print_flush("\n" + "="*70)
    print_flush("3. FIXED-POINT DISTRIBUTION STATISTICS")
    print_flush("="*70)

    # Per-dimension statistics
    mean_per_dim = contexts.mean(dim=0)
    std_per_dim = contexts.std(dim=0)

    print_flush(f"\n  Per-dimension statistics:")
    print_flush(f"    Mean (across dims): {mean_per_dim.mean().item():.6f}")
    print_flush(f"    Std (across dims): {std_per_dim.mean().item():.6f}")
    print_flush(f"    Mean range: [{mean_per_dim.min().item():.6f}, {mean_per_dim.max().item():.6f}]")
    print_flush(f"    Std range: [{std_per_dim.min().item():.6f}, {std_per_dim.max().item():.6f}]")

    # Norms
    norms = torch.norm(contexts, dim=1)
    print_flush(f"\n  Context vector norms:")
    print_flush(f"    Mean: {norms.mean().item():.6f}")
    print_flush(f"    Median: {norms.median().item():.6f}")
    print_flush(f"    Std: {norms.std().item():.6f}")
    print_flush(f"    Range: [{norms.min().item():.6f}, {norms.max().item():.6f}]")

    # Pairwise distances (sample)
    sample_size = min(1000, len(contexts) * (len(contexts) - 1) // 2)
    indices = torch.randperm(len(contexts))[:min(sample_size * 2, len(contexts))]

    pairwise_dists = []
    for i in range(0, len(indices) - 1, 2):
        idx1, idx2 = indices[i], indices[i + 1]
        dist = torch.norm(contexts[idx1] - contexts[idx2]).item()
        pairwise_dists.append(dist)

    print_flush(f"\n  Pairwise L2 distances (sampled {len(pairwise_dists)} pairs):")
    print_flush(f"    Mean: {np.mean(pairwise_dists):.6f}")
    print_flush(f"    Median: {np.median(pairwise_dists):.6f}")
    print_flush(f"    Std: {np.std(pairwise_dists):.6f}")
    print_flush(f"    Range: [{min(pairwise_dists):.6f}, {max(pairwise_dists):.6f}]")

    # Sparsity (percentage of values near zero)
    SPARSITY_THRESHOLD = 0.01
    near_zero = (torch.abs(contexts) < SPARSITY_THRESHOLD).float().mean().item()
    print_flush(f"\n  Sparsity:")
    print_flush(f"    Percentage of values < {SPARSITY_THRESHOLD}: {near_zero * 100:.2f}%")

    stats = {
        'mean_per_dim_avg': mean_per_dim.mean().item(),
        'std_per_dim_avg': std_per_dim.mean().item(),
        'norm_mean': norms.mean().item(),
        'norm_median': norms.median().item(),
        'norm_std': norms.std().item(),
        'pairwise_dist_mean': np.mean(pairwise_dists),
        'pairwise_dist_median': np.median(pairwise_dists),
        'sparsity': near_zero
    }

    return stats


def analyze_entropy(contexts):
    """Analyze information content (entropy) of fixed points

    Returns:
        stats: Dictionary of entropy statistics
    """
    print_flush("\n" + "="*70)
    print_flush("4. INFORMATION CONTENT ANALYSIS")
    print_flush("="*70)

    # Effective rank (approximate entropy)
    # Higher rank = more information content
    U, S, V = torch.svd(contexts)

    # Normalized singular values
    S_norm = S / S.sum()

    # Entropy of singular values
    entropy = -(S_norm * torch.log(S_norm + 1e-10)).sum().item()

    # Effective rank (inverse participation ratio)
    effective_rank = 1.0 / (S_norm ** 2).sum().item()

    print_flush(f"  Singular value entropy: {entropy:.4f}")
    print_flush(f"  Effective rank: {effective_rank:.2f} / {len(S)} (max)")
    print_flush(f"  Top 5 singular values: {S[:5].tolist()}")

    # Low effective rank suggests low diversity (potential degenerate solution)
    if effective_rank < len(S) * 0.1:
        print_flush(f"\n  ⚠️  Low effective rank detected")
        print_flush(f"  Fixed points may lack diversity")
    else:
        print_flush(f"\n  ✅ Good diversity in fixed points")

    stats = {
        'sv_entropy': entropy,
        'effective_rank': effective_rank,
        'max_rank': len(S),
        'top_singular_values': S[:5].tolist()
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description='Analyze Fixed-Point Contexts')
    parser.add_argument('--model', type=str, required=True, help='Model architecture (residual or residual_context_only)')
    parser.add_argument('--vocab-size', type=int, default=50257)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--context-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--layer-structure', type=int, nargs='+', default=[1, 1, 1, 1])
    parser.add_argument('--output-layers', type=int, default=None)
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print_flush("="*70)
    print_flush("FIXED-POINT ANALYSIS")
    print_flush("="*70)
    print_flush(f"Model: {args.model}")
    print_flush(f"Context dim: {args.context_dim}")
    print_flush(f"Layer structure: {args.layer_structure}")

    # Load data
    print_flush(f"\nLoading {args.num_samples} samples from UltraChat...")
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    texts = []
    for i in range(args.num_samples):
        sample = dataset[i]
        for msg in sample['messages']:
            texts.append(msg['content'])

    token_ids = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids.extend(ids)

    token_ids = torch.tensor(token_ids, dtype=torch.long)
    print_flush(f"  Total tokens: {len(token_ids)}")

    # Create model
    if args.model == 'residual':
        model = NewLLMResidual(
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            context_dim=args.context_dim,
            hidden_dim=args.hidden_dim,
            layer_structure=args.layer_structure,
            dropout=0.1
        )
    elif args.model == 'residual_context_only':
        model = NewLLMResidualContextOnly(
            vocab_size=args.vocab_size,
            embed_dim=args.embed_dim,
            context_dim=args.context_dim,
            hidden_dim=args.context_dim,
            layer_structure=args.layer_structure,
            dropout=0.1,
            output_layers=args.output_layers or 1
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    print_flush(f"\nComputing fixed-point contexts...")
    contexts, converged = compute_fixed_contexts(model, token_ids, device=args.device)

    conv_rate = converged.float().mean().item()
    print_flush(f"  Converged: {converged.sum()}/{len(token_ids)} ({conv_rate:.1%})")

    # Run analyses
    is_global_attractor, global_stats = analyze_global_attractor(contexts)
    is_zero, zero_stats = analyze_zero_solution(contexts)
    dist_stats = analyze_distribution(contexts)
    entropy_stats = analyze_entropy(contexts)

    # Summary
    print_flush("\n" + "="*70)
    print_flush("SUMMARY")
    print_flush("="*70)

    if is_global_attractor or is_zero:
        print_flush("⚠️  DEGENERATE SOLUTION DETECTED")
        if is_global_attractor:
            print_flush("  - Global Attractor: All tokens → same vector")
        if is_zero:
            print_flush("  - Zero Solution: All contexts → zero")
        print_flush("\n  Recommended actions:")
        print_flush("  1. Check model initialization")
        print_flush("  2. Increase learning rate")
        print_flush("  3. Reduce regularization")
        print_flush("  4. Try different architecture")
    else:
        print_flush("✅ No degenerate solution detected")
        print_flush("  - Token-specific fixed points")
        print_flush("  - Non-zero contexts")
        print_flush("  - Good diversity")

    print_flush("\n" + "="*70)


if __name__ == '__main__':
    main()
