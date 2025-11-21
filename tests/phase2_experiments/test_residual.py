"""
Test Residual Standard Architecture with CVFP

Tests Residual Standard architecture with configurable layer structure.
Supports Phase 1 (fixed-point context learning) and Phase 2 (token prediction).
"""

import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset

# Add project root to path
sys.path.insert(0, '/Users/sakajiritomoyoshi/Desktop/git/new-llm')

from transformers import AutoTokenizer
from src.models.new_llm_residual import NewLLMResidual
from src.utils.early_stopping import Phase1EarlyStopping, Phase2EarlyStopping


def print_flush(*args, **kwargs):
    """Print with immediate flush to ensure real-time output"""
    print(*args, **kwargs)
    sys.stdout.flush()


def load_ultrachat_samples(num_samples=10):
    """Load UltraChat samples"""
    print_flush(f"Loading {num_samples} samples from UltraChat...")

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    samples = dataset.select(range(num_samples))

    texts = []
    for sample in samples:
        for msg in sample['messages']:
            texts.append(msg['content'])

    return texts


def tokenize_texts(texts):
    """Tokenize texts using GPT-2 tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    token_ids = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids.extend(ids)

    return torch.tensor(token_ids, dtype=torch.long), tokenizer


def phase1_train(model, token_ids, max_iters=50, threshold=0.01, device='cpu'):
    """Phase 1: Fixed-point context learning with CVFP (Context Vector Fixed-Point Property)

    CVFP Principle:
    - Iteration 1: Forward pass only, save contexts
    - Iteration 2+: Learn so that context[t] matches previous context[t-period]
    - Example: "red apple" → "red apple red" should output same context as "red"
    """
    print_flush("\n" + "="*70)
    print_flush("PHASE 1: Fixed-Point Context Learning (CVFP) - Train")
    print_flush("="*70)

    model.to(device)
    model.train()

    # Optimizer for Phase 1 - ONLY train context generation layers (exclude token_output)
    context_params = [p for name, p in model.named_parameters() if 'token_output' not in name]
    optimizer = torch.optim.Adam(context_params, lr=0.0001)

    # Early stopping - require near-perfect convergence (99.5%)
    early_stopping = Phase1EarlyStopping(
        convergence_threshold=0.995,  # 99.5% convergence required
        patience=999,                  # Effectively disable patience-based stopping
        min_delta=0.0                  # No min_delta check (convergence-only)
    )

    # Token embeddings
    with torch.no_grad():
        token_embeds = model.token_embedding(token_ids.unsqueeze(0).to(device))
        token_embeds = model.embed_norm(token_embeds).squeeze(0)

    # Fixed contexts storage
    fixed_contexts = torch.zeros(len(token_ids), model.context_dim).to(device)

    # Train until convergence
    converged_tokens = torch.zeros(len(token_ids), dtype=torch.bool)

    for iteration in range(max_iters):
        total_loss_value = 0

        # Process entire sequence with context carry-over
        context = torch.zeros(1, model.context_dim).to(device)

        for t, token_embed in enumerate(token_embeds):
            # Zero gradients before each token
            if iteration > 0:
                optimizer.zero_grad()

            # Update context one step
            context = model._update_context_one_step(
                token_embed.unsqueeze(0),
                context
            )

            # Loss: Match previous iteration's context (CVFP learning)
            if iteration > 0:
                loss = torch.nn.functional.mse_loss(context, fixed_contexts[t].unsqueeze(0))
                total_loss_value += loss.item()

                # Backprop and update weights for each token
                loss.backward()
                optimizer.step()

                # Check convergence
                if loss.item() < threshold:
                    converged_tokens[t] = True

            # Save context for next iteration and carry over to next token
            fixed_contexts[t] = context.detach().squeeze(0)
            context = context.detach()  # Detach to prevent growing computation graph
            context.requires_grad = True  # Re-enable gradients for next token

        # Compute convergence rate
        convergence_rate = converged_tokens.float().mean().item()

        if iteration == 0:
            print_flush(f"Iteration 1/{max_iters}: Forward pass only (saving contexts)")
        else:
            avg_loss = total_loss_value / len(token_ids)
            print_flush(f"Iteration {iteration+1}/{max_iters}: Loss={avg_loss:.6f}, Converged={convergence_rate*100:.1f}%")

            # Early stopping check
            if early_stopping(convergence_rate):
                print_flush(f"  → Early stopping: Convergence = {convergence_rate*100:.1f}%")
                break

        # Reset convergence tracking for next iteration
        converged_tokens.fill_(False)

    print_flush(f"\nPhase 1 Complete: {converged_tokens.sum()}/{len(token_ids)} tokens converged\n")
    return fixed_contexts.detach()


def compute_fixed_contexts(model, token_ids, max_iters=50, threshold=0.01, device='cpu'):
    """Compute fixed-point contexts without training (evaluation only)

    This function computes fixed-point contexts for validation data
    using the already-trained context generation layers.
    """
    print_flush("\n" + "="*70)
    print_flush("PHASE 1: Fixed-Point Context Learning (CVFP) - Val")
    print_flush("="*70)

    model.to(device)
    model.eval()

    # Token embeddings
    with torch.no_grad():
        token_embeds = model.token_embedding(token_ids.unsqueeze(0).to(device))
        token_embeds = model.embed_norm(token_embeds).squeeze(0)

    # Fixed contexts storage
    fixed_contexts = torch.zeros(len(token_ids), model.context_dim).to(device)

    # Convergence tracking
    converged_tokens = torch.zeros(len(token_ids), dtype=torch.bool)

    for iteration in range(max_iters):
        # Process entire sequence with context carry-over
        context = torch.zeros(1, model.context_dim).to(device)

        for t, token_embed in enumerate(token_embeds):
            with torch.no_grad():
                context = model._update_context_one_step(
                    token_embed.unsqueeze(0),
                    context
                )

                # Check convergence (compare to previous iteration)
                if iteration > 0:
                    loss = torch.nn.functional.mse_loss(context, fixed_contexts[t].unsqueeze(0))
                    if loss.item() < threshold:
                        converged_tokens[t] = True

                # Update fixed context
                fixed_contexts[t] = context.squeeze(0)

        convergence_rate = converged_tokens.float().mean().item()

        if iteration == 0:
            print_flush(f"Iteration 1/{max_iters}: Forward pass only (saving contexts)")
        else:
            print_flush(f"Iteration {iteration+1}/{max_iters}: Loss={loss.item():.6f}, Converged={convergence_rate*100:.1f}%")

            # Stop if converged
            if convergence_rate >= 0.995:
                print_flush(f"  → Early stopping: Convergence = {convergence_rate*100:.1f}%")
                break

        # Reset for next iteration
        converged_tokens.fill_(False)

    print_flush(f"\nPhase 1 Complete: {converged_tokens.sum()}/{len(token_ids)} tokens converged\n")
    return fixed_contexts.detach()


def analyze_fixed_points(contexts, label=""):
    """Analyze fixed-point contexts for degenerate solutions and statistics"""
    print_flush(f"\n{'='*70}")
    print_flush(f"FIXED-POINT ANALYSIS{' - ' + label if label else ''}")
    print_flush(f"{'='*70}")

    # 1. Global Attractor Detection
    print_flush(f"\n1. Global Attractor Detection:")
    n_tokens = len(contexts)
    sample_size = min(100, n_tokens * (n_tokens - 1) // 2)
    indices = torch.randperm(n_tokens)[:min(sample_size * 2, n_tokens)]

    l2_distances = []
    cosine_sims = []
    for i in range(0, len(indices) - 1, 2):
        idx1, idx2 = indices[i], indices[i + 1]
        ctx1, ctx2 = contexts[idx1], contexts[idx2]
        l2_dist = torch.norm(ctx1 - ctx2).item()
        l2_distances.append(l2_dist)
        cos_sim = torch.nn.functional.cosine_similarity(
            ctx1.unsqueeze(0), ctx2.unsqueeze(0)
        ).item()
        cosine_sims.append(cos_sim)

    avg_l2 = np.mean(l2_distances)
    avg_cosine = np.mean(cosine_sims)

    print_flush(f"  Avg L2 Distance: {avg_l2:.6f} (Range: [{min(l2_distances):.6f}, {max(l2_distances):.6f}])")
    print_flush(f"  Avg Cosine Sim:  {avg_cosine:.6f} (Range: [{min(cosine_sims):.6f}, {max(cosine_sims):.6f}])")

    if avg_l2 < 0.01 and avg_cosine > 0.999:
        print_flush(f"  ⚠️  DEGENERATE: Global Attractor detected")
    else:
        print_flush(f"  ✅ Token-specific fixed points")

    # 2. Zero Solution Detection
    print_flush(f"\n2. Zero Solution Detection:")
    norms = torch.norm(contexts, dim=1)
    avg_norm = norms.mean().item()
    print_flush(f"  Avg Norm: {avg_norm:.6f} (Range: [{norms.min().item():.6f}, {norms.max().item():.6f}])")

    if avg_norm < 0.1:
        print_flush(f"  ⚠️  DEGENERATE: Zero Solution detected")
    else:
        print_flush(f"  ✅ Non-zero contexts")

    # 3. Distribution Statistics
    print_flush(f"\n3. Distribution Statistics:")
    print_flush(f"  Norm - Mean: {norms.mean().item():.4f}, Median: {norms.median().item():.4f}, Std: {norms.std().item():.4f}")

    # Pairwise distances
    pairwise_dists = []
    for i in range(0, len(indices) - 1, 2):
        idx1, idx2 = indices[i], indices[i + 1]
        dist = torch.norm(contexts[idx1] - contexts[idx2]).item()
        pairwise_dists.append(dist)
    print_flush(f"  Pairwise Dist - Mean: {np.mean(pairwise_dists):.4f}, Median: {np.median(pairwise_dists):.4f}")

    # Sparsity
    near_zero = (torch.abs(contexts) < 0.01).float().mean().item()
    print_flush(f"  Sparsity: {near_zero*100:.2f}% of values < 0.01")

    # 4. Information Content (Effective Rank)
    print_flush(f"\n4. Information Content:")
    U, S, V = torch.svd(contexts)
    S_norm = S / S.sum()
    effective_rank = 1.0 / (S_norm ** 2).sum().item()
    print_flush(f"  Effective Rank: {effective_rank:.2f} / {len(S)} (max)")
    print_flush(f"  Top 5 Singular Values: {S[:5].tolist()}")

    if effective_rank < len(S) * 0.1:
        print_flush(f"  ⚠️  Low diversity in fixed points")
    else:
        print_flush(f"  ✅ Good diversity")

    print_flush(f"{'='*70}\n")


def phase2_train(model, token_ids, fixed_contexts, val_ids, val_contexts,
                 num_epochs=50, batch_size=32, freeze_context=False, device='cpu'):
    """Phase 2: Token prediction training with validation"""
    print_flush("\n" + "="*70)
    print_flush("PHASE 2: Token Prediction Training")
    if freeze_context:
        print_flush("  Mode: Freeze context (only train token output)")
    else:
        print_flush("  Mode: Train all layers")
    print_flush("="*70)

    model.to(device)
    model.train()

    # Freeze context generation layers if requested
    if freeze_context:
        for name, param in model.named_parameters():
            if 'token_output' not in name:
                param.requires_grad = False
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_flush(f"  Trainable parameters: {trainable_params:,} (token_output only)")
    else:
        trainable_params = model.count_parameters()
        print_flush(f"  Trainable parameters: {trainable_params:,} (all layers)")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=0.0001
    )

    # Early stopping
    early_stopping = Phase2EarlyStopping(patience=5, min_delta=0.0, restore_best=True)

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        correct = 0
        total_tokens = 0

        for i in range(0, len(token_ids) - 1, batch_size):
            batch_end = min(i + batch_size, len(token_ids) - 1)
            batch_contexts = fixed_contexts[i:batch_end].to(device)
            batch_targets = token_ids[i+1:batch_end+1].to(device)

            optimizer.zero_grad()

            logits = model.token_output(batch_contexts)
            loss = nn.functional.cross_entropy(logits, batch_targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_contexts)
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_targets).sum().item()
            total_tokens += len(batch_contexts)

        train_loss = total_loss / total_tokens
        train_acc = correct / total_tokens * 100
        train_ppl = np.exp(train_loss)

        # Validation
        model.eval()
        val_total_loss = 0
        val_correct = 0
        val_total_tokens = 0

        with torch.no_grad():
            for i in range(0, len(val_ids) - 1, batch_size):
                batch_end = min(i + batch_size, len(val_ids) - 1)
                batch_contexts = val_contexts[i:batch_end].to(device)
                batch_targets = val_ids[i+1:batch_end+1].to(device)

                logits = model.token_output(batch_contexts)
                loss = nn.functional.cross_entropy(logits, batch_targets)

                val_total_loss += loss.item() * len(batch_contexts)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == batch_targets).sum().item()
                val_total_tokens += len(batch_contexts)

        val_loss = val_total_loss / val_total_tokens
        val_acc = val_correct / val_total_tokens * 100
        val_ppl = np.exp(val_loss)

        print_flush(f"  Epoch {epoch+1}/{num_epochs}:")
        print_flush(f"    Train: Loss={train_loss:.4f}, PPL={train_ppl:.2f}, Acc={train_acc:.1f}%")
        print_flush(f"    Val:   Loss={val_loss:.4f}, PPL={val_ppl:.2f}, Acc={val_acc:.1f}%")

        # Early stopping check
        if early_stopping(val_loss, val_ppl, model):
            print_flush(f"  → Early stopping: Best Val PPL = {early_stopping.best_ppl:.2f}")
            break

    # Restore best model
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

    print_flush("\nPhase 2 Complete\n")

    # Return best validation metrics
    return {
        'loss': early_stopping.best_loss,
        'ppl': early_stopping.best_ppl,
        'acc': val_acc  # Use last val_acc (approximation)
    }




def main():
    parser = argparse.ArgumentParser(description='Test Residual Standard Architecture')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--layer-structure', type=int, nargs='+', default=[1, 1, 1, 1],
                        help='Layer structure (e.g., 1 1 1 1 for 4 blocks, 1 1 for 2 blocks)')
    parser.add_argument('--context-dim', type=int, default=256,
                        help='Context vector dimension (default: 256)')
    parser.add_argument('--embed-dim', type=int, default=256,
                        help='Token embedding dimension (default: 256)')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--skip-phase2', action='store_true',
                        help='Skip Phase 2 (only run Phase 1)')
    parser.add_argument('--freeze-context', action='store_true',
                        help='Freeze context in Phase 2 (only train token output)')
    args = parser.parse_args()

    print_flush("="*70)
    print_flush("Residual Standard Architecture Test")
    print_flush("="*70)

    # Load data
    texts = load_ultrachat_samples(args.num_samples)
    all_token_ids, tokenizer = tokenize_texts(texts)

    # Split into train/val (80/20)
    split_idx = int(len(all_token_ids) * 0.8)
    train_ids = all_token_ids[:split_idx]
    val_ids = all_token_ids[split_idx:]

    print_flush(f"  Train: {len(train_ids)} tokens")
    print_flush(f"  Val:   {len(val_ids)} tokens")

    print_flush(f"\n{'='*70}")
    print_flush(f"Testing: Residual Standard {args.layer_structure}")
    print_flush(f"  Context dim: {args.context_dim}, Embed dim: {args.embed_dim}")
    print_flush(f"{'='*70}")

    # Create model
    # hidden_dim = embed_dim + context_dim (for concatenation)
    hidden_dim = args.embed_dim + args.context_dim

    model = NewLLMResidual(
        vocab_size=tokenizer.vocab_size,
        embed_dim=args.embed_dim,
        context_dim=args.context_dim,
        hidden_dim=hidden_dim,
        layer_structure=args.layer_structure,
        dropout=0.1
    )

    print_flush(f"  Parameters: {model.count_parameters():,}")

    # Phase 1: Fixed-point learning
    # Train: Learn context generation layers
    # Val: Compute fixed contexts only (evaluation of generalization)
    train_contexts = phase1_train(model, train_ids, device=args.device)
    val_contexts = compute_fixed_contexts(model, val_ids, device=args.device)

    # Analyze fixed points for degenerate solutions
    analyze_fixed_points(train_contexts, label="Train")
    analyze_fixed_points(val_contexts, label="Val")

    # Phase 2: Token prediction (skip if requested)
    if not args.skip_phase2:
        val_metrics = phase2_train(model, train_ids, train_contexts, val_ids, val_contexts,
                                    freeze_context=args.freeze_context, device=args.device)

        # Final results
        print_flush("\n" + "="*70)
        print_flush("FINAL RESULTS")
        print_flush("="*70)
        print_flush(f"\nResidual Standard {args.layer_structure}:")
        print_flush(f"  Context dim: {args.context_dim}, Embed dim: {args.embed_dim}")
        print_flush(f"  Parameters: {model.count_parameters():,}")
        print_flush(f"  Best Val Loss: {val_metrics['loss']:.2f}")
        print_flush(f"  Best Val PPL:  {val_metrics['ppl']:.2f}")
        print_flush(f"  Best Val Acc:  {val_metrics['acc']:.1f}%")
    else:
        print_flush("\n" + "="*70)
        print_flush("PHASE 1 ONLY - SKIPPING PHASE 2")
        print_flush("="*70)


if __name__ == '__main__':
    main()
