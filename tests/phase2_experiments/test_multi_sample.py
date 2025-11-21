"""Multi-Sample Training with Early Stopping and Cache

Features:
1. Train/Validation split (80/20)
2. Phase 1 with early stopping (convergence-based)
3. Phase 2 with early stopping (validation loss-based)
4. Fixed-point context caching (auto load/save)
5. Mixed [2,2] architecture (default)

Usage:
    # 10 samples
    python3 tests/phase2_experiments/test_multi_sample.py --num-samples 10

    # 100 samples with custom architecture
    python3 tests/phase2_experiments/test_multi_sample.py --num-samples 100 --layer-structure 2 2 2
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from src.models.new_llm_flexible import NewLLMFlexible
from src.utils.early_stopping import Phase1EarlyStopping, Phase2EarlyStopping
from src.utils.cache_manager import FixedContextCache


def print_flush(*args, **kwargs):
    """Print with immediate flush to ensure real-time output"""
    print(*args, **kwargs)
    sys.stdout.flush()


def load_samples(num_samples=10, train_ratio=0.8, tokenizer_model='gpt2', max_length=512):
    """Load multiple UltraChat samples with train/val split"""
    print_flush(f"\nLoading {num_samples} samples from UltraChat...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    train_size = int(num_samples * train_ratio)

    train_tokens = []
    val_tokens = []

    for i in range(num_samples):
        sample = dataset[i]
        text = ""
        for msg in sample['messages']:
            text += msg['content'] + " "

        tokens = tokenizer.encode(text, add_special_tokens=False, max_length=max_length, truncation=True)

        if i < train_size:
            train_tokens.extend(tokens)
        else:
            val_tokens.extend(tokens)

    train_token_ids = torch.tensor(train_tokens)
    val_token_ids = torch.tensor(val_tokens)

    print_flush(f"  Train: {train_size} samples, {len(train_token_ids)} tokens")
    print_flush(f"  Val:   {num_samples - train_size} samples, {len(val_token_ids)} tokens")

    return train_token_ids, val_token_ids, tokenizer


def phase1_train(model, token_ids, early_stop, learning_rate=0.0001, max_epochs=20, threshold=0.01):
    """Phase 1: Train fixed-point contexts with CVFP (Context Vector Fixed-Point Property)

    CVFP Principle:
    - Iteration 1: Forward pass only, save contexts
    - Iteration 2+: Learn so that context[t] matches previous context[t-period]
    - Example: "red apple" → "red apple red" should output same context as "red"

    ONLY trains context generation layers (excludes token_output layer)
    """
    print_flush(f"\nPhase 1: Training Fixed-Point Contexts - CVFP ({len(token_ids)} tokens)")
    print_flush("="*70)

    # Optimizer for Phase 1 - ONLY train context generation layers (exclude token_output)
    context_params = [p for name, p in model.named_parameters() if 'token_output' not in name]
    optimizer = optim.Adam(context_params, lr=learning_rate)
    print_flush(f"  Trainable parameters: {sum(p.numel() for p in context_params):,} (context generation only)")

    model.train()

    input_ids = token_ids.unsqueeze(0)
    token_embeds = model.token_embedding(input_ids)
    token_embeds = model.embed_norm(token_embeds)
    token_embeds = token_embeds.squeeze(0)

    # Fixed contexts storage
    fixed_contexts = torch.zeros(len(token_ids), model.context_dim)

    # Convergence tracking
    converged_tokens = torch.zeros(len(token_ids), dtype=torch.bool)

    for epoch in range(max_epochs):
        total_loss = 0

        # Process entire sequence with context carry-over
        context = torch.zeros(1, model.context_dim)

        for t, token_embed in enumerate(token_embeds):
            if epoch == 0:
                # Epoch 1: Forward pass only, no learning
                with torch.no_grad():
                    context = model._update_context_one_step(
                        token_embed.unsqueeze(0),
                        context
                    )
                    fixed_contexts[t] = context.squeeze(0)
            else:
                # Epoch 2+: Learn to match previous epoch's context
                if converged_tokens[t]:
                    # Already converged, just forward (no grad)
                    with torch.no_grad():
                        context = model._update_context_one_step(
                            token_embed.unsqueeze(0),
                            context
                        )
                else:
                    # Not converged, learn
                    optimizer.zero_grad()
                    context_new = model._update_context_one_step(
                        token_embed.unsqueeze(0).detach(),
                        context.detach()
                    )

                    # Loss: match previous epoch's context
                    loss = torch.nn.functional.mse_loss(context_new, fixed_contexts[t].unsqueeze(0))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    total_loss += loss.item()

                    # Check convergence
                    if loss.item() < threshold:
                        converged_tokens[t] = True

                    # Update context for next token (detach to avoid backprop through sequence)
                    context = context_new.detach()

                    # Update fixed context for next epoch
                    fixed_contexts[t] = context.squeeze(0)

        if epoch > 0:
            convergence_rate = converged_tokens.float().mean().item()
            avg_loss = total_loss / (~converged_tokens).sum().item() if (~converged_tokens).sum() > 0 else 0

            print_flush(f"  Epoch {epoch+1}/{max_epochs}: Loss = {avg_loss:.6f}, Convergence = {convergence_rate:.1%}")

            if early_stop(convergence_rate):
                print_flush(f"  → Early stopping: Convergence = {convergence_rate:.1%}")
                break
        else:
            print_flush(f"  Epoch 1/{max_epochs}: Forward pass only (saving contexts)")

    print_flush(f"Phase 1 Complete\n")
    return converged_tokens.float().mean().item()


def compute_fixed_contexts(model, token_ids, threshold=0.01):
    """Compute fixed-point contexts"""
    print_flush(f"Computing fixed-point contexts for {len(token_ids)} tokens...")

    model.eval()
    with torch.no_grad():
        contexts, converged, iters = model.get_fixed_point_context(
            token_ids.unsqueeze(0),
            max_iterations=50,
            tolerance=threshold,
            warmup_iterations=0
        )

    conv_count = converged.sum().item()
    print_flush(f"  Converged: {conv_count}/{len(token_ids)} ({conv_count/len(token_ids):.1%})")
    print_flush(f"  Avg Iterations: {iters.float().mean():.1f}\n")

    return contexts.squeeze(0), converged.squeeze(0)


def phase2_train(model, train_token_ids, train_contexts, val_token_ids, val_contexts,
                early_stop, learning_rate=0.0001, max_epochs=50):
    """Phase 2: Train token prediction with early stopping

    ONLY trains token_output layer (freezes context generation layers)
    """
    print_flush(f"Phase 2: Training Token Prediction")
    print_flush("="*70)

    # Optimizer for Phase 2 - ONLY train token_output layer
    output_params = [p for name, p in model.named_parameters() if 'token_output' in name]
    optimizer = optim.Adam(output_params, lr=learning_rate)
    print_flush(f"  Trainable parameters: {sum(p.numel() for p in output_params):,} (token_output only)\n")

    for epoch in range(max_epochs):
        # Training
        train_loss, train_ppl, train_acc = phase2_epoch(
            model, train_token_ids, train_contexts, optimizer, is_training=True
        )

        # Validation
        val_loss, val_ppl, val_acc = phase2_epoch(
            model, val_token_ids, val_contexts, optimizer=None, is_training=False
        )

        print_flush(f"  Epoch {epoch+1}/{max_epochs}:")
        print_flush(f"    Train: Loss={train_loss:.4f}, PPL={train_ppl:.2f}, Acc={train_acc:.1%}")
        print_flush(f"    Val:   Loss={val_loss:.4f}, PPL={val_ppl:.2f}, Acc={val_acc:.1%}")

        if early_stop(val_loss, val_ppl, model):
            print_flush(f"  → Early stopping: Best Val PPL = {early_stop.best_ppl:.2f}")
            break

    print_flush(f"\nPhase 2 Complete\n")
    return (train_loss, train_ppl, train_acc), (val_loss, val_ppl, val_acc)


def phase2_epoch(model, token_ids, fixed_contexts, optimizer=None, is_training=True, batch_size=32):
    """Single epoch of Phase 2 with batch processing"""
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0
    num_samples = len(token_ids) - 1

    # Process in batches
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        actual_batch_size = batch_end - batch_start

        # Prepare batch
        contexts_batch = []
        token_embeds_batch = []
        targets_batch = []

        for t in range(batch_start, batch_end):
            contexts_batch.append(fixed_contexts[t])
            token_embeds_batch.append(model.token_embedding(token_ids[t].unsqueeze(0).unsqueeze(0)))
            targets_batch.append(token_ids[t + 1])

        # Stack into batches
        contexts_batch = torch.stack(contexts_batch)  # (batch, context_dim)
        token_embeds_batch = torch.cat(token_embeds_batch, dim=0)  # (batch, 1, embed_dim)
        token_embeds_batch = model.embed_norm(token_embeds_batch).squeeze(1)  # (batch, embed_dim)
        targets_batch = torch.stack(targets_batch)  # (batch,)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            _, hidden_batch = model._update_context_one_step(
                token_embeds_batch,
                contexts_batch,
                return_hidden=True
            )

            logits_batch = model.token_output(hidden_batch)  # (batch, vocab_size)
            loss = nn.functional.cross_entropy(logits_batch, targets_batch)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item() * actual_batch_size
        pred_batch = logits_batch.argmax(dim=-1)
        correct += (pred_batch == targets_batch).sum().item()
        total += actual_batch_size

    avg_loss = total_loss / total
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = correct / total

    return avg_loss, ppl, accuracy


def main():
    parser = argparse.ArgumentParser(description='Multi-Sample Training')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--train-ratio', type=float, default=0.8)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--vocab-size', type=int, default=50257)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--context-dim', type=int, default=256)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--layer-structure', type=int, nargs='+', default=[2, 2])
    parser.add_argument('--phase1-lr', type=float, default=0.0001)
    parser.add_argument('--phase2-lr', type=float, default=0.0001)
    parser.add_argument('--phase1-max-epochs', type=int, default=20)
    parser.add_argument('--phase2-max-epochs', type=int, default=50)
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache before run')

    args = parser.parse_args()

    print_flush("\n" + "="*70)
    print_flush("Multi-Sample Training with Early Stopping and Cache")
    print_flush("="*70)

    # Initialize cache
    cache = FixedContextCache()
    if args.clear_cache:
        cache.clear()

    # Load data
    train_token_ids, val_token_ids, tokenizer = load_samples(
        num_samples=args.num_samples,
        train_ratio=args.train_ratio,
        max_length=args.max_length
    )

    # Create model
    print_flush(f"\nCreating model with layer structure {args.layer_structure}...")
    model = NewLLMFlexible(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        layer_structure=args.layer_structure,
        dropout=0.1
    )
    arch_desc = model.get_architecture_description()
    print_flush(f"  Architecture: {arch_desc}")
    print_flush(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    model_config = {
        'vocab_size': args.vocab_size,
        'embed_dim': args.embed_dim,
        'context_dim': args.context_dim,
        'hidden_dim': args.hidden_dim,
        'layer_structure': args.layer_structure
    }

    # Phase 1: Train or load fixed contexts
    print_flush("\n" + "="*70)
    print_flush("PHASE 1: Fixed-Point Context Learning")
    print_flush("="*70)

    # Try to load from cache
    train_contexts, train_converged = cache.load(train_token_ids, arch_desc, model_config)
    val_contexts, val_converged = cache.load(val_token_ids, arch_desc, model_config)

    if train_contexts is None:
        # Train Phase 1 - require near-perfect convergence (99.5%)
        # This is critical for stable Phase 2 training
        phase1_early_stop = Phase1EarlyStopping(
            convergence_threshold=0.995,  # 99.5% convergence required
            patience=999,                  # Effectively disable patience-based stopping
            min_delta=0.0                  # No min_delta check (convergence-only)
        )

        phase1_train(model, train_token_ids, phase1_early_stop,
                    learning_rate=args.phase1_lr,
                    max_epochs=args.phase1_max_epochs,
                    threshold=0.01)

        # Compute and cache
        train_contexts, train_converged = compute_fixed_contexts(model, train_token_ids)
        cache.save(train_token_ids, train_contexts, train_converged, arch_desc, model_config)
    else:
        print_flush("Using cached training fixed contexts")

    if val_contexts is None:
        val_contexts, val_converged = compute_fixed_contexts(model, val_token_ids)
        cache.save(val_token_ids, val_contexts, val_converged, arch_desc, model_config)
    else:
        print_flush("Using cached validation fixed contexts")

    # Phase 2: Train token prediction
    print_flush("\n" + "="*70)
    print_flush("PHASE 2: Token Prediction Training")
    print_flush("="*70)

    phase2_early_stop = Phase2EarlyStopping(
        patience=5,
        min_delta=0.001,
        restore_best=True
    )

    train_metrics, val_metrics = phase2_train(
        model, train_token_ids, train_contexts,
        val_token_ids, val_contexts, phase2_early_stop,
        learning_rate=args.phase2_lr,
        max_epochs=args.phase2_max_epochs
    )

    # Final results
    print_flush("\n" + "="*70)
    print_flush("FINAL RESULTS")
    print_flush("="*70)
    print_flush(f"Training:   Loss={train_metrics[0]:.4f}, PPL={train_metrics[1]:.2f}, Acc={train_metrics[2]:.1%}")
    print_flush(f"Validation: Loss={val_metrics[0]:.4f}, PPL={val_metrics[1]:.2f}, Acc={val_metrics[2]:.1%}")
    print_flush(f"PPL Difference (Val - Train): {val_metrics[1] - train_metrics[1]:.2f}")
    print_flush("="*70)

    # Cache stats
    cache.stats()


if __name__ == '__main__':
    main()
