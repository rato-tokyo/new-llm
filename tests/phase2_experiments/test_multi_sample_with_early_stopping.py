"""Test Multi-Sample Training with Early Stopping

Features:
1. Load multiple UltraChat samples (10, 50, or 100)
2. Train/Validation split (80/20)
3. Phase 1 with early stopping (convergence-based)
4. Phase 2 with early stopping (validation loss-based)
5. Mixed [2,2] architecture (current best)

Usage:
    python3 tests/phase2_experiments/test_multi_sample_with_early_stopping.py --num-samples 10
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from src.models.new_llm_flexible import NewLLMFlexible
from src.utils.early_stopping import Phase1EarlyStopping, Phase2EarlyStopping


def load_samples(num_samples=10, train_ratio=0.8, tokenizer_model='gpt2', max_length=512):
    """Load multiple UltraChat samples with train/val split

    Args:
        num_samples: Total number of samples to load
        train_ratio: Ratio of training samples (default: 0.8)
        tokenizer_model: Tokenizer to use
        max_length: Maximum tokens per sample

    Returns:
        train_token_ids: Tensor of training tokens
        val_token_ids: Tensor of validation tokens
        tokenizer: Loaded tokenizer
    """
    print(f"\nLoading {num_samples} samples from UltraChat...")
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

    print(f"  Train: {train_size} samples, {len(train_token_ids)} tokens")
    print(f"  Val:   {num_samples - train_size} samples, {len(val_token_ids)} tokens")

    return train_token_ids, val_token_ids, tokenizer


def phase1_train_with_early_stopping(model, token_ids, early_stop, learning_rate=0.0001,
                                     max_epochs=20, threshold=0.01):
    """Phase 1: Train fixed-point contexts with early stopping

    Args:
        model: NewLLMFlexible model
        token_ids: Training token IDs
        early_stop: Phase1EarlyStopping instance
        learning_rate: Learning rate
        max_epochs: Maximum epochs
        threshold: Convergence threshold per token

    Returns:
        final_convergence_rate: Final convergence rate
    """
    print(f"\nPhase 1: Training Fixed-Point Contexts ({len(token_ids)} tokens)")
    print("="*70)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    input_ids = token_ids.unsqueeze(0)
    token_embeds = model.token_embedding(input_ids)
    token_embeds = model.embed_norm(token_embeds)
    token_embeds = token_embeds.squeeze(0)

    for epoch in range(max_epochs):
        total_loss = 0
        converged_count = 0

        for t, token_embed in enumerate(token_embeds):
            context = torch.zeros(1, model.context_dim)

            for iteration in range(50):
                optimizer.zero_grad()
                context_new = model._update_context_one_step(
                    token_embed.unsqueeze(0).detach(),
                    context.detach()
                )

                loss = torch.nn.functional.mse_loss(context_new, context.detach())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                context = context_new.detach()
                total_loss += loss.item()

                if loss.item() < threshold:
                    converged_count += 1
                    break

        convergence_rate = converged_count / len(token_ids)
        avg_loss = total_loss / len(token_ids)

        print(f"  Epoch {epoch+1}/{max_epochs}: Loss = {avg_loss:.6f}, Convergence = {convergence_rate:.1%}")

        # Check early stopping
        if early_stop(convergence_rate):
            print(f"  → Early stopping triggered (convergence = {convergence_rate:.1%})")
            break

    print(f"Phase 1 Complete: Best Convergence = {early_stop.best_convergence:.1%}\n")
    return convergence_rate


def compute_fixed_contexts(model, token_ids, threshold=0.01):
    """Compute fixed-point contexts for token IDs"""
    print(f"Computing Fixed-Point Contexts for {len(token_ids)} tokens...")

    model.eval()
    with torch.no_grad():
        contexts, converged, iters = model.get_fixed_point_context(
            token_ids.unsqueeze(0),
            max_iterations=50,
            tolerance=threshold,
            warmup_iterations=0
        )

    conv_count = converged.sum().item()
    print(f"  Converged: {conv_count}/{len(token_ids)} ({conv_count/len(token_ids):.1%})")
    print(f"  Avg Iterations: {iters.float().mean():.1f}\n")

    return contexts.squeeze(0), converged.squeeze(0)


def phase2_train_with_early_stopping(model, train_token_ids, train_contexts,
                                     val_token_ids, val_contexts, early_stop,
                                     learning_rate=0.0001, max_epochs=50):
    """Phase 2: Train token prediction with early stopping

    Args:
        model: NewLLMFlexible model
        train_token_ids: Training token IDs
        train_contexts: Fixed contexts for training
        val_token_ids: Validation token IDs
        val_contexts: Fixed contexts for validation
        early_stop: Phase2EarlyStopping instance
        learning_rate: Learning rate
        max_epochs: Maximum epochs

    Returns:
        train_metrics: Final training metrics
        val_metrics: Final validation metrics
    """
    print(f"Phase 2: Training Token Prediction")
    print("="*70)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(max_epochs):
        # Training
        train_loss, train_ppl, train_acc = phase2_epoch(
            model, train_token_ids, train_contexts, optimizer, is_training=True
        )

        # Validation
        val_loss, val_ppl, val_acc = phase2_epoch(
            model, val_token_ids, val_contexts, optimizer=None, is_training=False
        )

        print(f"  Epoch {epoch+1}/{max_epochs}:")
        print(f"    Train: Loss = {train_loss:.4f}, PPL = {train_ppl:.2f}, Acc = {train_acc:.1%}")
        print(f"    Val:   Loss = {val_loss:.4f}, PPL = {val_ppl:.2f}, Acc = {val_acc:.1%}")

        # Check early stopping
        if early_stop(val_loss, val_ppl, model):
            print(f"  → Early stopping triggered")
            print(f"    Best Val Loss = {early_stop.best_loss:.4f}, Best Val PPL = {early_stop.best_ppl:.2f}")
            break

    print(f"\nPhase 2 Complete\n")
    return (train_loss, train_ppl, train_acc), (val_loss, val_ppl, val_acc)


def phase2_epoch(model, token_ids, fixed_contexts, optimizer=None, is_training=True):
    """Single epoch of Phase 2 training or evaluation"""
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for t in range(len(token_ids) - 1):
        context = fixed_contexts[t].unsqueeze(0)
        target = token_ids[t + 1].unsqueeze(0)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            # Get last hidden state (from final block)
            _, hidden = model._update_context_one_step(
                model.embed_norm(model.token_embedding(token_ids[t].unsqueeze(0).unsqueeze(0))).squeeze(0),
                context,
                return_hidden=True
            )

            logits = model.output_proj(hidden)
            loss = nn.functional.cross_entropy(logits, target)

            if is_training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == target).sum().item()
        total += 1

    avg_loss = total_loss / total
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    accuracy = correct / total

    return avg_loss, ppl, accuracy


def main():
    parser = argparse.ArgumentParser(description='Multi-Sample Training with Early Stopping')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples (default: 10)')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/val split ratio (default: 0.8)')
    parser.add_argument('--max-length', type=int, default=512, help='Max tokens per sample (default: 512)')
    parser.add_argument('--vocab-size', type=int, default=50257, help='Vocabulary size (default: 50257 for GPT2)')
    parser.add_argument('--embed-dim', type=int, default=256, help='Embedding dimension (default: 256)')
    parser.add_argument('--context-dim', type=int, default=256, help='Context dimension (default: 256)')
    parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension (default: 512)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Multi-Sample Training with Early Stopping")
    print("="*70)

    # Load data
    train_token_ids, val_token_ids, tokenizer = load_samples(
        num_samples=args.num_samples,
        train_ratio=args.train_ratio,
        max_length=args.max_length
    )

    # Create model (Mixed [2,2] - current best architecture)
    print(f"\nCreating Mixed [2,2] model...")
    model = NewLLMFlexible(
        vocab_size=args.vocab_size,
        embed_dim=args.embed_dim,
        context_dim=args.context_dim,
        hidden_dim=args.hidden_dim,
        layer_structure=[2, 2],
        dropout=0.1
    )
    print(f"  Architecture: {model.get_architecture_description()}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Phase 1: Train fixed-point contexts
    phase1_early_stop = Phase1EarlyStopping(
        convergence_threshold=0.95,
        patience=5,
        min_delta=0.01
    )

    phase1_train_with_early_stopping(
        model, train_token_ids, phase1_early_stop,
        learning_rate=0.0001, max_epochs=20, threshold=0.01
    )

    # Compute fixed contexts
    train_contexts, train_converged = compute_fixed_contexts(model, train_token_ids, threshold=0.01)
    val_contexts, val_converged = compute_fixed_contexts(model, val_token_ids, threshold=0.01)

    # Phase 2: Train token prediction
    phase2_early_stop = Phase2EarlyStopping(
        patience=10,
        min_delta=0.001,
        ppl_threshold=None,  # No target PPL
        restore_best=True
    )

    train_metrics, val_metrics = phase2_train_with_early_stopping(
        model, train_token_ids, train_contexts,
        val_token_ids, val_contexts, phase2_early_stop,
        learning_rate=0.0001, max_epochs=50
    )

    # Final results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Training:   Loss = {train_metrics[0]:.4f}, PPL = {train_metrics[1]:.2f}, Acc = {train_metrics[2]:.1%}")
    print(f"Validation: Loss = {val_metrics[0]:.4f}, PPL = {val_metrics[1]:.2f}, Acc = {val_metrics[2]:.1%}")
    print("="*70)


if __name__ == '__main__':
    main()
