#!/usr/bin/env python3
"""CVFPT Validation Test

This script validates that CVFPT (Context Vector Fixed Point Training)
works correctly by testing on different sequence types:

1. Single token repetition (cycle=1) - Should converge to zero loss
2. Random sequence - Should NOT converge (high loss)
3. 2-gram repetition (cycle=2) - Should converge to zero loss
4. 3-gram repetition (cycle=3) - Should converge to zero loss

This helps verify that:
- Loss calculation is correct
- Model can learn fixed points
- Model doesn't trivially memorize
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.new_llm import NewLLM
from src.utils.config import NewLLMConfig
from src.training.convergence_loss import ContextConvergenceLoss, compute_context_change_rate


class ValidationDataset(Dataset):
    """Dataset for validation tests"""

    def __init__(self, sequences, tokenizer, max_length=512):
        """
        Args:
            sequences: List of token ID sequences
            tokenizer: Tokenizer
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Truncate if too long
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]

        return torch.tensor(seq, dtype=torch.long)


def collate_fn(batch):
    """Collate function for DataLoader"""
    # Find max length in batch
    max_len = max(len(seq) for seq in batch)

    # Pad sequences
    padded = []
    for seq in batch:
        if len(seq) < max_len:
            padding = torch.zeros(max_len - len(seq), dtype=torch.long)
            seq = torch.cat([seq, padding])
        padded.append(seq)

    return torch.stack(padded)


def create_single_token_dataset(token_id, repetitions, num_samples, tokenizer):
    """Create dataset with single token repetition

    Example: [5, 5, 5, 5, 5, ...] (token_id=5, repeated)
    Expected: Should converge (cycle_length=1)
    """
    sequences = []
    for _ in range(num_samples):
        seq = [token_id] * repetitions
        sequences.append(seq)

    return ValidationDataset(sequences, tokenizer)


def create_random_dataset(vocab_size, seq_length, num_samples, tokenizer):
    """Create dataset with random sequences

    Example: [3, 7, 2, 9, 1, ...] (random tokens)
    Expected: Should NOT converge (no fixed point)
    """
    sequences = []
    for _ in range(num_samples):
        seq = torch.randint(1, vocab_size, (seq_length,)).tolist()
        sequences.append(seq)

    return ValidationDataset(sequences, tokenizer)


def create_ngram_dataset(tokens, cycle_length, repetitions, num_samples, tokenizer):
    """Create dataset with n-gram repetition

    Example (cycle_length=2): [5, 7, 5, 7, 5, 7, ...]
    Example (cycle_length=3): [5, 7, 9, 5, 7, 9, ...]
    Expected: Should converge
    """
    sequences = []
    for _ in range(num_samples):
        pattern = tokens[:cycle_length]
        seq = (pattern * repetitions)[:repetitions * cycle_length]
        sequences.append(seq)

    return ValidationDataset(sequences, tokenizer)


def train_and_evaluate(dataset, model, loss_fn, optimizer, device, epochs, test_name):
    """Train model and evaluate convergence

    Returns:
        Dictionary with test results
    """
    print(f"\n{'='*80}")
    print(f"🧪 Test: {test_name}")
    print(f"{'='*80}")

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    model.train()

    epoch_losses = []
    epoch_change_rates = []

    # Initial evaluation (before training)
    with torch.no_grad():
        batch = next(iter(dataloader))
        input_ids = batch.to(device)
        logits, context_trajectory = model(input_ids)
        loss, metrics = loss_fn(
            context_vectors=context_trajectory,
            logits=logits,
            targets=input_ids
        )
        epoch_losses.append(metrics['loss'])
        change_rate = compute_context_change_rate(context_trajectory)
        epoch_change_rates.append(change_rate)

        print(f"Epoch 0 (Initial): Loss={metrics['loss']:.6f}, Change={change_rate:.6f}")

    # Training loop
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_change = 0.0
        num_batches = 0

        for input_ids in dataloader:
            input_ids = input_ids.to(device)

            optimizer.zero_grad()
            logits, context_trajectory = model(input_ids)

            loss, metrics = loss_fn(
                context_vectors=context_trajectory,
                logits=logits,
                targets=input_ids
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += metrics['loss']
            change_rate = compute_context_change_rate(context_trajectory)
            epoch_change += change_rate
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_change = epoch_change / num_batches

        epoch_losses.append(avg_loss)
        epoch_change_rates.append(avg_change)

        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.6f}, Change={avg_change:.6f}")

    total_time = time.time() - start_time

    # Results
    initial_loss = epoch_losses[0]
    final_loss = epoch_losses[-1]
    loss_reduction = initial_loss - final_loss
    improvement_rate = (loss_reduction / initial_loss * 100) if initial_loss > 0 else 0

    converged = final_loss < 0.01

    print(f"\n📊 Results:")
    print(f"  Initial Loss: {initial_loss:.6f}")
    print(f"  Final Loss: {final_loss:.6f}")
    print(f"  Loss Reduction: {loss_reduction:.6f} ({improvement_rate:.1f}%)")
    print(f"  Converged: {'✓ YES' if converged else '✗ NO'}")
    print(f"  Time: {total_time:.2f}s")

    return {
        'test_name': test_name,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_reduction': loss_reduction,
        'improvement_rate': improvement_rate,
        'converged': converged,
        'time': total_time,
        'epoch_losses': epoch_losses,
        'epoch_change_rates': epoch_change_rates
    }


def main():
    print("="*80)
    print("🔬 CVFPT Validation Tests (Gated Context Updater)")
    print("="*80)
    print("\nPurpose: Verify that CVFPT loss calculation works correctly")
    print("Context Update: LSTM-style gated additive (forget_gate * old + input_gate * new)")
    print("\nTests:")
    print("  1. Single token repetition (cycle=1) - Should converge ✓")
    print("  2. Random sequence - Should NOT converge ✗")
    print("  3. 2-gram repetition (cycle=2) - Should converge ✓")
    print("  4. 3-gram repetition (cycle=3) - Should converge ✓")
    print("="*80)

    # Device
    device = torch.device('cpu')
    print(f"\n🔧 Using device: {device}")

    # Tokenizer
    print("\n📚 Initializing tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    # Model config (small for fast testing)
    config = NewLLMConfig()
    config.vocab_size = len(tokenizer)
    config.context_vector_dim = 128  # Small for fast testing
    config.embed_dim = 128
    config.hidden_dim = 256
    config.num_layers = 2
    config.max_seq_length = 512
    config.context_update_strategy = 'gated'  # Use gated updater instead of simple

    # Training params
    epochs = 10  # Increased from 5 to allow full convergence
    repetitions = 30
    num_samples = 10

    # CRITICAL: Add reconstruction loss to prevent degenerate solutions
    # Without this, model can learn to "not update context" which gives
    # low loss on both repetitive AND random sequences (incorrect!)
    # Use small weight (0.01) to keep focus on convergence while preventing degeneration
    reconstruction_weight = 0.01  # Token prediction / reconstruction loss (reduced from 0.1)

    results = []

    # ========== Test 1: Single Token Repetition ==========
    print("\n" + "="*80)
    print("Test 1: Single Token Repetition (cycle_length=1)")
    print("="*80)
    print("Expected: Should converge to near-zero loss")

    # Create model and optimizer
    model = NewLLM(config).to(device)
    loss_fn = ContextConvergenceLoss(
        cycle_length=1,
        convergence_weight=1.0,
        token_weight=reconstruction_weight  # Add reconstruction loss
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create dataset
    dataset = create_single_token_dataset(
        token_id=100,  # Arbitrary token
        repetitions=repetitions,
        num_samples=num_samples,
        tokenizer=tokenizer
    )

    result = train_and_evaluate(
        dataset, model, loss_fn, optimizer, device, epochs,
        test_name="Single Token Repetition"
    )
    results.append(result)

    # ========== Test 2: Random Sequence ==========
    print("\n" + "="*80)
    print("Test 2: Random Sequence")
    print("="*80)
    print("Expected: Should NOT converge (high loss)")

    # Create new model
    model = NewLLM(config).to(device)
    loss_fn = ContextConvergenceLoss(
        cycle_length=1,
        convergence_weight=1.0,
        token_weight=reconstruction_weight  # Add reconstruction loss
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create dataset
    dataset = create_random_dataset(
        vocab_size=1000,
        seq_length=repetitions,
        num_samples=num_samples,
        tokenizer=tokenizer
    )

    result = train_and_evaluate(
        dataset, model, loss_fn, optimizer, device, epochs,
        test_name="Random Sequence"
    )
    results.append(result)

    # ========== Test 3: 2-gram Repetition ==========
    print("\n" + "="*80)
    print("Test 3: 2-gram Repetition (cycle_length=2)")
    print("="*80)
    print("Expected: Should converge to near-zero loss")

    # Create new model
    model = NewLLM(config).to(device)
    loss_fn = ContextConvergenceLoss(
        cycle_length=2,
        convergence_weight=1.0,
        token_weight=reconstruction_weight  # Add reconstruction loss
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create dataset
    dataset = create_ngram_dataset(
        tokens=[100, 200],  # 2 tokens
        cycle_length=2,
        repetitions=repetitions // 2,
        num_samples=num_samples,
        tokenizer=tokenizer
    )

    result = train_and_evaluate(
        dataset, model, loss_fn, optimizer, device, epochs,
        test_name="2-gram Repetition"
    )
    results.append(result)

    # ========== Test 4: 3-gram Repetition ==========
    print("\n" + "="*80)
    print("Test 4: 3-gram Repetition (cycle_length=3)")
    print("="*80)
    print("Expected: Should converge to near-zero loss")

    # Create new model
    model = NewLLM(config).to(device)
    loss_fn = ContextConvergenceLoss(
        cycle_length=3,
        convergence_weight=1.0,
        token_weight=reconstruction_weight  # Add reconstruction loss
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create dataset
    dataset = create_ngram_dataset(
        tokens=[100, 200, 300],  # 3 tokens
        cycle_length=3,
        repetitions=repetitions // 3,
        num_samples=num_samples,
        tokenizer=tokenizer
    )

    result = train_and_evaluate(
        dataset, model, loss_fn, optimizer, device, epochs,
        test_name="3-gram Repetition"
    )
    results.append(result)

    # ========== Summary ==========
    print("\n" + "="*80)
    print("📊 Validation Summary")
    print("="*80)

    print(f"\n{'Test':<30} {'Initial':<12} {'Final':<12} {'Converged':<12} {'Status'}")
    print("-"*80)

    for result in results:
        status = "✓ PASS" if result['converged'] else "✗ FAIL"

        # Special case: Random sequence should NOT converge
        if result['test_name'] == "Random Sequence":
            status = "✗ FAIL" if result['converged'] else "✓ PASS"

        print(f"{result['test_name']:<30} "
              f"{result['initial_loss']:<12.6f} "
              f"{result['final_loss']:<12.6f} "
              f"{'YES' if result['converged'] else 'NO':<12} "
              f"{status}")

    # Validation verdict
    print("\n" + "="*80)
    print("🎯 Validation Verdict")
    print("="*80)

    # Check if results match expectations
    test1_pass = results[0]['converged']  # Single token should converge
    test2_pass = not results[1]['converged']  # Random should NOT converge
    test3_pass = results[2]['converged']  # 2-gram should converge
    test4_pass = results[3]['converged']  # 3-gram should converge

    all_pass = test1_pass and test2_pass and test3_pass and test4_pass

    if all_pass:
        print("✓ ALL TESTS PASSED")
        print("\nCVFPT is working correctly:")
        print("  - Converges on repeated patterns ✓")
        print("  - Does NOT converge on random sequences ✓")
        print("  - Works for n-gram patterns (n=1,2,3) ✓")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nIssues detected:")
        if not test1_pass:
            print("  ✗ Single token repetition did not converge")
        if not test2_pass:
            print("  ✗ Random sequence converged (should not!)")
        if not test3_pass:
            print("  ✗ 2-gram repetition did not converge")
        if not test4_pass:
            print("  ✗ 3-gram repetition did not converge")

    print("="*80)


if __name__ == "__main__":
    main()
