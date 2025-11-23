"""
Phase 2: Token Prediction Training with Multi-Output Architecture

Trains per-block token outputs initialized from Phase 1's trained output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def train_phase2_multioutput(phase2_model, train_token_ids, val_token_ids, config, device):
    """
    Phase 2: Train multi-output token prediction

    Args:
        phase2_model: NewLLMPhase2 model (expanded from Phase 1)
        train_token_ids: Training token IDs [num_tokens]
        val_token_ids: Validation token IDs [num_tokens]
        config: Configuration object
        device: torch device
    """
    print_flush(f"\n{'='*70}")
    print_flush("PHASE 2: Multi-Output Token Prediction Training")
    print_flush(f"{'='*70}")

    phase2_model.to(device)
    phase2_model.train()

    # Setup parameters
    if config.freeze_context:
        # Only train block_outputs
        for name, param in phase2_model.named_parameters():
            if 'block_outputs' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        params_to_train = [p for p in phase2_model.parameters() if p.requires_grad]
        print_flush("  Training: block_outputs only (context frozen)")
    else:
        # Train all trainable parameters (excluding frozen embeddings)
        for param in phase2_model.parameters():
            if param.requires_grad:  # Respect existing frozen params (e.g., embeddings)
                param.requires_grad = True
        params_to_train = [p for p in phase2_model.parameters() if p.requires_grad]
        print_flush("  Training: all parameters (embeddings remain frozen)")

    trainable_params = sum(p.numel() for p in params_to_train)
    print_flush(f"  Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.Adam(params_to_train, lr=config.phase2_learning_rate)

    # Prepare data loaders
    train_dataset = create_sequence_dataset_phase2(train_token_ids, device)
    val_dataset = create_sequence_dataset_phase2(val_token_ids, device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.phase2_batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.phase2_batch_size,
        shuffle=False
    )

    print_flush(f"  Batch size: {config.phase2_batch_size}")
    print_flush(f"  Learning rate: {config.phase2_learning_rate}")
    print_flush(f"  Epochs: {config.phase2_epochs}")
    print_flush(f"  Block output weights: {config.phase2_block_weights}")

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(config.phase2_epochs):
        # Train
        phase2_model.train()
        train_loss = 0
        train_batches = 0

        for batch_input_ids, batch_target_ids in train_loader:
            optimizer.zero_grad()

            # Forward pass: get logits from all blocks
            # block_logits: [num_blocks, batch, seq_len, vocab_size]
            block_logits = phase2_model(batch_input_ids, return_all_logits=True)

            # Compute weighted loss across all blocks
            total_loss = compute_multiblock_loss(
                block_logits,
                batch_target_ids,
                config.phase2_block_weights
            )

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            if config.phase2_gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(params_to_train, config.phase2_gradient_clip)

            optimizer.step()

            train_loss += total_loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # Validation
        phase2_model.eval()
        val_loss = 0
        val_batches = 0
        block_accuracies = None

        with torch.no_grad():
            for batch_input_ids, batch_target_ids in val_loader:
                block_logits = phase2_model(batch_input_ids, return_all_logits=True)

                # Loss
                total_loss = compute_multiblock_loss(
                    block_logits,
                    batch_target_ids,
                    config.phase2_block_weights
                )
                val_loss += total_loss.item()
                val_batches += 1

                # Accuracy per block
                batch_accuracies = compute_block_accuracies(block_logits, batch_target_ids)
                if block_accuracies is None:
                    block_accuracies = batch_accuracies
                else:
                    block_accuracies += batch_accuracies

        avg_val_loss = val_loss / val_batches
        block_accuracies = block_accuracies / val_batches

        # Log progress
        acc_str = " | ".join([f"B{i+1}:{acc*100:.1f}%" for i, acc in enumerate(block_accuracies)])
        print_flush(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f} | {acc_str}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint_phase2(phase2_model, config, epoch, best_val_loss)
            print_flush(f"  → New best validation loss: {best_val_loss:.4f}")

    print_flush(f"\nPhase 2 Training Complete")
    print_flush(f"  Best validation loss: {best_val_loss:.4f}")


def compute_multiblock_loss(block_logits, target_ids, block_weights):
    """
    Compute weighted loss across all blocks

    Args:
        block_logits: [num_blocks, batch, seq_len, vocab_size]
        target_ids: [batch, seq_len]
        block_weights: List of weights for each block

    Returns:
        total_loss: Weighted sum of losses
    """
    num_blocks = block_logits.size(0)
    total_loss = 0

    for block_idx in range(num_blocks):
        logits = block_logits[block_idx]  # [batch, seq_len, vocab_size]

        # Reshape for cross_entropy
        logits_flat = logits.reshape(-1, logits.size(-1))  # [batch*seq_len, vocab_size]
        targets_flat = target_ids.reshape(-1)  # [batch*seq_len]

        loss = F.cross_entropy(logits_flat, targets_flat)
        total_loss += block_weights[block_idx] * loss

    return total_loss


def compute_block_accuracies(block_logits, target_ids):
    """
    Compute accuracy for each block

    Args:
        block_logits: [num_blocks, batch, seq_len, vocab_size]
        target_ids: [batch, seq_len]

    Returns:
        accuracies: [num_blocks] tensor of accuracies
    """
    num_blocks = block_logits.size(0)
    accuracies = []

    for block_idx in range(num_blocks):
        logits = block_logits[block_idx]  # [batch, seq_len, vocab_size]
        predictions = torch.argmax(logits, dim=-1)  # [batch, seq_len]

        correct = (predictions == target_ids).sum().item()
        total = target_ids.numel()
        accuracy = correct / total

        accuracies.append(accuracy)

    return torch.tensor(accuracies)


def create_sequence_dataset_phase2(token_ids, device):
    """
    Create dataset for sequence modeling

    Args:
        token_ids: Token IDs [num_tokens]
        device: torch device

    Returns:
        TensorDataset: Dataset with (input_sequence, target_sequence) pairs
    """
    # For now, create simple pairs: input[:-1], target[1:]
    # This can be extended to longer sequences
    input_ids = token_ids[:-1].unsqueeze(0)  # [1, seq_len-1]
    target_ids = token_ids[1:].unsqueeze(0)  # [1, seq_len-1]

    return TensorDataset(
        input_ids.to(device),
        target_ids.to(device)
    )


def save_checkpoint_phase2(model, config, epoch, loss):
    """Save Phase 2 model checkpoint"""
    import os

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'phase': 2
    }

    save_path = os.path.join(config.checkpoint_dir, "model_phase2_latest.pt")
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    torch.save(checkpoint, save_path)
    print_flush(f"  Phase 2 checkpoint saved: {save_path}")


def print_flush(msg):
    """Print with immediate flush"""
    print(msg, flush=True)
