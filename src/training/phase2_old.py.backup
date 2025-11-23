"""
Phase 2: Token Prediction Training

Train token_output layer using fixed contexts from Phase 1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def train_phase2(model, train_token_ids, val_token_ids, train_contexts, val_contexts, config, device):
    """
    Phase 2: Train token prediction using fixed contexts.

    Args:
        model: The language model
        train_token_ids: Training token IDs
        val_token_ids: Validation token IDs
        train_contexts: Fixed contexts from Phase 1 (train)
        val_contexts: Fixed contexts from Phase 1 (val)
        config: Configuration object
        device: torch device
    """
    print_flush(f"\n{'='*70}")
    print_flush("PHASE 2: Token Prediction Training")
    print_flush(f"{'='*70}")

    model.to(device)
    model.train()

    # Setup parameters
    if config.freeze_context:
        # Only train token_output layer
        for name, param in model.named_parameters():
            if 'token_output' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        params_to_train = [p for p in model.parameters() if p.requires_grad]
        print_flush("  Training: token_output layer only (context frozen)")
    else:
        # Train all parameters
        for param in model.parameters():
            param.requires_grad = True
        params_to_train = model.parameters()
        print_flush("  Training: all parameters")

    optimizer = torch.optim.Adam(params_to_train, lr=config.phase2_learning_rate)

    # Prepare data loaders
    train_dataset = create_sequence_dataset(train_token_ids, train_contexts, device)
    val_dataset = create_sequence_dataset(val_token_ids, val_contexts, device)

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

    best_val_loss = float('inf')

    # Training loop
    for epoch in range(config.phase2_epochs):
        # Train
        model.train()
        train_loss = 0
        train_batches = 0

        for batch_contexts, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.phase2_epochs}"):
            optimizer.zero_grad()

            # Forward pass: predict next token from context
            logits = model.token_output(batch_contexts)
            loss = F.cross_entropy(logits, batch_targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if config.phase2_gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(params_to_train, config.phase2_gradient_clip)

            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch_contexts, batch_targets in val_loader:
                logits = model.token_output(batch_contexts)
                loss = F.cross_entropy(logits, batch_targets)

                val_loss += loss.item()
                val_batches += 1

                # Calculate accuracy
                predictions = torch.argmax(logits, dim=1)
                correct_predictions += (predictions == batch_targets).sum().item()
                total_predictions += batch_targets.size(0)

        avg_val_loss = val_loss / val_batches
        val_accuracy = correct_predictions / total_predictions

        # Log progress
        print_flush(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy*100:.2f}%")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, config, epoch, best_val_loss, "best_phase2_model.pt")
            print_flush(f"  → New best validation loss: {best_val_loss:.4f}")

    print_flush(f"\nPhase 2 Training Complete")
    print_flush(f"  Best validation loss: {best_val_loss:.4f}")


def create_sequence_dataset(token_ids, contexts, device):
    """
    Create dataset for sequence modeling.

    Args:
        token_ids: Token IDs [num_tokens]
        contexts: Fixed contexts [num_tokens, context_dim]
        device: torch device

    Returns:
        TensorDataset: Dataset with (context, next_token) pairs
    """
    # Create input-target pairs
    # Context at position t predicts token at position t+1
    input_contexts = contexts[:-1]  # All but last
    target_tokens = token_ids[1:]   # All but first

    return TensorDataset(
        input_contexts.to(device),
        target_tokens.to(device)
    )


def save_checkpoint(model, config, epoch, loss, filename):
    """Save model checkpoint"""
    import os

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
        'config': config
    }

    save_path = os.path.join(config.cache_dir, filename)
    torch.save(checkpoint, save_path)
    print_flush(f"  Checkpoint saved: {save_path}")


def print_flush(msg):
    """Print with immediate flush"""
    print(msg, flush=True)