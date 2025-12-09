"""
Training utilities for experiments.

共通の学習・評価関数を提供する。
"""

import time
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils.io import print_flush
from src.utils.device import clear_gpu_cache


def prepare_data_loaders(
    num_samples: int,
    seq_length: int,
    tokenizer_name: str,
    val_split: float = 0.1,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare training and validation DataLoaders from Japanese Wikipedia.

    Args:
        num_samples: Number of samples
        seq_length: Sequence length
        tokenizer_name: Tokenizer name
        val_split: Validation split ratio
        batch_size: Batch size

    Returns:
        train_loader, val_loader
    """
    from src.utils.data_utils import load_wiki_ja_tokens_cached

    print_flush(f"Preparing data: {num_samples:,} samples, seq_len={seq_length}")

    total_tokens_needed = num_samples * seq_length
    tokens = load_wiki_ja_tokens_cached(total_tokens_needed + seq_length, tokenizer_name)

    # Create samples from Wikipedia
    all_input_ids_list = []
    all_labels_list = []

    for i in range(num_samples):
        start = i * seq_length
        input_ids = tokens[start:start + seq_length]
        labels = tokens[start + 1:start + seq_length + 1]
        all_input_ids_list.append(input_ids)
        all_labels_list.append(labels)

    # Stack samples
    wiki_input_ids = torch.stack(all_input_ids_list)
    wiki_labels = torch.stack(all_labels_list)

    # Split data
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size

    # Shuffle data
    perm = torch.randperm(num_samples)
    wiki_input_ids = wiki_input_ids[perm]
    wiki_labels = wiki_labels[perm]

    train_input = wiki_input_ids[:train_size]
    train_labels = wiki_labels[:train_size]
    val_input = wiki_input_ids[train_size:]
    val_labels = wiki_labels[train_size:]

    train_dataset = TensorDataset(train_input, train_labels)
    val_dataset = TensorDataset(val_input, val_labels)

    print_flush(f"  Train: {len(train_input):,} samples")
    print_flush(f"  Val: {val_size:,} samples")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    grad_clip: float = 1.0,
) -> float:
    """
    Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device
        criterion: Loss function (default: CrossEntropyLoss)
        grad_clip: Gradient clipping value

    Returns:
        Average loss
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    for input_ids, labels in train_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        del input_ids, labels, logits
        clear_gpu_cache(device)

    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Tuple[float, float]:
    """
    Evaluate model.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device
        criterion: Loss function (default: CrossEntropyLoss)

    Returns:
        avg_loss, perplexity
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    for input_ids, labels in val_loader:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        logits = model(input_ids)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        total_loss += loss.item()
        num_batches += 1

        del input_ids, labels, logits
        clear_gpu_cache(device)

    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return avg_loss, perplexity


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    patience: int = 1,
    model_name: str = "Model",
) -> Dict[str, Any]:
    """
    Train model with early stopping.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device
        num_epochs: Number of epochs
        learning_rate: Learning rate
        patience: Early stopping patience
        model_name: Model name for logging

    Returns:
        Results dict with best_val_ppl, best_epoch, history
    """
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_flush(f"  Trainable: {trainable_params:,} / {total_params:,} parameters")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_ppl = evaluate(model, val_loader, device)

        train_ppl = torch.exp(torch.tensor(train_loss)).item()
        epoch_time = time.time() - start_time

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        print_flush(
            f"  Epoch {epoch:2d}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
            f"[{epoch_time:.1f}s]{marker}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_ppl": train_ppl,
            "val_loss": val_loss,
            "val_ppl": val_ppl,
        })

        if patience_counter >= patience:
            print_flush(f"  → Early stop: val_ppl worsened for {patience} epochs")
            break

    best_val_ppl = torch.exp(torch.tensor(best_val_loss)).item()
    print_flush(f"  Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}")

    return {
        "best_epoch": best_epoch,
        "best_val_ppl": best_val_ppl,
        "history": history,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }


def get_device() -> torch.device:
    """Get device and print info."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_flush(f"Device: cuda ({gpu_name}, {gpu_mem:.1f}GB)")
    else:
        device = torch.device("cpu")
        print_flush("Device: cpu")
    return device
