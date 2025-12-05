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
    include_reversal_pairs: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare training and validation DataLoaders from Pile.

    Args:
        num_samples: Number of samples
        seq_length: Sequence length
        tokenizer_name: Tokenizer name
        val_split: Validation split ratio
        batch_size: Batch size
        include_reversal_pairs: Include forward sentences from reversal pairs in training

    Returns:
        train_loader, val_loader
    """
    from src.utils.data_pythia import load_pile_tokens_cached

    print_flush(f"Preparing data: {num_samples:,} samples, seq_len={seq_length}")

    total_tokens_needed = num_samples * seq_length
    tokens = load_pile_tokens_cached(total_tokens_needed + seq_length, tokenizer_name)

    # Create samples from Pile
    all_input_ids_list = []
    all_labels_list = []

    for i in range(num_samples):
        start = i * seq_length
        input_ids = tokens[start:start + seq_length]
        labels = tokens[start + 1:start + seq_length + 1]
        all_input_ids_list.append(input_ids)
        all_labels_list.append(labels)

    # Stack Pile samples
    pile_input_ids = torch.stack(all_input_ids_list)
    pile_labels = torch.stack(all_labels_list)

    # Split Pile data FIRST (before adding reversal pairs)
    val_size = int(num_samples * val_split)
    train_size = num_samples - val_size

    # Shuffle Pile data
    perm = torch.randperm(num_samples)
    pile_input_ids = pile_input_ids[perm]
    pile_labels = pile_labels[perm]

    train_input = pile_input_ids[:train_size]
    train_labels = pile_labels[:train_size]
    val_input = pile_input_ids[train_size:]
    val_labels = pile_labels[train_size:]

    # Add reversal pairs ONLY to training data (not validation)
    # This prevents data leakage
    if include_reversal_pairs:
        reversal_samples = _create_reversal_training_samples(
            tokenizer_name, seq_length
        )
        if reversal_samples:
            rev_inputs, rev_labels = reversal_samples
            rev_inputs_tensor = torch.stack(rev_inputs)
            rev_labels_tensor = torch.stack(rev_labels)

            train_input = torch.cat([train_input, rev_inputs_tensor], dim=0)
            train_labels = torch.cat([train_labels, rev_labels_tensor], dim=0)
            print_flush(f"  Added {len(rev_inputs)} reversal pair samples to training only")

            # Shuffle training data (with reversal pairs mixed in)
            train_perm = torch.randperm(len(train_input))
            train_input = train_input[train_perm]
            train_labels = train_labels[train_perm]

    train_dataset = TensorDataset(train_input, train_labels)
    val_dataset = TensorDataset(val_input, val_labels)

    print_flush(f"  Train: {len(train_input):,} samples (Pile: {train_size:,})")
    print_flush(f"  Val: {val_size:,} samples (Pile only, no reversal pairs)")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def _create_reversal_training_samples(
    tokenizer_name: str,
    seq_length: int,
    repeat_count: int = 10,
) -> Optional[Tuple[list, list]]:
    """
    Create training samples from reversal pairs (forward direction only).

    各ペアを複数回繰り返して訓練データに含める。

    Args:
        tokenizer_name: Tokenizer name
        seq_length: Sequence length
        repeat_count: Number of times to repeat each sentence

    Returns:
        (input_ids_list, labels_list) or None if import fails
    """
    try:
        from src.data.reversal_pairs import get_training_sentences
        from transformers import AutoTokenizer
    except ImportError:
        return None

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    sentences = get_training_sentences(include_backward=False)  # Forward only

    input_ids_list = []
    labels_list = []

    for sentence in sentences:
        # Tokenize
        tokens = tokenizer.encode(sentence, add_special_tokens=False)

        # Pad or truncate to seq_length
        if len(tokens) < seq_length:
            # Pad with EOS token
            pad_token = tokenizer.eos_token_id or 0
            tokens = tokens + [pad_token] * (seq_length - len(tokens))
        else:
            tokens = tokens[:seq_length]

        tokens_tensor = torch.tensor(tokens)

        # Create input/label pairs (shift by 1)
        input_ids = tokens_tensor[:-1]
        labels = tokens_tensor[1:]

        # Pad to seq_length if needed
        if len(input_ids) < seq_length:
            pad_token = tokenizer.eos_token_id or 0
            pad_len = seq_length - len(input_ids)
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_token)])
            labels = torch.cat([labels, torch.full((pad_len,), pad_token)])

        # Repeat each sentence multiple times for better learning
        for _ in range(repeat_count):
            input_ids_list.append(input_ids[:seq_length])
            labels_list.append(labels[:seq_length])

    return input_ids_list, labels_list


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
    patience: int = 3,
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


def get_tokenizer(tokenizer_name: str) -> Any:
    """
    Get tokenizer by name.

    Args:
        tokenizer_name: Tokenizer name (e.g., "EleutherAI/pythia-70m")

    Returns:
        Tokenizer instance
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_name)


def prepare_long_document_data(
    num_documents: int,
    tokens_per_document: int,
    segment_length: int,
    tokenizer_name: str,
    val_split: float = 0.1,
) -> Tuple[list, list]:
    """
    Prepare long document data for Infini-Attention evaluation.

    Each document is split into multiple segments.
    Memory should be reset at document boundaries, not segment boundaries.

    Args:
        num_documents: Number of documents
        tokens_per_document: Tokens per document (e.g., 4096, 8192)
        segment_length: Segment length (e.g., 256)
        tokenizer_name: Tokenizer name
        val_split: Validation split ratio

    Returns:
        train_documents: List of documents, each is list of (input_ids, labels) segments
        val_documents: List of documents for validation
    """
    from src.utils.data_pythia import load_pile_tokens_cached

    total_tokens_needed = num_documents * tokens_per_document
    tokens = load_pile_tokens_cached(total_tokens_needed + segment_length, tokenizer_name)

    segments_per_document = tokens_per_document // segment_length

    print_flush(f"Preparing long document data:")
    print_flush(f"  Documents: {num_documents}")
    print_flush(f"  Tokens per document: {tokens_per_document:,}")
    print_flush(f"  Segment length: {segment_length}")
    print_flush(f"  Segments per document: {segments_per_document}")

    all_documents = []

    for doc_idx in range(num_documents):
        doc_start = doc_idx * tokens_per_document
        document_segments = []

        for seg_idx in range(segments_per_document):
            seg_start = doc_start + seg_idx * segment_length
            input_ids = tokens[seg_start:seg_start + segment_length]
            labels = tokens[seg_start + 1:seg_start + segment_length + 1]
            document_segments.append((input_ids, labels))

        all_documents.append(document_segments)

    # Split into train/val
    val_size = int(num_documents * val_split)
    train_size = num_documents - val_size

    # Shuffle documents
    perm = torch.randperm(num_documents).tolist()
    all_documents = [all_documents[i] for i in perm]

    train_documents = all_documents[:train_size]
    val_documents = all_documents[train_size:]

    print_flush(f"  Train documents: {len(train_documents)}")
    print_flush(f"  Val documents: {len(val_documents)}")

    return train_documents, val_documents


@torch.no_grad()
def evaluate_long_documents(
    model: nn.Module,
    documents: list,
    device: torch.device,
    update_memory: bool = True,
    has_reset_memory: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate model on long documents with proper memory management.

    For Infini-Attention:
    - Memory is reset at the start of each document
    - Memory is updated after each segment (if update_memory=True)

    For standard models (Pythia):
    - No memory to manage, just evaluate each segment

    Args:
        model: Model to evaluate
        documents: List of documents, each is list of (input_ids, labels) segments
        device: Device
        update_memory: Whether to update memory between segments (Infini only)
        has_reset_memory: Whether model has reset_memory method

    Returns:
        Dict with:
        - total_ppl: Overall perplexity
        - segment_ppls: PPL by segment position (1st, 2nd, 3rd, ...)
        - per_document_ppls: PPL for each document
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    segment_losses = {}  # segment_idx -> (total_loss, total_tokens)
    per_document_ppls = []

    for doc_idx, document in enumerate(documents):
        # Reset memory at document boundary
        if has_reset_memory:
            model.reset_memory()

        doc_loss = 0.0
        doc_tokens = 0

        for seg_idx, (input_ids, labels) in enumerate(document):
            input_ids = input_ids.unsqueeze(0).to(device)  # [1, seq_len]
            labels = labels.unsqueeze(0).to(device)

            # Forward pass
            if has_reset_memory:
                logits = model(input_ids, update_memory=update_memory)
            else:
                logits = model(input_ids)

            # Calculate loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )

            num_tokens = labels.numel()
            total_loss += loss.item()
            total_tokens += num_tokens
            doc_loss += loss.item()
            doc_tokens += num_tokens

            # Track by segment position
            if seg_idx not in segment_losses:
                segment_losses[seg_idx] = (0.0, 0)
            seg_total, seg_count = segment_losses[seg_idx]
            segment_losses[seg_idx] = (seg_total + loss.item(), seg_count + num_tokens)

        # Per-document PPL
        doc_ppl = torch.exp(torch.tensor(doc_loss / doc_tokens)).item()
        per_document_ppls.append(doc_ppl)

    # Overall PPL
    total_ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()

    # Segment-wise PPL
    segment_ppls = {}
    for seg_idx, (seg_loss, seg_tokens) in sorted(segment_losses.items()):
        seg_ppl = torch.exp(torch.tensor(seg_loss / seg_tokens)).item()
        segment_ppls[f"segment_{seg_idx}"] = seg_ppl

    return {
        "total_ppl": total_ppl,
        "segment_ppls": segment_ppls,
        "per_document_ppls": per_document_ppls,
        "num_documents": len(documents),
        "segments_per_document": len(documents[0]) if documents else 0,
    }
