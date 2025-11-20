#!/usr/bin/env python3
"""
New-LLM Training Script - Reconstruction Learning Approach

Pure PyTorch implementation with context reconstruction loss.
"""

import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
import math
import matplotlib.pyplot as plt

from src.models.new_llm import NewLLM
from src.utils.config import NewLLMConfig


class TextDataset(Dataset):
    """Simple text dataset for language modeling"""
    def __init__(self, encodings, max_length=512):
        self.input_ids = encodings['input_ids']
        self.max_length = max_length

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = self.input_ids[idx]
        # Truncate if needed
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        return torch.tensor(ids, dtype=torch.long)


def collate_fn(batch):
    """Collate function with padding"""
    # Find max length in batch
    max_len = max(len(x) for x in batch)

    # Pad sequences
    padded = []
    for x in batch:
        if len(x) < max_len:
            padded.append(torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)]))
        else:
            padded.append(x)

    return torch.stack(padded)


def load_and_tokenize_wikitext(tokenizer, max_samples=None, max_length=512, cache_dir=None):
    """Load and tokenize WikiText-103 dataset efficiently with file caching

    Args:
        cache_dir: Cache directory. If None, uses:
                   1. /content/drive/MyDrive/new-llm-cache/ (if Google Drive mounted)
                   2. /content/wikitext_cache/ (fallback, outside repo)
    """
    import pickle

    # Auto-detect cache location if not specified
    if cache_dir is None:
        if os.path.exists('/content/drive/MyDrive'):
            # Google Drive is mounted
            cache_dir = '/content/drive/MyDrive/new-llm-cache'
            print(f"💾 Using Google Drive cache: {cache_dir}")
        else:
            # Fallback to /content (outside repo, survives rm -rf new-llm)
            cache_dir = '/content/wikitext_cache'
            print(f"💾 Using local cache: {cache_dir}")

    cache_file = f"{cache_dir}/tokenized_wikitext103_gpt2.pkl"

    # Check if tokenized dataset already exists
    if os.path.exists(cache_file):
        cache_size_mb = os.path.getsize(cache_file) / 1024 / 1024
        print(f"\n📂 Loading tokenized dataset from {cache_file} ({cache_size_mb:.1f} MB)...")
        print("⏳ This may take 2-3 minutes for large datasets. Please wait...")

        import time
        start_time = time.time()
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        load_time = time.time() - start_time

        train_encodings = cached_data['train']
        val_encodings = cached_data['val']

        print(f"✓ Loaded {len(train_encodings['input_ids'])} train, {len(val_encodings['input_ids'])} val samples from cache in {load_time:.1f}s")

        # Apply max_samples AFTER loading from cache
        if max_samples:
            print(f"\n✂️  Applying max_samples={max_samples}...")
            train_encodings['input_ids'] = train_encodings['input_ids'][:max_samples]
            val_encodings['input_ids'] = val_encodings['input_ids'][:max_samples // 10]
            print(f"✓ Using {len(train_encodings['input_ids'])} train, {len(val_encodings['input_ids'])} val samples")

        return train_encodings, val_encodings

    # First time: tokenize and save
    print("\n📥 Loading WikiText dataset (first time, will be cached)...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    train_data = dataset['train']
    val_data = dataset['validation']

    print(f"✓ Loaded {len(train_data)} train, {len(val_data)} val samples")

    # Tokenize using HuggingFace map() for parallel processing
    print("\n⚙️  Tokenizing FULL dataset (this will be cached for future use)...")

    def tokenize_function(examples):
        """Tokenize a batch of examples"""
        return tokenizer(
            examples['text'],
            max_length=max_length,
            truncation=True,
            padding=False,  # We'll pad in collate_fn
        )

    # Parallel tokenization
    train_tokenized = train_data.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=['text'],
        desc="Tokenizing train"
    )

    val_tokenized = val_data.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=['text'],
        desc="Tokenizing val"
    )

    # Filter out sequences that are too short
    train_tokenized = train_tokenized.filter(lambda x: len(x['input_ids']) >= 2)
    val_tokenized = val_tokenized.filter(lambda x: len(x['input_ids']) >= 2)

    print(f"✓ Tokenized {len(train_tokenized)} train, {len(val_tokenized)} val samples")

    # Convert to Python lists (necessary for proper pickling)
    print("\n📦 Converting to Python lists for caching...")
    print("⏳ This may take 3-5 minutes for large datasets. Please wait...")

    from tqdm import tqdm
    train_encodings = {'input_ids': [x for x in tqdm(train_tokenized['input_ids'], desc="Converting train")]}
    val_encodings = {'input_ids': [x for x in tqdm(val_tokenized['input_ids'], desc="Converting val")]}

    # Save to cache file
    print(f"\n💾 Saving tokenized dataset to {cache_file}...")
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump({
            'train': train_encodings,
            'val': val_encodings
        }, f)
    print(f"✓ Cached for future use (file size: {os.path.getsize(cache_file) / 1024 / 1024:.1f} MB)")

    # Apply max_samples
    if max_samples:
        print(f"\n✂️  Applying max_samples={max_samples}...")
        train_encodings['input_ids'] = train_encodings['input_ids'][:max_samples]
        val_encodings['input_ids'] = val_encodings['input_ids'][:max_samples // 10]
        print(f"✓ Using {len(train_encodings['input_ids'])} train, {len(val_encodings['input_ids'])} val samples")

    return train_encodings, val_encodings


def _compute_batch_metrics(model, input_ids, device, context_loss_weight=1.0):
    """
    Compute metrics for a single batch (shared between train and eval)

    Returns:
        dict with keys: loss, token_loss, recon_loss, recon_accuracy, context_change,
                       shift_logits, shift_labels
    """
    batch_size, seq_len = input_ids.shape

    # Skip if too short
    if seq_len < 2:
        return None

    # Forward pass
    logits, context_trajectory = model(input_ids)

    # Token prediction loss
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=0,
        reduction='mean'
    )

    # Reconstruction loss
    recon_loss = torch.tensor(0.0, device=device)
    recon_accuracy = 0.0

    if hasattr(model, 'reconstruction_targets') and model.reconstruction_targets is not None:
        recon_targets = model.reconstruction_targets
        batch_size, seq_len, context_dim = context_trajectory.shape

        # Decode context vectors
        flat_context = context_trajectory.view(-1, context_dim)
        reconstructed = model.context_decoder(flat_context)
        reconstructed = reconstructed.view(batch_size, seq_len, -1)

        # MSE loss
        recon_loss = F.mse_loss(reconstructed, recon_targets)

        # Reconstruction accuracy (cosine similarity)
        recon_flat = reconstructed.view(-1, reconstructed.size(-1))
        target_flat = recon_targets.view(-1, recon_targets.size(-1))
        recon_accuracy = F.cosine_similarity(recon_flat, target_flat, dim=-1).mean().item()

    # Combined loss
    loss = token_loss + context_loss_weight * recon_loss

    # Calculate context change
    context_change = 0.0
    if context_trajectory.size(1) > 1:
        context_diff = context_trajectory[:, 1:, :] - context_trajectory[:, :-1, :]
        context_change = torch.norm(context_diff, dim=-1).mean().item()

    return {
        'loss': loss,
        'token_loss': token_loss,
        'recon_loss': recon_loss,
        'recon_accuracy': recon_accuracy,
        'context_change': context_change,
        'shift_logits': shift_logits,
        'shift_labels': shift_labels,
    }


def train_epoch(model, dataloader, optimizer, device, context_loss_weight=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_token_loss = 0
    total_recon_loss = 0
    total_context_change = 0
    total_recon_accuracy = 0
    num_batches = 0

    for batch_idx, input_ids in enumerate(dataloader):
        input_ids = input_ids.to(device)

        # Compute batch metrics (shared logic)
        metrics = _compute_batch_metrics(model, input_ids, device, context_loss_weight)
        if metrics is None:
            continue

        # Backward pass (train-specific)
        optimizer.zero_grad()
        metrics['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate stats
        total_loss += metrics['loss'].item()
        total_token_loss += metrics['token_loss'].item()
        total_recon_loss += metrics['recon_loss'].item()
        total_recon_accuracy += metrics['recon_accuracy']
        total_context_change += metrics['context_change']
        num_batches += 1

    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0,
        'token_loss': total_token_loss / num_batches if num_batches > 0 else 0,
        'recon_loss': total_recon_loss / num_batches if num_batches > 0 else 0,
        'recon_accuracy': total_recon_accuracy / num_batches if num_batches > 0 else 0,
        'context_change': total_context_change / num_batches if num_batches > 0 else 0,
    }


def evaluate(model, dataloader, device, context_loss_weight=1.0):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_token_loss = 0
    total_recon_loss = 0
    total_recon_cosine_sim = 0
    total_context_change = 0
    correct = 0
    total_tokens = 0
    num_batches = 0
    with torch.no_grad():
        for input_ids in dataloader:
            input_ids = input_ids.to(device)

            # Compute batch metrics (shared logic)
            metrics = _compute_batch_metrics(model, input_ids, device, context_loss_weight)
            if metrics is None:
                continue

            # Accuracy calculation (eval-specific)
            predictions = metrics['shift_logits'].argmax(dim=-1)
            mask = metrics['shift_labels'] != 0
            correct += ((predictions == metrics['shift_labels']) & mask).sum().item()
            total_tokens += mask.sum().item()

            # Accumulate stats
            total_loss += metrics['loss'].item()
            total_token_loss += metrics['token_loss'].item()
            total_recon_loss += metrics['recon_loss'].item()
            total_recon_cosine_sim += metrics['recon_accuracy']
            total_context_change += metrics['context_change']
            num_batches += 1

    accuracy = correct / total_tokens if total_tokens > 0 else 0
    avg_token_loss = total_token_loss / num_batches if num_batches > 0 else 0
    perplexity = math.exp(min(avg_token_loss, 20))  # Cap at 20 to avoid overflow
    avg_context_change = total_context_change / num_batches if num_batches > 0 else 0
    avg_recon_accuracy = total_recon_cosine_sim / num_batches if num_batches > 0 else 0

    return {
        'loss': total_loss / num_batches if num_batches > 0 else 0,
        'token_loss': avg_token_loss,
        'recon_loss': total_recon_loss / num_batches if num_batches > 0 else 0,
        'recon_accuracy': avg_recon_accuracy,
        'perplexity': perplexity,
        'accuracy': accuracy,
        'context_change': avg_context_change,
    }


def main():
    parser = argparse.ArgumentParser(description='Train New-LLM with reconstruction learning')
    parser.add_argument('--max-samples', type=int, default=None, help='Max training samples')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--context-loss-weight', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--layers', type=int, default=1, help='Number of FNN layers')
    parser.add_argument('--context-dim', type=int, default=256, help='Context vector dimension')
    parser.add_argument('--context-update-strategy', type=str, default='simple', choices=['simple', 'gated'], help='Context update strategy')
    parser.add_argument('--max-length', type=int, default=256, help='Maximum sequence length')
    parser.add_argument('--output-dir', type=str, default='./experiments', help='Output directory')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')

    args = parser.parse_args()

    print("=" * 80)
    print("New-LLM Training - Reconstruction Learning")
    print("=" * 80)
    print("\n📋 Training Parameters:")
    print(f"  Dataset: WikiText-103")
    print(f"  Max samples: {args.max_samples if args.max_samples else 'All'}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Context loss weight: {args.context_loss_weight}")
    print(f"  FNN layers: {args.layers}")
    print(f"  Context vector dim: {args.context_dim}")
    print(f"  Context update strategy: {args.context_update_strategy}")
    print(f"  Device: {args.device}")
    print(f"  Output directory: {args.output_dir}")
    print()

    # Load GPT-2 tokenizer (pretrained, no training needed)
    print("\n🔤 Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have pad token by default
    print(f"✓ Tokenizer loaded: {len(tokenizer):,} tokens")

    # Load and tokenize data (efficient with caching)
    train_encodings, val_encodings = load_and_tokenize_wikitext(
        tokenizer,
        max_samples=args.max_samples,
        max_length=args.max_length
    )

    # Create datasets
    print(f"\n📊 Creating TextDataset objects (max_length={args.max_length})...")
    train_dataset = TextDataset(train_encodings, max_length=args.max_length)
    val_dataset = TextDataset(val_encodings, max_length=args.max_length)
    print(f"✓ Created train dataset ({len(train_dataset)} sequences)")
    print(f"✓ Created val dataset ({len(val_dataset)} sequences)")

    print(f"\n⚙️  Creating DataLoaders (batch_size={args.batch_size})...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)
    print(f"✓ DataLoaders ready")

    print(f"\n✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    # Create model
    print(f"\n🧠 Creating model...")
    config = NewLLMConfig()
    config.vocab_size = len(tokenizer)  # GPT-2 has 50,257 tokens
    # Use config defaults for embed_dim (256) and hidden_dim (1024)
    # These can be changed in src/utils/config.py if needed
    config.context_vector_dim = args.context_dim
    config.context_update_strategy = args.context_update_strategy  # "simple" or "gated"
    config.num_layers = args.layers
    config.max_seq_length = args.max_length
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id else tokenizer.eos_token_id
    config.eos_token_id = tokenizer.eos_token_id

    device = torch.device(args.device)
    print(f"✓ Using device: {device}")

    # Create model (context update strategy is controlled by config.context_update_strategy)
    print(f"\n🏗️  Building model architecture...")
    model = NewLLM(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params:,} parameters")

    print(f"\n📲 Moving model to {device}...")
    model = model.to(device)
    print(f"✓ Model ready on {device}")

    # Optimizer
    print(f"\n⚙️  Creating optimizer (lr={args.lr})...")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"✓ Optimizer ready")

    # Training loop
    print(f"\n{'='*80}")
    print("🚀 Ready to Start Training")
    print("=" * 80)
    print(f"📊 Dataset: {len(train_dataset)} train, {len(val_dataset)} val sequences")
    print(f"📦 Batches: {len(train_loader)} train, {len(val_loader)} val batches")
    print(f"🧠 Model: {total_params:,} parameters on {device}")
    print(f"⚙️  Config: lr={args.lr}, batch={args.batch_size}, epochs={args.epochs}")
    print(f"📏 Sequence: max_length={args.max_length}, context_dim={args.context_dim}")
    print("=" * 80)
    print(f"\n🏁 Starting training...\n")

    # Store metrics for plotting
    history = {
        'train_loss': [],
        'train_token_loss': [],
        'train_recon_loss': [],
        'train_recon_accuracy': [],
        'train_context_change': [],
        'val_loss': [],
        'val_token_loss': [],
        'val_recon_loss': [],
        'val_recon_accuracy': [],
        'val_perplexity': [],
        'val_accuracy': [],
        'val_context_change': [],
    }

    for epoch in range(args.epochs):
        # Progress display - Starting
        progress_pct = (epoch) / args.epochs * 100
        print(f"\n{'='*80}")
        print(f"🔄 Epoch {epoch + 1}/{args.epochs} - Starting... ({progress_pct:.0f}% complete)")
        print("=" * 80)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, args.context_loss_weight)

        # Evaluate
        val_metrics = evaluate(model, val_loader, device, args.context_loss_weight)

        # Store metrics
        history['train_loss'].append(train_metrics['loss'])
        history['train_token_loss'].append(train_metrics['token_loss'])
        history['train_recon_loss'].append(train_metrics['recon_loss'])
        history['train_recon_accuracy'].append(train_metrics['recon_accuracy'])
        history['train_context_change'].append(train_metrics['context_change'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_token_loss'].append(val_metrics['token_loss'])
        history['val_recon_loss'].append(val_metrics['recon_loss'])
        history['val_recon_accuracy'].append(val_metrics['recon_accuracy'])
        history['val_perplexity'].append(val_metrics['perplexity'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_context_change'].append(val_metrics['context_change'])

        # Quick summary (1 line)
        progress_pct = (epoch + 1) / args.epochs * 100
        print(f"\n✓ Epoch {epoch + 1}/{args.epochs} DONE - "
              f"PPL: {val_metrics['perplexity']:.2f}, "
              f"Loss: {val_metrics['loss']:.2f}, "
              f"Acc: {val_metrics['accuracy']:.2%}, "
              f"Recon Acc: {val_metrics['recon_accuracy']:.2%} "
              f"({progress_pct:.0f}% complete)")

        # Print detailed results
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1} Results (Detail)")
        print("=" * 80)
        print(f"Train - Loss: {train_metrics['loss']:.4f} | "
              f"Token: {train_metrics['token_loss']:.4f} | "
              f"Recon: {train_metrics['recon_loss']:.4f} | "
              f"ReconAcc: {train_metrics['recon_accuracy']:.2%} | "
              f"CtxΔ: {train_metrics['context_change']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | "
              f"Token: {val_metrics['token_loss']:.4f} | "
              f"Recon: {val_metrics['recon_loss']:.4f} | "
              f"ReconAcc: {val_metrics['recon_accuracy']:.2%} | "
              f"CtxΔ: {val_metrics['context_change']:.4f}")
        print(f"Val   - PPL: {val_metrics['perplexity']:.2f} | "
              f"Acc: {val_metrics['accuracy']:.2%}")
        print("=" * 80)

    # Save model
    print(f"\n💾 Saving model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
    }, f"{args.output_dir}/final_model.pt")

    # Log experiment parameters and results
    from datetime import datetime
    log_path = f"{args.output_dir}/experiment_log.txt"
    with open(log_path, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Experiment: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Parameters:\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  FNN layers: {args.layers}\n")
        f.write(f"  Context vector dim: {args.context_dim}\n")
        f.write(f"  Max samples: {args.max_samples if args.max_samples else 'All'}\n")
        f.write(f"  Device: {args.device}\n")
        f.write(f"  Context update strategy: {config.context_update_strategy}\n")
        f.write(f"\nFinal Results:\n")
        f.write(f"  Train Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"  Val Loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"  Val PPL: {history['val_perplexity'][-1]:.2f}\n")
        f.write(f"  Val Accuracy: {history['val_accuracy'][-1]:.2%}\n")
        f.write(f"  Train Context Change: {history['train_context_change'][-1]:.4f}\n")
        f.write(f"  Val Context Change: {history['val_context_change'][-1]:.4f}\n")
        f.write(f"  Model params: {total_params:,}\n")
        f.write(f"\nBest Results:\n")
        best_ppl_idx = history['val_perplexity'].index(min(history['val_perplexity']))
        f.write(f"  Best PPL: {history['val_perplexity'][best_ppl_idx]:.2f} (Epoch {best_ppl_idx + 1})\n")
        f.write(f"  Best Accuracy: {max(history['val_accuracy']):.2%}\n")
        f.write(f"{'='*80}\n")
    print(f"✓ Experiment log saved to {log_path}")

    # Plot training curves
    print(f"\n📊 Generating training curves...")
    epochs = range(1, args.epochs + 1)

    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle('New-LLM Training - Reconstruction Learning', fontsize=16, fontweight='bold')

    # Plot 1: Total Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss (Token + Reconstruction)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Perplexity
    axes[0, 1].plot(epochs, history['val_perplexity'], 'g-^', label='Val Perplexity', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Perplexity')
    axes[0, 1].set_title('Validation Perplexity')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Accuracy
    axes[1, 0].plot(epochs, [acc * 100 for acc in history['val_accuracy']], 'purple', marker='d', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Validation Accuracy')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Reconstruction Accuracy
    axes[1, 1].plot(epochs, [acc * 100 for acc in history['train_recon_accuracy']], 'b-o', label='Train Recon Acc', linewidth=2)
    axes[1, 1].plot(epochs, [acc * 100 for acc in history['val_recon_accuracy']], 'r-s', label='Val Recon Acc', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Reconstruction Accuracy (%)')
    axes[1, 1].set_title('Context Vector Reconstruction Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 5: Context Vector Change
    axes[2, 0].plot(epochs, history['train_context_change'], 'b-o', label='Train Ctx Change', linewidth=2)
    axes[2, 0].plot(epochs, history['val_context_change'], 'r-s', label='Val Ctx Change', linewidth=2)
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Context Change (L2 norm)')
    axes[2, 0].set_title('Context Vector Change per Timestep')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Token Loss vs Reconstruction Loss
    axes[2, 1].plot(epochs, history['train_token_loss'], 'b-o', label='Train Token Loss', linewidth=2)
    axes[2, 1].plot(epochs, history['train_recon_loss'], 'r-s', label='Train Recon Loss', linewidth=2)
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Loss')
    axes[2, 1].set_title('Token Loss vs Reconstruction Loss')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"{args.output_dir}/training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {plot_path}")

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
