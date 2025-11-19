#!/usr/bin/env python3
"""
New-LLM Training Script - Reconstruction Learning Approach

Pure PyTorch implementation with context reconstruction loss.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import math
import matplotlib.pyplot as plt

from src.models.context_vector_llm import ContextVectorLLM
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


def load_wikitext_data(max_samples=None):
    """Load WikiText-103 dataset"""
    print("\n📥 Loading WikiText dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    train_data = dataset['train']
    val_data = dataset['validation']

    if max_samples:
        train_data = train_data.select(range(min(max_samples, len(train_data))))
        val_data = val_data.select(range(min(max_samples // 10, len(val_data))))

    # Extract non-empty texts
    train_texts = [ex['text'] for ex in train_data if ex['text'].strip()]
    val_texts = [ex['text'] for ex in val_data if ex['text'].strip()]

    print(f"✓ Loaded {len(train_texts)} train, {len(val_texts)} val texts")

    return train_texts, val_texts


def create_tokenizer(texts, vocab_size=10000, output_dir='./tokenizer'):
    """Create BPE tokenizer"""
    print(f"\n🔤 Training BPE tokenizer (vocab_size={vocab_size})...")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"],
        min_frequency=2
    )

    tokenizer.train_from_iterator(texts, trainer)

    # Save tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(f"{output_dir}/tokenizer.json")

    print(f"✓ Tokenizer created: {tokenizer.get_vocab_size()} tokens")
    print(f"✓ Saved to {output_dir}/tokenizer.json")

    return tokenizer


def tokenize_texts(texts, tokenizer, max_length=512):
    """Tokenize texts"""
    print("\n⚙️  Tokenizing texts...")

    encodings = {'input_ids': []}

    for text in texts:
        encoding = tokenizer.encode(text)
        ids = encoding.ids

        # Skip if too short
        if len(ids) < 2:
            continue

        encodings['input_ids'].append(ids)

    print(f"✓ Tokenized {len(encodings['input_ids'])} texts")

    return encodings


def train_epoch(model, dataloader, optimizer, device, context_loss_weight=1.0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_token_loss = 0
    total_recon_loss = 0
    total_tokens = 0
    total_context_change = 0
    num_context_samples = 0

    for batch_idx, input_ids in enumerate(dataloader):
        input_ids = input_ids.to(device)
        batch_size, seq_len = input_ids.shape

        # Skip if too short
        if seq_len < 2:
            continue

        # Forward pass
        logits, context_trajectory = model(input_ids)

        # Token prediction loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=0,  # Ignore padding
            reduction='mean'
        )

        # Reconstruction loss
        if hasattr(model, 'reconstruction_targets') and model.reconstruction_targets is not None:
            # Get context trajectory and reconstruction targets
            # context_trajectory: (batch, seq_len, context_dim)
            # reconstruction_targets: (batch, seq_len, context_dim + embed_dim)

            recon_targets = model.reconstruction_targets
            batch_size, seq_len, context_dim = context_trajectory.shape

            # Decode context vectors
            flat_context = context_trajectory.view(-1, context_dim)
            reconstructed = model.context_decoder(flat_context)
            reconstructed = reconstructed.view(batch_size, seq_len, -1)

            # MSE loss
            recon_loss = F.mse_loss(reconstructed, recon_targets)
        else:
            recon_loss = torch.tensor(0.0, device=device)

        # Combined loss
        loss = token_loss + context_loss_weight * recon_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Calculate context change (how much context vector changes per timestep)
        if context_trajectory.size(1) > 1:
            # context_trajectory: (batch, seq_len, context_dim)
            # Calculate ||context[t] - context[t-1]|| for each t
            context_diff = context_trajectory[:, 1:, :] - context_trajectory[:, :-1, :]
            context_change = torch.norm(context_diff, dim=-1)  # (batch, seq_len-1)
            avg_context_change = context_change.mean().item()
            total_context_change += avg_context_change
            num_context_samples += 1

        # Stats
        total_loss += loss.item()
        total_token_loss += token_loss.item()
        total_recon_loss += recon_loss.item()
        total_tokens += shift_labels.numel()

    avg_context_change = total_context_change / num_context_samples if num_context_samples > 0 else 0

    return {
        'loss': total_loss / len(dataloader),
        'token_loss': total_token_loss / len(dataloader),
        'recon_loss': total_recon_loss / len(dataloader),
        'context_change': avg_context_change,
    }


def evaluate(model, dataloader, device, context_loss_weight=1.0):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_token_loss = 0
    total_recon_loss = 0
    total_tokens = 0
    correct = 0
    total_context_change = 0
    num_context_samples = 0

    with torch.no_grad():
        for input_ids in dataloader:
            input_ids = input_ids.to(device)
            batch_size, seq_len = input_ids.shape

            if seq_len < 2:
                continue

            # Forward
            logits, context_trajectory = model(input_ids)

            # Token loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            token_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=0,
                reduction='mean'
            )

            # Reconstruction loss
            if hasattr(model, 'reconstruction_targets') and model.reconstruction_targets is not None:
                recon_targets = model.reconstruction_targets
                batch_size, seq_len, context_dim = context_trajectory.shape

                flat_context = context_trajectory.view(-1, context_dim)
                reconstructed = model.context_decoder(flat_context)
                reconstructed = reconstructed.view(batch_size, seq_len, -1)

                recon_loss = F.mse_loss(reconstructed, recon_targets)
            else:
                recon_loss = torch.tensor(0.0, device=device)

            loss = token_loss + context_loss_weight * recon_loss

            # Accuracy
            predictions = shift_logits.argmax(dim=-1)
            mask = shift_labels != 0
            correct += ((predictions == shift_labels) & mask).sum().item()
            total_tokens += mask.sum().item()

            # Calculate context change
            if context_trajectory.size(1) > 1:
                context_diff = context_trajectory[:, 1:, :] - context_trajectory[:, :-1, :]
                context_change = torch.norm(context_diff, dim=-1)
                avg_context_change = context_change.mean().item()
                total_context_change += avg_context_change
                num_context_samples += 1

            total_loss += loss.item()
            total_token_loss += token_loss.item()
            total_recon_loss += recon_loss.item()

    accuracy = correct / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(min(total_token_loss / len(dataloader), 20))  # Cap at 20 to avoid overflow
    avg_context_change = total_context_change / num_context_samples if num_context_samples > 0 else 0

    return {
        'loss': total_loss / len(dataloader),
        'token_loss': total_token_loss / len(dataloader),
        'recon_loss': total_recon_loss / len(dataloader),
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
    parser.add_argument('--vocab-size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--output-dir', type=str, default='./checkpoints', help='Output directory')
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
    print(f"  Vocabulary size: {args.vocab_size}")
    print(f"  Device: {args.device}")
    print(f"  Output directory: {args.output_dir}")
    print()

    # Load data
    train_texts, val_texts = load_wikitext_data(args.max_samples)

    # Create tokenizer
    tokenizer = create_tokenizer(train_texts, vocab_size=args.vocab_size,
                                  output_dir=f"{args.output_dir}/tokenizer")

    # Tokenize
    train_encodings = tokenize_texts(train_texts, tokenizer)
    val_encodings = tokenize_texts(val_texts, tokenizer)

    # Create datasets
    train_dataset = TextDataset(train_encodings)
    val_dataset = TextDataset(val_encodings)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate_fn)

    print(f"\n✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")

    # Create model
    print(f"\n🧠 Creating model...")
    config = NewLLMConfig()
    config.vocab_size = tokenizer.get_vocab_size()
    config.embed_dim = 256
    config.hidden_dim = 512
    config.context_vector_dim = 256
    config.num_layers = args.layers
    config.max_seq_length = 512
    config.pad_token_id = 1
    config.bos_token_id = 2
    config.eos_token_id = 3

    device = torch.device(args.device)
    model = ContextVectorLLM(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params:,} parameters")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print(f"\n{'='*80}")
    print("Training")
    print("=" * 80)

    # Store metrics for plotting
    history = {
        'train_loss': [],
        'train_token_loss': [],
        'train_recon_loss': [],
        'train_context_change': [],
        'val_loss': [],
        'val_token_loss': [],
        'val_recon_loss': [],
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
        history['train_context_change'].append(train_metrics['context_change'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_token_loss'].append(val_metrics['token_loss'])
        history['val_recon_loss'].append(val_metrics['recon_loss'])
        history['val_perplexity'].append(val_metrics['perplexity'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_context_change'].append(val_metrics['context_change'])

        # Quick summary (1 line)
        progress_pct = (epoch + 1) / args.epochs * 100
        print(f"\n✓ Epoch {epoch + 1}/{args.epochs} DONE - "
              f"PPL: {val_metrics['perplexity']:.2f}, "
              f"Loss: {val_metrics['loss']:.2f}, "
              f"Acc: {val_metrics['accuracy']:.2%} "
              f"({progress_pct:.0f}% complete)")

        # Print detailed results
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1} Results (Detail)")
        print("=" * 80)
        print(f"Train - Loss: {train_metrics['loss']:.4f} | "
              f"Token: {train_metrics['token_loss']:.4f} | "
              f"Recon: {train_metrics['recon_loss']:.4f} | "
              f"CtxΔ: {train_metrics['context_change']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | "
              f"Token: {val_metrics['token_loss']:.4f} | "
              f"Recon: {val_metrics['recon_loss']:.4f} | "
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

    # Plot training curves
    print(f"\n📊 Generating training curves...")
    epochs = range(1, args.epochs + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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

    # Plot 4: Context Vector Change
    axes[1, 1].plot(epochs, history['train_context_change'], 'b-o', label='Train Ctx Change', linewidth=2)
    axes[1, 1].plot(epochs, history['val_context_change'], 'r-s', label='Val Ctx Change', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Context Change (L2 norm)')
    axes[1, 1].set_title('Context Vector Change per Timestep')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"{args.output_dir}/training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {plot_path}")

    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
