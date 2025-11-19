#!/usr/bin/env python3
"""
New-LLM Training Script

Simple, unified training interface for all datasets.

Usage:
    python train.py --dataset ultrachat --epochs 50
    python train.py --dataset ultrachat --epochs 50 --batch-size 2048 --device cuda
    python train.py --checkpoint checkpoints/model.pt --epochs 100  # Resume
"""

import argparse
import torch
from torch.utils.data import DataLoader

from src.models.context_vector_llm import ContextVectorLLM
from src.training.fp16_trainer import FP16Trainer
from src.utils.config import NewLLMConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Train New-LLM')

    # Dataset
    parser.add_argument('--dataset', type=str, default='ultrachat',
                       choices=['ultrachat', 'wikitext'],
                       help='Dataset to train on')

    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=512,
                       help='Batch size (default: 512 for T4 GPU)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate (default: 0.0001)')

    # Model
    parser.add_argument('--layers', type=int, default=1,
                       help='Number of layers (default: 1)')
    parser.add_argument('--context-dim', type=int, default=256,
                       help='Context vector dimension (default: 256)')
    parser.add_argument('--vocab-size', type=int, default=1000,
                       help='Vocabulary size (default: 1000)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--no-amp', action='store_true',
                       help='Disable automatic mixed precision (FP16)')

    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Limit dataset to N samples (for quick experiments)')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("New-LLM Training")
    print("=" * 80)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'

    # Create config
    config = NewLLMConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.num_layers = args.layers
    config.context_vector_dim = args.context_dim
    config.vocab_size = args.vocab_size
    config.device = args.device
    config.use_amp = (args.device == 'cuda' and not args.no_amp)

    print(f"\n📋 Configuration:")
    print(f"   Dataset: {args.dataset}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Layers: {config.num_layers}")
    print(f"   Context dim: {config.context_vector_dim}")
    print(f"   Vocab size: {config.vocab_size}")
    print(f"   Device: {config.device}")
    print(f"   Mixed precision (FP16): {config.use_amp}")

    # Load dataset
    print(f"\n📥 Loading {args.dataset} dataset...")

    if args.dataset == 'ultrachat':
        from src.training.ultrachat_dataset import load_ultrachat_data
        train_dataset, val_dataset, tokenizer = load_ultrachat_data(
            config,
            max_samples=args.max_samples
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported yet")

    print(f"\n✓ Dataset loaded:")
    print(f"   Training samples: {len(train_dataset):,}")
    print(f"   Validation samples: {len(val_dataset):,}")
    print(f"   Vocabulary: {len(tokenizer.word2idx)} words")

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(config.device == 'cuda')
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(config.device == 'cuda')
    )

    # Create model
    print(f"\n🧠 Creating model...")
    model = ContextVectorLLM(config).to(config.device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {total_params:,} parameters")

    # Create trainer
    print(f"\n🏋️  Creating trainer...")
    model_name = f"new_llm_{args.dataset}_layers{config.num_layers}"

    if config.use_amp:
        trainer = FP16Trainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
            model_name=model_name,
            tokenizer=tokenizer,  # IMPORTANT: Include tokenizer
            use_amp=True
        )
        print(f"✓ FP16 Trainer created (mixed precision enabled)")
    else:
        from src.training.trainer import Trainer
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
            model_name=model_name,
            tokenizer=tokenizer  # IMPORTANT: Include tokenizer
        )
        print(f"✓ Trainer created")

    # Resume from checkpoint if specified
    if args.checkpoint:
        print(f"\n📂 Resuming from: {args.checkpoint}")
        # TODO: Implement resume logic

    # Train
    print(f"\n{'='*80}")
    print(f"Starting Training")
    print(f"{'='*80}\n")

    trainer.train()

    print(f"\n{'='*80}")
    print(f"✅ Training Complete!")
    print(f"{'='*80}")
    print(f"\nBest checkpoint saved as: checkpoints/best_{model_name}.pt")
    print(f"\nTo chat with your model:")
    print(f"  python chat.py --checkpoint checkpoints/best_{model_name}.pt")
    print()


if __name__ == "__main__":
    main()
