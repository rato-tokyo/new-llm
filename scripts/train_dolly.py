#!/usr/bin/env python3
"""
Train New-LLM on Dolly-15k from scratch for dialog/instruction-following

Usage (L4 GPU):
    python scripts/train_dolly.py --num_layers 1
    python scripts/train_dolly.py --num_layers 4

Dolly-15k dataset:
    - 15,000 instruction-response pairs
    - Categories: open_qa, closed_qa, summarization, classification, etc.
    - Format: "Instruction: ... Context: ... Response: ..."
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
from src.models.context_vector_llm import ContextVectorLLM
from src.training.fp16_trainer import FP16Trainer
from src.training.dolly_dataset import load_dolly_data
from src.utils.config import NewLLML4Config
from src.utils.train_utils import (
    print_git_info, print_gpu_info, print_model_info,
    print_config_info, print_dataset_info, print_dataloader_info
)
import argparse


class DollyTrainConfig(NewLLML4Config):
    """Config for Dolly-15k training on L4 GPU (from scratch)

    Inherits L4 optimization from NewLLML4Config:
    - batch_size = 2048
    - learning_rate = 0.0008 (Square Root Scaling)
    - device = "cuda"

    Dolly-15k specific adjustments:
    - Longer sequences (128 tokens) for dialog
    - More epochs for instruction learning
    """
    # Data settings (Dolly-15k)
    max_seq_length = 128  # Longer for dialog (WikiText-2 was 64)
    vocab_size = 1000  # Same as WikiText-2

    # Model architecture
    embed_dim = 256
    hidden_dim = 512
    num_layers = 1  # Default, will be overridden in __init__
    context_vector_dim = 256
    dropout = 0.1

    # Training hyperparameters (inherited from NewLLML4Config)
    # batch_size = 2048      ← L4 GPU optimized
    # learning_rate = 0.0008 ← Square Root Scaling Rule
    # device = "cuda"
    num_epochs = 100  # Dolly-15k is smaller than WikiText-2
    weight_decay = 0.0
    gradient_clip = 1.0

    # Early stopping
    patience = 20

    # FP16 mixed precision
    use_amp = True

    def __init__(self, num_layers=1):
        """Initialize config with specified num_layers"""
        super().__init__()
        self.num_layers = num_layers


def main():
    """Train New-LLM on Dolly-15k from scratch"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train New-LLM on Dolly-15k')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='Number of layers (default: 1)')
    parser.add_argument('--max_seq_length', type=int, default=128,
                       help='Maximum sequence length (default: 128)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print(f"Dolly-15k Training - {args.num_layers} Layer(s)")
    print("="*80)

    # Git version information
    print_git_info()

    # Create configuration
    config = DollyTrainConfig(num_layers=args.num_layers)
    config.max_seq_length = args.max_seq_length

    # Print configuration
    print_config_info(config)

    # GPU check and info
    print_gpu_info()

    # Load Dolly-15k dataset
    print("\n" + "="*80)
    print("Loading Dolly-15k Dataset")
    print("="*80)

    train_dataset, val_dataset, tokenizer = load_dolly_data(config)
    print_dataset_info(train_dataset, val_dataset, tokenizer)

    # Create DataLoaders
    print(f"\n📦 Creating DataLoaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    print_dataloader_info(train_dataloader, val_dataloader)

    # Create model
    print("\n" + "="*80)
    print("Creating Model")
    print("="*80)

    model = ContextVectorLLM(config).to(config.device)
    print_model_info(model)

    # Create trainer
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)

    trainer = FP16Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        model_name=f"new_llm_dolly_layers{config.num_layers}",
        use_amp=config.use_amp
    )

    # Train
    trainer.train()

    print("\n" + "="*80)
    print("✅ Training Complete!")
    print(f"   Best model saved: checkpoints/best_new_llm_dolly_layers{config.num_layers}.pt")
    print("="*80)


if __name__ == "__main__":
    main()
