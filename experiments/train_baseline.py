"""Train baseline FNN-based language model"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader

from src.models.baseline_llm import BaselineLLM
from src.training.dataset import load_data
from src.training.trainer import Trainer
from src.utils.config import BaseConfig


def main():
    # Configuration
    config = BaseConfig()

    print("="*60)
    print("Baseline FNN-based Language Model Training")
    print("="*60)
    print(f"Configuration:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Num epochs: {config.num_epochs}")
    print("="*60)

    # Load data
    data_path = "data/sample_texts.txt"
    print(f"\nLoading data from {data_path}...")
    train_dataset, val_dataset, tokenizer = load_data(data_path, config)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Vocabulary size: {len(tokenizer.word2idx)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Create model
    print("\nInitializing baseline model...")
    model = BaselineLLM(config)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        model_name="baseline_llm"
    )

    # Train
    trainer.train()

    # Test generation
    print("\n" + "="*60)
    print("Testing text generation...")
    print("="*60)

    model.eval()
    test_prompts = [
        "the cat",
        "i like",
        "the sun",
    ]

    for prompt in test_prompts:
        # Encode prompt
        token_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([token_ids], device=config.device)

        # Generate
        generated = model.generate(input_tensor, max_new_tokens=10, temperature=0.8)

        # Decode
        generated_text = tokenizer.decode(generated[0].tolist())

        print(f"\nPrompt: {prompt}")
        print(f"Generated: {generated_text}")

    print("\n" + "="*60)
    print("Training completed!")
    print("="*60)


if __name__ == "__main__":
    main()
