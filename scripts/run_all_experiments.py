# -*- coding: utf-8 -*-
"""Run all New-LLM experiments in a single execution

This script runs 2 experiments sequentially with different configurations:
- Experiment 1: 512 hidden_dim, 9 layers, 50 epochs
- Experiment 2: 1024 hidden_dim, 11 layers, 50 epochs

Each experiment produces its own training curve image:
- checkpoints/experiment_1_training_curves.png
- checkpoints/experiment_2_training_curves.png
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader

from src.models.context_vector_llm import ContextVectorLLM
from src.training.dataset import load_data
from src.training.trainer import Trainer
from src.utils.config import NewLLMConfig


class ExperimentConfig:
    """Configuration for a single experiment"""
    def __init__(self, name, hidden_dim, num_layers, num_epochs):
        self.name = name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_epochs = num_epochs


def create_config(exp):
    """Create a NewLLMConfig with experiment-specific parameters"""
    config = NewLLMConfig()

    # Override with experiment-specific values
    config.hidden_dim = exp.hidden_dim
    config.num_layers = exp.num_layers
    config.num_epochs = exp.num_epochs

    return config


def run_experiment(exp, train_loader, val_loader):
    """Run a single experiment and return results"""

    print("\n" + "="*80)
    print("RUNNING {}".format(exp.name.upper()))
    print("="*80)

    # Create config
    config = create_config(exp)

    print("Configuration for {}:".format(exp.name))
    print("  Vocab size: {}".format(config.vocab_size))
    print("  Embed dim: {}".format(config.embed_dim))
    print("  Hidden dim: {}".format(config.hidden_dim))
    print("  Num layers: {}".format(config.num_layers))
    print("  Context vector dim: {}".format(config.context_vector_dim))
    print("  Batch size: {}".format(config.batch_size))
    print("  Learning rate: {}".format(config.learning_rate))
    print("  Num epochs: {}".format(config.num_epochs))
    print("="*80)

    # Create model
    print("\nInitializing model for {}...".format(exp.name))
    model = ContextVectorLLM(config)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print("Model parameters: {:,}".format(num_params))

    # Create trainer with experiment name
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=config,
        model_name="new_llm",
        experiment_name=exp.name  # This ensures unique image filenames
    )

    # Train
    trainer.train()

    # Get best validation loss
    best_val_loss = min(trainer.val_losses)
    best_epoch = trainer.val_losses.index(best_val_loss) + 1

    print("\n✓ {} completed!".format(exp.name))
    print("  Best Val Loss: {:.4f} at epoch {}".format(best_val_loss, best_epoch))
    print("  Parameters: {:,}".format(num_params))

    return {
        "name": exp.name,
        "hidden_dim": exp.hidden_dim,
        "num_layers": exp.num_layers,
        "num_epochs": exp.num_epochs,
        "num_params": num_params,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def print_summary(results):
    """Print summary table of all experiments"""

    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)

    print("\n| Experiment | Hidden Dim | Layers | Epochs | Parameters | Best Val Loss | Best Epoch |")
    print("|------------|------------|--------|--------|------------|---------------|------------|")

    for r in results:
        print("| {} | {} | {} | {} | {:,} | {:.4f} | {} |".format(
            r['name'], r['hidden_dim'], r['num_layers'], r['num_epochs'],
            r['num_params'], r['best_val_loss'], r['best_epoch']))

    # Find best overall
    best = min(results, key=lambda x: x['best_val_loss'])

    print("\n" + "="*80)
    print("BEST RESULT: {}".format(best['name']))
    print("  Val Loss: {:.4f}".format(best['best_val_loss']))
    print("  Epoch: {}".format(best['best_epoch']))
    print("  Parameters: {:,}".format(best['num_params']))
    print("="*80)

    print("\nGenerated images:")
    for r in results:
        print("  - checkpoints/{}_training_curves.png".format(r['name']))

    print("\n")


def main():
    """Main execution function"""

    print("="*80)
    print("NEW-LLM: ALL EXPERIMENTS RUNNER")
    print("="*80)
    print("\nThis script will run 2 experiments sequentially:")
    print("  1. Experiment 1: 512 hidden, 9 layers, 50 epochs")
    print("  2. Experiment 2: 1024 hidden, 11 layers, 50 epochs")
    print("\nEach experiment will generate its own training curve image.")
    print("="*80)

    # Define experiments (Final: High-quality 500-sentence dataset + Early Stopping)
    experiments = [
        ExperimentConfig("final_exp1", hidden_dim=512, num_layers=9, num_epochs=50),
        ExperimentConfig("final_exp2", hidden_dim=1024, num_layers=11, num_epochs=50),
    ]

    # Load data once (same for all experiments)
    print("\nLoading data...")
    base_config = NewLLMConfig()
    data_path = "data/sample_texts.txt"
    train_dataset, val_dataset, tokenizer = load_data(data_path, base_config)

    print("Train samples: {}".format(len(train_dataset)))
    print("Val samples: {}".format(len(val_dataset)))
    print("Vocabulary size: {}".format(len(tokenizer.word2idx)))

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=base_config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=base_config.batch_size,
        shuffle=False
    )

    # Run all experiments
    results = []
    for exp in experiments:
        result = run_experiment(exp, train_loader, val_loader)
        results.append(result)

    # Print summary
    print_summary(results)

    print("="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
