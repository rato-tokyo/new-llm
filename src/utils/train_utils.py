"""
Training utility functions

Common functions used across training scripts to reduce code duplication.
"""

import subprocess
import torch


def get_git_info():
    """Get git version information

    Returns:
        dict: Git information with keys:
            - commit_hash: Full commit hash
            - commit_short: Short commit hash
            - commit_date: Commit date
            - commit_message: Commit message (first line)
    """
    try:
        commit_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
        ).decode().strip()

        commit_short = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD']
        ).decode().strip()

        commit_date = subprocess.check_output(
            ['git', 'log', '-1', '--format=%cd', '--date=short']
        ).decode().strip()

        commit_message = subprocess.check_output(
            ['git', 'log', '-1', '--format=%s']
        ).decode().strip()

        return {
            'commit_hash': commit_hash,
            'commit_short': commit_short,
            'commit_date': commit_date,
            'commit_message': commit_message
        }
    except Exception:
        return {
            'commit_hash': 'unknown',
            'commit_short': 'unknown',
            'commit_date': 'unknown',
            'commit_message': 'Git repository not found'
        }


def print_git_info():
    """Print git version information"""
    git_info = get_git_info()
    print(f"\n📌 Git Version: {git_info['commit_short']} ({git_info['commit_date']})")
    print(f"   Full commit: {git_info['commit_hash']}")


def print_gpu_info():
    """Print GPU information

    Raises:
        RuntimeError: If GPU is not available
    """
    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU not available! FP16 training requires CUDA GPU.")

    print(f"\n🎮 GPU Information:")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   VRAM: {vram_gb:.1f} GB")


def print_model_info(model):
    """Print model parameter information

    Args:
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024**2:.2f} MB (FP32)")


def print_config_info(config):
    """Print configuration information

    Args:
        config: Configuration object
    """
    print(f"\n⚙️  Configuration:")
    print(f"   Num layers: {config.num_layers}")
    print(f"   Max seq length: {config.max_seq_length}")
    print(f"   Embed dim: {config.embed_dim}")
    print(f"   Hidden dim: {config.hidden_dim}")

    if hasattr(config, 'context_vector_dim'):
        print(f"   Context dim: {config.context_vector_dim}")

    print(f"   Batch size: {config.batch_size}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Device: {config.device}")


def print_dataset_info(train_dataset, val_dataset, tokenizer=None):
    """Print dataset information

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        tokenizer: Tokenizer (optional)
    """
    print(f"\n✓ Dataset ready")
    print(f"   Training sequences: {len(train_dataset)}")
    print(f"   Validation sequences: {len(val_dataset)}")

    if tokenizer and hasattr(tokenizer, 'word2idx'):
        print(f"   Vocabulary size: {len(tokenizer.word2idx)}")


def print_dataloader_info(train_dataloader, val_dataloader):
    """Print DataLoader information

    Args:
        train_dataloader: Training DataLoader
        val_dataloader: Validation DataLoader
    """
    print(f"\n✓ DataLoaders created")
    print(f"   Train batches: {len(train_dataloader)}")
    print(f"   Val batches: {len(val_dataloader)}")
