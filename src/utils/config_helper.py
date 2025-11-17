"""
Configuration helper utilities

Provides utilities for validating and displaying configurations.
"""

from .config import BaseConfig, NewLLMConfig


def print_config(config, title="Configuration"):
    """
    Print configuration in a formatted way

    Args:
        config: Configuration object (BaseConfig or NewLLMConfig)
        title: Title to display
    """
    print("=" * 70)
    print(f"{title}")
    print("=" * 70)

    # Model Architecture
    print("\nModel Architecture:")
    print(f"  Vocabulary size:      {config.vocab_size:>10,}")
    print(f"  Embedding dimension:  {config.embed_dim:>10,}")
    print(f"  Hidden dimension:     {config.hidden_dim:>10,}")
    print(f"  Number of layers:     {config.num_layers:>10,}")
    print(f"  Max sequence length:  {config.max_seq_length:>10,}")
    print(f"  Dropout rate:         {config.dropout:>10.3f}")

    # Context vector specific (if NewLLMConfig)
    if hasattr(config, 'context_vector_dim'):
        print(f"  Context vector dim:   {config.context_vector_dim:>10,}")

    # Training
    print("\nTraining Hyperparameters:")
    print(f"  Batch size:           {config.batch_size:>10,}")
    print(f"  Learning rate:        {config.learning_rate:>10.5f}")
    print(f"  Number of epochs:     {config.num_epochs:>10,}")
    print(f"  Gradient clipping:    {config.gradient_clip:>10.3f}")

    # Data
    print("\nData Configuration:")
    print(f"  Train split:          {config.train_split:>10.1%}")
    print(f"  Random seed:          {config.random_seed:>10,}")

    # Device
    print("\nDevice:")
    print(f"  Device:               {config.device:>10s}")

    print("=" * 70)


def validate_config(config):
    """
    Validate configuration parameters

    Args:
        config: Configuration object

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate positive integers
    positive_int_params = [
        'vocab_size', 'embed_dim', 'hidden_dim', 'num_layers',
        'max_seq_length', 'batch_size', 'num_epochs'
    ]
    for param in positive_int_params:
        value = getattr(config, param)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{param} must be a positive integer, got {value}")

    # Validate positive floats
    if not (0.0 < config.learning_rate < 1.0):
        raise ValueError(f"learning_rate must be in (0, 1), got {config.learning_rate}")

    if not (0.0 <= config.dropout < 1.0):
        raise ValueError(f"dropout must be in [0, 1), got {config.dropout}")

    if not (0.0 < config.train_split < 1.0):
        raise ValueError(f"train_split must be in (0, 1), got {config.train_split}")

    # Validate context vector dim for NewLLMConfig
    if hasattr(config, 'context_vector_dim'):
        if not isinstance(config.context_vector_dim, int) or config.context_vector_dim <= 0:
            raise ValueError(f"context_vector_dim must be a positive integer, got {config.context_vector_dim}")

    print("✓ Configuration validated successfully")


def get_model_info(config, model):
    """
    Get model information based on configuration

    Args:
        config: Configuration object
        model: PyTorch model

    Returns:
        dict: Dictionary with model information
    """
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        'total_parameters': num_params,
        'trainable_parameters': num_trainable,
        'vocab_size': config.vocab_size,
        'embed_dim': config.embed_dim,
        'hidden_dim': config.hidden_dim,
        'num_layers': config.num_layers,
    }

    if hasattr(config, 'context_vector_dim'):
        info['context_vector_dim'] = config.context_vector_dim
        info['model_type'] = 'New-LLM'
    else:
        info['model_type'] = 'Baseline LSTM'

    return info


if __name__ == "__main__":
    # Example usage
    print("\nValidating BaseConfig:")
    base_config = BaseConfig()
    validate_config(base_config)
    print_config(base_config, "Baseline LSTM Configuration")

    print("\n" + "="*70 + "\n")

    print("Validating NewLLMConfig:")
    new_config = NewLLMConfig()
    validate_config(new_config)
    print_config(new_config, "New-LLM Configuration")
