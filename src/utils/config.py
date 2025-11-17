"""
Configuration for new-llm experiments

IMPORTANT: This is the single source of truth for all model configurations.
All parameters should be defined here and referenced from other modules.
Do not hardcode configuration values elsewhere.
"""

class BaseConfig:
    """
    Base configuration for models

    This configuration is used by the Baseline LSTM model.
    All size parameters affect model capacity and memory usage.
    """
    # ========== Model Architecture ==========
    vocab_size = 1000        # Size of vocabulary (number of unique tokens)
    embed_dim = 128          # Dimension of token embeddings
    hidden_dim = 256         # Dimension of hidden states (LSTM) or FNN hidden layers
    num_layers = 3           # Number of stacked layers (LSTM or FNN)
    max_seq_length = 32      # Maximum sequence length for positional embeddings
    dropout = 0.1            # Dropout rate for regularization

    # ========== Training Hyperparameters ==========
    batch_size = 16          # Number of samples per batch
    learning_rate = 0.001    # Learning rate for optimizer
    num_epochs = 50          # Number of training epochs
    gradient_clip = 1.0      # Gradient clipping threshold

    # ========== Data ==========
    train_split = 0.8        # Fraction of data for training (rest for validation)
    random_seed = 42         # Random seed for reproducibility

    # ========== Device ==========
    device = "cpu"           # Device for training (cpu/cuda) - CPU for 16GB RAM systems


class NewLLMConfig(BaseConfig):
    """
    Configuration for New-LLM with context vector propagation

    Inherits all parameters from BaseConfig and adds context vector specific settings.
    The context vector is the key difference from baseline models.
    """
    # ========== Context Vector Specific ==========
    context_vector_dim = 64  # Dimension of the context vector that propagates information
                             # This is the fixed-size representation carrying contextual info
                             # Smaller than hidden_dim (256) for efficiency:
                             # - 4x smaller than LSTM hidden state
                             # - 8x smaller than LSTM total context (h + c)
