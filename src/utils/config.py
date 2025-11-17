"""
Configuration for new-llm experiments

IMPORTANT: This is the single source of truth for all model configurations.
All parameters should be defined here and referenced from other modules.
Do not hardcode configuration values elsewhere.

NOTE: The primary comparison is now between:
- TransformerConfig: Standard Transformer with self-attention (like GPT)
- NewLLMConfig: Context vector propagation (NO attention mechanism)

Goal: Verify if context vector propagation can compete with attention mechanisms.
"""

class BaseConfig:
    """
    Base configuration - NOT USED for main experiments

    Kept for backward compatibility with LSTM experiments.
    Use TransformerConfig or NewLLMConfig for primary experiments.
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


class TransformerConfig:
    """
    Configuration for Transformer-based Language Model (Baseline with Attention)

    This is a standard GPT-like model with:
    - Multi-head self-attention
    - Feed-forward layers
    - Layer normalization
    - Positional embeddings

    This serves as the primary baseline to compare against New-LLM.
    """
    # ========== Model Architecture ==========
    vocab_size = 1000        # Size of vocabulary
    embed_dim = 256          # Token embedding dimension (increased from 128)
    num_heads = 4            # Number of attention heads (embed_dim must be divisible by this)
    hidden_dim = 1024        # FFN hidden dimension (typically 4x embed_dim)
    num_layers = 6           # Number of Transformer blocks
    max_seq_length = 32      # Maximum sequence length
    dropout = 0.1            # Dropout rate

    # ========== Training Hyperparameters ==========
    batch_size = 16          # Batch size
    learning_rate = 0.0001   # Lower LR for Transformer (more stable)
    num_epochs = 50          # Training epochs
    gradient_clip = 1.0      # Gradient clipping

    # ========== Data ==========
    train_split = 0.8        # Train/val split
    random_seed = 42         # Random seed

    # ========== Device ==========
    device = "cpu"           # Device


class NewLLMConfig:
    """
    Configuration for New-LLM with Context Vector Propagation

    Key innovation: NO ATTENTION MECHANISM
    Instead uses:
    - Fixed-size context vector that accumulates information
    - Additive updates: context[t] = context[t-1] + delta[t]
    - Indirect learning: only token prediction loss (context emerges naturally)

    Parameters are scaled up to compete fairly with Transformer baseline.
    """
    # ========== Model Architecture ==========
    vocab_size = 1000        # Size of vocabulary
    embed_dim = 256          # Token embedding dimension (SAME as Transformer)
    hidden_dim = 1024        # FNN hidden dimension (EXPERIMENT 2: 512→1024, SAME as Transformer)
    num_layers = 10          # Number of FNN layers (EXPERIMENT 2: 8→10)
    max_seq_length = 32      # Maximum sequence length
    dropout = 0.1            # Dropout rate

    # ========== Context Vector Specific ==========
    context_vector_dim = 512  # Context vector dimension (EXPERIMENT 1: 256→512)
                              # Increased to provide more capacity for context compression
                              # This vector carries ALL contextual information
                              # Unlike attention which can look at all positions,
                              # this must compress everything into fixed size

    # ========== Training Hyperparameters ==========
    batch_size = 16          # Batch size
    learning_rate = 0.0001   # Learning rate (same as Transformer)
    num_epochs = 150         # Number of epochs (EXPERIMENT 3: 50→150 for longer training)
    gradient_clip = 1.0      # Gradient clipping

    # ========== Data ==========
    train_split = 0.8        # Train/val split
    random_seed = 42         # Random seed

    # ========== Device ==========
    device = "cpu"           # Device


# Legacy alias for backward compatibility
class LSTMConfig(BaseConfig):
    """Legacy LSTM configuration (for old experiments)"""
    pass
