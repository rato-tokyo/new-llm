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
    hidden_dim = 1024        # FNN hidden dimension (EXPERIMENT 2: increased capacity + gating)
    num_layers = 11          # Number of FNN layers (EXPERIMENT 2: 10→11, +1 layer)
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
    weight_decay = 0.0       # L2 regularization (0.0 = no weight decay)
    num_epochs = 150         # Number of epochs (EXPERIMENT 3: long-term training)
    gradient_clip = 1.0      # Gradient clipping (adaptive in trainer)

    # ========== Data ==========
    train_split = 0.8        # Train/val split
    random_seed = 42         # Random seed

    # ========== Device ==========
    device = "cpu"           # Device


class NewLLMGPUConfig(NewLLMConfig):
    """
    GPU-optimized configuration for New-LLM (Colab/Cloud)

    Default for T4 GPU (16GB VRAM).
    For L4/A100, use specialized config classes below.
    """
    # ========== GPU-Optimized Training ==========
    batch_size = 512         # T4 GPU (16GB) baseline
    learning_rate = 0.0001   # T4 baseline learning rate
    device = "cuda"          # GPU device

    # Inherit all other settings from NewLLMConfig
    # (vocab_size, embed_dim, hidden_dim, num_layers, etc.)


class NewLLML4Config(NewLLMConfig):
    """
    L4 GPU-optimized configuration (24GB VRAM)

    L4 has 1.5x more VRAM than T4, so batch_size can be 4x larger
    (measured: 512 → 5.5GB, so 2048 → ~22GB with safety margin)

    Linear Scaling Rule: batch_size 4x → learning_rate 4x
    """
    # ========== L4 GPU-Optimized Training ==========
    batch_size = 2048        # L4 GPU (24GB) - 4x T4
    learning_rate = 0.0004   # 4x T4 learning rate (Linear Scaling Rule)
    device = "cuda"          # GPU device


class NewLLMA100Config(NewLLMConfig):
    """
    A100 GPU-optimized configuration (40GB VRAM)

    A100 has 2.5x more VRAM than T4, so batch_size can be ~8x larger

    Linear Scaling Rule: batch_size 8x → learning_rate 8x
    """
    # ========== A100 GPU-Optimized Training ==========
    batch_size = 4096        # A100 GPU (40GB) - 8x T4
    learning_rate = 0.0008   # 8x T4 learning rate (Linear Scaling Rule)
    device = "cuda"          # GPU device


class NewLLMAdvancedL4Config(NewLLML4Config):
    """
    Advanced L4 GPU configuration with larger model capacity

    For experiments with:
    - Larger context vectors (512, 1024, 2048)
    - More layers (12, 24, 48)

    Inherits L4 optimization (batch_size=2048) from NewLLML4Config.
    """
    # ========== Scaled-up Architecture ==========
    context_vector_dim = 512  # 2x larger context (can be 1024, 2048)
    num_layers = 12           # 2x more layers

    # batch_size=2048, device="cuda" inherited from NewLLML4Config


class NewLLMAdvancedA100Config(NewLLMA100Config):
    """
    Advanced A100 GPU configuration with larger model capacity
    """
    # ========== Scaled-up Architecture ==========
    context_vector_dim = 512  # 2x larger context
    num_layers = 12           # 2x more layers

    # batch_size=4096, device="cuda" inherited from NewLLMA100Config


# Legacy alias for backward compatibility
class NewLLMAdvancedGPUConfig(NewLLMAdvancedL4Config):
    """
    Legacy alias - use NewLLMAdvancedL4Config or NewLLMAdvancedA100Config instead
    Defaults to L4 optimization.
    """
    pass


# Legacy alias for backward compatibility
class LSTMConfig(BaseConfig):
    """Legacy LSTM configuration (for old experiments)"""
    pass
