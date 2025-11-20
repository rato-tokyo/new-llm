"""
Configuration for new-llm experiments

IMPORTANT: This is the single source of truth for all model configurations.
All parameters should be defined here and referenced from other modules.
Do not hardcode configuration values elsewhere.

Primary architecture: New-LLM with context vector propagation (NO attention mechanism)
"""


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

    context_update_strategy = "gated"   # Context update strategy: "simple" or "gated" (DEFAULT: gated)
                                        # - "simple": Complete overwrite (context_new = f(hidden)) - NOT RECOMMENDED
                                        # - "gated": LSTM-style gated addition (context_new = forget*context + input*delta) - RECOMMENDED
                                        # Gated is superior for complex patterns (verified by CVFPT experiments)
                                        # See src/models/components/context_updaters.py for implementation

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

    Scaling Rule: batch_size 32→2048 (64x) → LR sqrt scaling
    Base: batch=32, lr=0.0001
    L4: batch=2048, lr=0.0001 * sqrt(64) = 0.0008
    """
    # ========== L4 GPU-Optimized Training ==========
    batch_size = 2048        # L4 GPU (24GB) - 64x CPU baseline
    learning_rate = 0.0008   # sqrt(64) = 8x CPU baseline (Square Root Scaling)
    device = "cuda"          # GPU device


class NewLLMA100Config(NewLLMConfig):
    """
    A100 GPU-optimized configuration (40GB VRAM)

    A100 has 2.5x more VRAM than T4, so batch_size can be ~8x larger

    Scaling Rule: batch_size 32→4096 (128x) → LR sqrt scaling
    Base: batch=32, lr=0.0001
    A100: batch=4096, lr=0.0001 * sqrt(128) ≈ 0.0011
    """
    # ========== A100 GPU-Optimized Training ==========
    batch_size = 4096        # A100 GPU (40GB) - 128x CPU baseline
    learning_rate = 0.0011   # sqrt(128) ≈ 11.3x CPU baseline (Square Root Scaling)
    device = "cuda"          # GPU device


class NewLLMAdvancedL4Config(NewLLML4Config):
    """
    Advanced L4 GPU configuration with larger model capacity

    For experiments with:
    - Larger context vectors (512, 1024, 2048)
    - More layers (12, 24, 48)

    Model Size Scaling Rule: Larger model (4.84M vs 2.74M) → Lower learning rate
    """
    # ========== Scaled-up Architecture ==========
    context_vector_dim = 512  # 2x larger context (can be 1024, 2048)
    num_layers = 12           # 2x more layers

    # ========== Learning Rate Adjustment for Model Size ==========
    learning_rate = 0.0004    # Half of baseline (Model Size Scaling Rule)
                              # Baseline (2.74M): 0.0008
                              # Advanced (4.84M, 1.77x larger): 0.0004

    # batch_size=2048, device="cuda" inherited from NewLLML4Config


class NewLLMAdvancedA100Config(NewLLMA100Config):
    """
    Advanced A100 GPU configuration with larger model capacity

    Model Size Scaling Rule: Larger model → Lower learning rate
    """
    # ========== Scaled-up Architecture ==========
    context_vector_dim = 512  # 2x larger context
    num_layers = 12           # 2x more layers

    # ========== Learning Rate Adjustment for Model Size ==========
    learning_rate = 0.0006    # ~Half of baseline (Model Size Scaling Rule)
                              # Baseline (2.74M): 0.0011
                              # Advanced (4.84M, 1.77x larger): 0.0006

    # batch_size=4096, device="cuda" inherited from NewLLMA100Config
