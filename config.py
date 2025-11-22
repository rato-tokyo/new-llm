"""
New-LLM Configuration File

This is the main configuration file for the project.
Edit values here to change model architecture, training settings, and data parameters.

Usage:
    python3 tests/phase2_experiments/test_residual.py --config Residual4Layer
    python3 tests/phase2_experiments/test_residual.py --config Residual2Layer
"""


# ============================================================================
# DEFAULT CONFIGURATION (Residual4Layer - Recommended)
# ============================================================================

class ResidualConfig:
    """
    Base configuration for all Residual Standard architectures.

    To create custom config:
    1. Create new class inheriting from ResidualConfig
    2. Override desired parameters
    3. Use with: --config YourConfigName
    """

    # ========== Model Architecture ==========
    architecture = "residual_standard"
    layer_structure = [1, 1, 1, 1]  # 4 FNN blocks (recommended)
    context_dim = 256               # Context vector dimension
    embed_dim = 256                 # Token embedding dimension
    hidden_dim = 512                # Hidden dimension (auto-calculated if not set)
    vocab_size = 50257              # GPT-2 tokenizer vocabulary
    dropout = 0.1                   # Dropout rate

    # ========== CVFP Settings (Do not change unless you know what you're doing) ==========
    context_update_strategy = "gated"  # Must be "gated" for CVFP
    use_layer_norm = True
    use_context_clipping = True

    # ========== Phase 1: Fixed-Point Context Learning ==========
    phase1_max_iterations = 50           # Maximum iterations for fixed-point search
    phase1_convergence_threshold = 0.02  # MSE threshold for convergence (0.02 = relaxed, 0.01 = strict)
    phase1_min_converged_ratio = 0.95    # Stop when 95% of tokens converged

    # LR Schedule (optimized for fast convergence)
    phase1_lr_warmup = 0.001      # High LR for iterations 1-3 (fast initial convergence)
    phase1_lr_medium = 0.0005     # Medium LR for iterations 4-10
    phase1_lr_finetune = 0.0001   # Low LR for iterations 11+ (fine-tuning)

    # Batch processing (not implemented yet)
    phase1_batch_size = 32

    # ========== Phase 2: Token Prediction ==========
    phase2_learning_rate = 0.0001   # Learning rate for token prediction
    phase2_epochs = 10              # Number of training epochs
    phase2_batch_size = 32          # Batch size
    phase2_gradient_clip = 1.0      # Gradient clipping value

    # ========== Data ==========
    max_seq_length = 1024                          # Maximum sequence length
    dataset_name = "HuggingFaceH4/ultrachat_200k"  # HuggingFace dataset
    dataset_split = "train_sft"                    # Dataset split to use
    cache_dir = "./cache"                          # Cache directory
    num_samples = 10                               # Number of samples to train on
    train_val_split = 0.8                          # Train/Val split ratio (80/20)

    # ========== Device ==========
    device = "cpu"        # "cpu" or "cuda"
    random_seed = 42      # Random seed for reproducibility

    # ========== Logging ==========
    log_every_steps = 1
    save_every_samples = 10


# ============================================================================
# PREDEFINED CONFIGURATIONS
# ============================================================================

class Residual2Layer(ResidualConfig):
    """2-layer architecture (minimum viable, fast for testing)"""
    layer_structure = [1, 1]


class Residual4Layer(ResidualConfig):
    """4-layer architecture (recommended for production)"""
    layer_structure = [1, 1, 1, 1]


class Residual8Layer(ResidualConfig):
    """8-layer architecture (high performance, slower training)"""
    layer_structure = [1, 1, 1, 1, 1, 1, 1, 1]


class Residual4Layer512Ctx(ResidualConfig):
    """4-layer with 512-dim context (large context, more expressive)"""
    layer_structure = [1, 1, 1, 1]
    context_dim = 512
    hidden_dim = 1024


# ============================================================================
# CUSTOM CONFIGURATION EXAMPLE
# ============================================================================

# Uncomment and modify to create your own config:
#
# class MyCustomConfig(ResidualConfig):
#     """My custom configuration"""
#     layer_structure = [2, 2]         # 2 blocks with 2 layers each
#     context_dim = 384                # Custom context dimension
#     num_samples = 100                # Train on 100 samples
#     phase1_lr_warmup = 0.002         # Higher initial LR
