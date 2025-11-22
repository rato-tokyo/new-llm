"""Configuration for UltraChat Dialogue Training - Residual Standard Architecture

Unified to Residual Standard architecture with CVFP training.
All configurations use [1,1,1,...] layer structure.
"""


class ResidualConfig:
    """
    Base configuration for Residual Standard architecture

    Easy to modify:
    - layer_structure: List of FNN blocks [1,1,1,1] for 4 layers
    - context_dim: Context vector dimension
    - embed_dim: Token embedding dimension
    """

    # ========== Model Architecture ==========
    architecture = "residual_standard"
    layer_structure = [1, 1, 1, 1]  # 4 layers (4 FNN blocks with 1 layer each)
    context_dim = 256        # Context vector dimension
    embed_dim = 256          # Token embedding dimension
    hidden_dim = 512         # Hidden dimension per FNN block
    vocab_size = 50257       # GPT-2 tokenizer vocabulary size
    dropout = 0.1

    # ========== CVFP Settings ==========
    context_update_strategy = "gated"  # Must be "gated" (not "simple")
    use_layer_norm = True
    use_context_clipping = True

    # ========== Phase 1: Fixed-Point Context Learning (CVFP) ==========
    phase1_max_iterations = 50
    phase1_convergence_threshold = 0.02  # MSE threshold (relaxed from 0.01)
    phase1_min_converged_ratio = 0.95    # 95% tokens must converge

    # LR Schedule (optimized for fast convergence)
    phase1_lr_warmup = 0.001      # High LR for iterations 1-3
    phase1_lr_medium = 0.0005     # Medium LR for iterations 4-10
    phase1_lr_finetune = 0.0001   # Low LR for iterations 11+

    # Batch processing
    phase1_batch_size = 32        # Batch size for parallel processing

    # ========== Phase 2: Token Prediction ==========
    phase2_learning_rate = 0.0001
    phase2_epochs = 10
    phase2_batch_size = 32
    phase2_gradient_clip = 1.0

    # ========== Data ==========
    max_seq_length = 1024
    dataset_name = "HuggingFaceH4/ultrachat_200k"
    dataset_split = "train_sft"
    cache_dir = "./cache"
    num_samples = 10              # Number of samples for training
    train_val_split = 0.8         # Train/Val split ratio

    # ========== Device ==========
    device = "cpu"
    random_seed = 42

    # ========== Logging ==========
    log_every_steps = 1
    save_every_samples = 10


class Residual2Layer(ResidualConfig):
    """2-layer Residual Standard (minimum viable)"""
    layer_structure = [1, 1]
    context_dim = 256
    embed_dim = 256


class Residual4Layer(ResidualConfig):
    """4-layer Residual Standard (recommended)"""
    layer_structure = [1, 1, 1, 1]
    context_dim = 256
    embed_dim = 256


class Residual8Layer(ResidualConfig):
    """8-layer Residual Standard (high performance)"""
    layer_structure = [1, 1, 1, 1, 1, 1, 1, 1]
    context_dim = 256
    embed_dim = 256
    hidden_dim = 512


class Residual4Layer512Ctx(ResidualConfig):
    """4-layer with 512-dim context (large context)"""
    layer_structure = [1, 1, 1, 1]
    context_dim = 512
    embed_dim = 256
    hidden_dim = 1024
