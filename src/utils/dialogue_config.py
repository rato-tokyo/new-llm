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
    phase1_convergence_threshold = 0.01  # MSE threshold
    phase1_learning_rate = 0.0001
    phase1_min_converged_ratio = 0.95    # 95% tokens must converge

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
