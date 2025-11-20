"""Configuration for UltraChat Dialogue Training

This config is specifically designed for:
- UltraChat dataset
- Two-phase training (context then tokens)
- Flexible architecture (easy to change layers and context_dim)
- Local GPU training (no Colab)
"""


class DialogueConfig:
    """
    Base configuration for dialogue training

    Easy to modify:
    - num_layers: Change number of FNN layers
    - context_dim: Change context vector size
    - hidden_dim: Change hidden layer size
    """

    # ========== Model Architecture (EASY TO CHANGE) ==========
    architecture = "sequential"  # "sequential" or "layerwise"
    num_layers = 2           # Number of FNN layers (1, 2, 3, 4, ...)
    context_dim = 256        # Context vector dimension (128, 256, 512, 1024, ...)
    hidden_dim = 512         # Hidden dimension (256, 512, 1024, ...)

    # ========== Fixed Parameters ==========
    vocab_size = 50259       # Vocabulary size (GPT-2 tokenizer)
    embed_dim = 256          # Token embedding dimension
    dropout = 0.1            # Dropout rate

    # ========== Phase 1: Context Learning ==========
    phase1_max_samples = 100          # Number of dialogue samples for Phase 1
    phase1_epochs = 3                 # Number of training epochs for Phase 1
    phase1_learning_rate = 0.0001     # Learning rate for Phase 1 training
    phase1_max_iterations = 200       # Max iterations for fixed-point search
    phase1_warmup_iterations = 100    # Warmup iterations before checking convergence (n)
    phase1_convergence_threshold = 1e-2  # Convergence threshold (L2 distance) - relaxed from 1e-4
    phase1_min_converged_ratio = 0.95    # Minimum ratio of converged tokens to proceed

    # ========== Phase 2: Token Prediction ==========
    phase2_batch_size = 1              # Batch size (start with 1 for debugging)
    phase2_learning_rate = 0.0001      # Learning rate
    phase2_epochs = 10                 # Number of epochs per sample
    phase2_gradient_clip = 1.0         # Gradient clipping

    # ========== Data ==========
    max_seq_length = 512     # Maximum sequence length
    dataset_name = "HuggingFaceH4/ultrachat_200k"  # UltraChat dataset
    dataset_split = "train_sft"  # Use supervised fine-tuning split
    cache_dir = "./cache"    # Cache directory for datasets and contexts

    # ========== Device ==========
    device = "cuda"          # GPU device
    random_seed = 42         # Random seed

    # ========== Logging ==========
    log_every_steps = 1      # Log metrics every N steps
    save_every_samples = 10  # Save checkpoint every N samples


class TinyDialogueConfig(DialogueConfig):
    """Tiny model for initial baseline (1 layer, 128 dim)"""
    num_layers = 1
    context_dim = 128
    hidden_dim = 256
    phase1_max_samples = 1  # Start with 1 sample for initial testing
    # Inherit phase1_max_iterations=200, phase1_warmup_iterations=100 from parent


class Tiny256DialogueConfig(DialogueConfig):
    """Tiny model with doubled context_dim (1 layer, 256 dim)"""
    num_layers = 1
    context_dim = 256  # Doubled from Tiny
    hidden_dim = 512   # Doubled to match context_dim ratio
    phase1_max_samples = 1
    # Inherit phase1_max_iterations=200, phase1_warmup_iterations=100 from parent


class SmallDialogueConfig(DialogueConfig):
    """Small model for quick testing (2 layers, 256 dim)"""
    num_layers = 2
    context_dim = 256
    hidden_dim = 512
    phase1_max_samples = 1  # Changed from 10 to 1 for comparison
    # Inherit phase1_max_iterations=200, phase1_warmup_iterations=100 from parent


class Small2LayerSequentialConfig(DialogueConfig):
    """Architecture A: Sequential FNN (2 layers, 256 dim)"""
    architecture = "sequential"
    num_layers = 2
    context_dim = 256
    hidden_dim = 256  # Smaller hidden_dim
    phase1_max_samples = 1


class Small2LayerLayerwiseConfig(DialogueConfig):
    """Architecture B: Layer-wise context update (2 layers, 256 dim)"""
    architecture = "layerwise"
    num_layers = 2
    context_dim = 256
    hidden_dim = 256  # Smaller hidden_dim
    phase1_max_samples = 1


class MediumDialogueConfig(DialogueConfig):
    """Medium model for experiments (3 layers)"""
    num_layers = 3
    context_dim = 512
    hidden_dim = 1024
    phase1_max_samples = 10
    # Inherit phase1_max_iterations=200, phase1_warmup_iterations=100 from parent


class LargeDialogueConfig(DialogueConfig):
    """Large model for full training"""
    num_layers = 4
    context_dim = 512
    hidden_dim = 1024
    phase1_max_samples = 1000
