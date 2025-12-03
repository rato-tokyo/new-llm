"""
Pythia Configuration

Pythia-70M および Context-Pythia の設定。
"""

import torch


class PythiaConfig:
    """Pythia-70M アーキテクチャ設定"""

    # ========== モデルアーキテクチャ ==========
    vocab_size = 50304              # Pythia vocabulary size
    hidden_size = 512               # Hidden dimension
    num_layers = 6                  # Number of transformer layers
    num_heads = 8                   # Number of attention heads
    intermediate_size = 2048        # FFN intermediate dimension
    max_position_embeddings = 2048  # Maximum sequence length
    rotary_pct = 0.25               # Percentage of head_dim for rotary embedding
    layer_norm_eps = 1e-5           # Layer norm epsilon

    # ========== Context圧縮 ==========
    context_dim = 256               # Context dimension (50% of hidden_size)

    # ========== 学習設定 ==========
    learning_rate = 1e-4            # Learning rate
    weight_decay = 0.01             # Weight decay
    warmup_steps = 100              # Warmup steps

    # ========== Phase 1 (OACD) ==========
    phase1_min_iterations = 5           # Minimum iterations (skip early stopping)
    phase1_max_iterations = 60          # Maximum iterations for OACD
    phase1_learning_rate = 0.002        # Phase 1 learning rate
    phase1_convergence_threshold = 0.001  # Convergence threshold (context change)
    phase1_early_stopping_rate = 0.90   # Early stop when convergence rate >= this
    phase1_no_improvement_patience = 3  # Stop if no improvement for N iterations
    phase1_batches_per_iteration = 10   # Batches per iteration
    phase1_checkpoint_path = "checkpoints/context_block_phase1.pt"  # Phase 1 checkpoint
    phase1_val_split = 0.1              # Validation split ratio

    # ========== Phase 2 (Fine-tuning) ==========
    phase2_epochs = 10              # Number of fine-tuning epochs
    phase2_learning_rate = 1e-4     # Phase 2 learning rate
    phase2_batch_size = 8           # Batch size (sequences)
    phase2_gradient_accumulation = 4  # Gradient accumulation steps

    # ========== データ ==========
    dataset_name = "EleutherAI/pile"  # Training dataset
    tokenizer_name = "EleutherAI/pythia-70m"  # Tokenizer
    max_seq_length = 512            # Maximum sequence length for training

    # ========== デバイス ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed = 42


class ContextPythiaConfig(PythiaConfig):
    """Context-Pythia 専用設定"""

    # Context-Pythia specific
    context_dim = 256               # 50% compression

    # KV cache reduction target
    kv_reduction_target = 0.5       # 50% reduction

    @classmethod
    def kv_cache_reduction(cls) -> float:
        """Calculate KV cache reduction ratio."""
        return 1 - (cls.context_dim / cls.hidden_size)
