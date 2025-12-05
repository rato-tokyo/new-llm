"""
Utility modules for New-LLM
"""

from .device import is_cuda_device, clear_gpu_cache, synchronize_device
from .io import print_flush
from .seed import set_seed
from .training import (
    prepare_data_loaders,
    train_epoch,
    evaluate,
    train_model,
    get_device,
    get_tokenizer,
)
from .evaluation import evaluate_ppl, evaluate_position_wise_ppl, evaluate_reversal_curse
from .rope import (
    RotaryEmbedding,
    CustomRotaryEmbedding,
    RoPEConfig,
    apply_rotary_pos_emb,
    create_rope_from_config,
    standard_rope_config,
    custom_frequencies_config,
    custom_list_config,
    linear_frequency_config,
    exponential_frequency_config,
)

__all__ = [
    # Device utilities
    'is_cuda_device',
    'clear_gpu_cache',
    'synchronize_device',
    # IO utilities
    'print_flush',
    # Seed utilities
    'set_seed',
    # Training utilities
    'prepare_data_loaders',
    'train_epoch',
    'evaluate',
    'train_model',
    'get_device',
    'get_tokenizer',
    # Evaluation utilities
    'evaluate_ppl',
    'evaluate_position_wise_ppl',
    'evaluate_reversal_curse',
    # RoPE utilities
    'RotaryEmbedding',
    'CustomRotaryEmbedding',
    'RoPEConfig',
    'apply_rotary_pos_emb',
    'create_rope_from_config',
    'standard_rope_config',
    'custom_frequencies_config',
    'custom_list_config',
    'linear_frequency_config',
    'exponential_frequency_config',
]
