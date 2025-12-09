"""
Utility modules for Senri
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
)
from .tokenizer_utils import get_tokenizer
from .evaluation import (
    evaluate_ppl,
    evaluate_position_wise_ppl,
)
from .data_utils import (
    load_wiki_ja_tokens_cached,
    load_long_documents_from_wiki_ja,
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
    # Data loading (Japanese Wikipedia)
    'load_wiki_ja_tokens_cached',
    'load_long_documents_from_wiki_ja',
]
