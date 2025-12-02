"""
Utility modules
"""

from .memory import (
    get_gpu_memory_info,
    estimate_cache_size,
    estimate_training_memory,
    calculate_safe_batch_size,
    calculate_optimal_batch_size,
    can_fit_in_memory,
    print_memory_report
)
from .initialization import init_linear_weights, count_parameters
from .device import is_cuda_device, clear_gpu_cache, synchronize_device
from .cache import (
    collect_context_cache_sequential,
    collect_context_cache_sequential_multiblock,
)

__all__ = [
    # Memory utilities
    'get_gpu_memory_info',
    'estimate_cache_size',
    'estimate_training_memory',
    'calculate_safe_batch_size',
    'calculate_optimal_batch_size',
    'can_fit_in_memory',
    'print_memory_report',
    # Initialization utilities
    'init_linear_weights',
    'count_parameters',
    # Device utilities
    'is_cuda_device',
    'clear_gpu_cache',
    'synchronize_device',
    # Cache utilities
    'collect_context_cache_sequential',
    'collect_context_cache_sequential_multiblock',
]
