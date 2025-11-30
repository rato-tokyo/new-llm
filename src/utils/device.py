"""
Device utilities for new-llm.

Provides centralized device detection and GPU memory management.
"""

from typing import Union

import torch


def is_cuda_device(device: Union[str, torch.device]) -> bool:
    """
    Check if the device is CUDA.

    Args:
        device: Device string or torch.device object

    Returns:
        True if device is CUDA, False otherwise
    """
    if isinstance(device, str):
        return device == 'cuda'
    return hasattr(device, 'type') and device.type == 'cuda'


def clear_gpu_cache(device: Union[str, torch.device]) -> None:
    """
    Clear GPU cache if device is CUDA.

    Args:
        device: Device string or torch.device object
    """
    if is_cuda_device(device):
        torch.cuda.empty_cache()


def synchronize_device(device: Union[str, torch.device]) -> None:
    """
    Synchronize CUDA device if applicable.

    Args:
        device: Device string or torch.device object
    """
    if is_cuda_device(device):
        torch.cuda.synchronize()
