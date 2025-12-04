"""
Random seed utilities for reproducibility.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.

    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDNN deterministic mode

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
