"""
Reproducibility utilities for new-llm.

完全な再現性を保証するためのシード固定機能を提供。
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    全ての乱数生成器のシードを固定（完全な再現性保証）

    Args:
        seed: 乱数シード（デフォルト: 42）
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
