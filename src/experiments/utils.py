"""
実験用ユーティリティ関数
"""

import random
import sys

import numpy as np
import torch


def print_flush(msg: str) -> None:
    """即時フラッシュ付きprint"""
    print(msg, flush=True)
    sys.stdout.flush()


def set_seed(seed: int = 42) -> None:
    """再現性のためのシード固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def format_time(seconds: float) -> str:
    """秒を読みやすい形式に変換"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}h"


def format_number(n: int) -> str:
    """数値を読みやすい形式に変換"""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    elif n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f}K"
    else:
        return str(n)


def get_device() -> torch.device:
    """利用可能なデバイスを取得"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_flush(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = torch.device("cpu")
        print_flush("WARNING: Running on CPU")
    return device


def cleanup_memory(device: torch.device) -> None:
    """メモリをクリーンアップ"""
    import gc

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
