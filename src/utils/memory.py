"""
Memory Management Utilities

GPU/CPUメモリに基づく自動パラメータ調整
"""

import torch
from typing import Dict, Any


def get_gpu_memory_info() -> Dict[str, float]:
    """
    GPU メモリ情報を取得

    Returns:
        dict: {
            'total_gb': 総メモリ (GB),
            'allocated_gb': 使用中メモリ (GB),
            'free_gb': 空きメモリ (GB),
            'cached_gb': キャッシュメモリ (GB)
        }
    """
    if not torch.cuda.is_available():
        return {
            'total_gb': 0.0,
            'allocated_gb': 0.0,
            'free_gb': 0.0,
            'cached_gb': 0.0
        }

    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated() / (1024**3)
    cached = torch.cuda.memory_reserved() / (1024**3)
    free = total - allocated

    return {
        'total_gb': total,
        'allocated_gb': allocated,
        'free_gb': free,
        'cached_gb': cached
    }


def estimate_cache_size(
    num_tokens: int,
    num_layers: int,
    context_dim: int,
    embed_dim: int,
    num_input_tokens: int = 1
) -> Dict[str, float]:
    """
    キャッシュサイズを事前見積もり

    Args:
        num_tokens: トークン数
        num_layers: レイヤー数
        context_dim: コンテキスト次元
        embed_dim: 埋め込み次元
        num_input_tokens: 入力トークン数

    Returns:
        dict: {
            'context_cache_mb': コンテキストキャッシュ (MB),
            'token_embeds_mb': トークン埋め込み (MB),
            'total_mb': 合計 (MB),
            'total_gb': 合計 (GB)
        }
    """
    # context_cache: [num_layers, num_tokens, context_dim] (float32 = 4 bytes)
    context_cache_bytes = num_layers * num_tokens * context_dim * 4

    # token_embeds: [num_tokens, embed_dim * num_input_tokens] (float32)
    token_embeds_bytes = num_tokens * embed_dim * num_input_tokens * 4

    context_cache_mb = context_cache_bytes / (1024**2)
    token_embeds_mb = token_embeds_bytes / (1024**2)
    total_mb = context_cache_mb + token_embeds_mb
    total_gb = total_mb / 1024

    return {
        'context_cache_mb': context_cache_mb,
        'token_embeds_mb': token_embeds_mb,
        'total_mb': total_mb,
        'total_gb': total_gb
    }


def estimate_training_memory(
    batch_size: int,
    vocab_size: int = 50257,
    embed_dim: int = 768,
    num_layers: int = 6
) -> Dict[str, float]:
    """
    訓練時のメモリ使用量を見積もり

    Args:
        batch_size: バッチサイズ
        vocab_size: 語彙数
        embed_dim: 埋め込み次元
        num_layers: レイヤー数

    Returns:
        dict: 各コンポーネントのメモリ使用量 (MB)
    """
    # logits: [batch_size, vocab_size] (float32)
    logits_mb = batch_size * vocab_size * 4 / (1024**2)

    # gradients (roughly equal to logits + intermediate)
    gradients_mb = logits_mb * 2

    # intermediate activations (rough estimate)
    activations_mb = batch_size * embed_dim * num_layers * 4 / (1024**2) * 2

    total_mb = logits_mb + gradients_mb + activations_mb

    return {
        'logits_mb': logits_mb,
        'gradients_mb': gradients_mb,
        'activations_mb': activations_mb,
        'total_mb': total_mb,
        'per_token_mb': total_mb / batch_size if batch_size > 0 else 0
    }


def calculate_safe_batch_size(
    available_memory_gb: float,
    vocab_size: int = 50257,
    safety_factor: float = 0.5,
    min_batch_size: int = 256,
    max_batch_size: int = 16384
) -> int:
    """
    利用可能メモリから安全なバッチサイズを計算

    Args:
        available_memory_gb: 利用可能メモリ (GB)
        vocab_size: 語彙数
        safety_factor: 安全係数 (0.0-1.0)
        min_batch_size: 最小バッチサイズ
        max_batch_size: 最大バッチサイズ

    Returns:
        int: 安全なバッチサイズ
    """
    # 1トークンあたりのメモリ使用量 (MB)
    # logits(50257 * 4) + gradients ≈ 0.4MB/token
    per_token_mb = vocab_size * 4 / (1024**2) * 2.5  # 2.5x for safety

    available_mb = available_memory_gb * 1024 * safety_factor
    safe_batch_size = int(available_mb / per_token_mb)

    # 範囲制限
    safe_batch_size = max(min_batch_size, min(safe_batch_size, max_batch_size))

    return safe_batch_size


def can_fit_in_memory(
    train_tokens: int,
    val_tokens: int,
    num_layers: int,
    context_dim: int,
    embed_dim: int,
    num_input_tokens: int = 1,
    model_size_gb: float = 0.5,
    required_batch_memory_gb: float = 1.0
) -> Dict[str, Any]:
    """
    データがGPUメモリに収まるか確認

    Args:
        train_tokens: 訓練トークン数
        val_tokens: 検証トークン数
        num_layers: レイヤー数
        context_dim: コンテキスト次元
        embed_dim: 埋め込み次元
        num_input_tokens: 入力トークン数
        model_size_gb: モデルサイズの見積もり (GB)
        required_batch_memory_gb: バッチ処理に必要なメモリ (GB)

    Returns:
        dict: {
            'fits': True/False,
            'total_required_gb': 必要メモリ合計,
            'available_gb': 利用可能メモリ,
            'train_cache_gb': 訓練キャッシュサイズ,
            'val_cache_gb': 検証キャッシュサイズ,
            'recommendation': 推奨事項
        }
    """
    gpu_info = get_gpu_memory_info()

    if gpu_info['total_gb'] == 0:
        return {
            'fits': True,  # CPUでは常にTrue（遅いが動く）
            'total_required_gb': 0,
            'available_gb': 0,
            'recommendation': 'CPU mode - memory managed by system'
        }

    # キャッシュサイズ計算
    train_cache = estimate_cache_size(
        train_tokens, num_layers, context_dim, embed_dim, num_input_tokens
    )
    val_cache = estimate_cache_size(
        val_tokens, num_layers, context_dim, embed_dim, num_input_tokens
    )

    train_cache_gb = train_cache['total_gb']
    val_cache_gb = val_cache['total_gb']

    # 必要メモリ合計
    total_required = (
        model_size_gb +
        train_cache_gb +
        val_cache_gb +
        required_batch_memory_gb
    )

    # 利用可能メモリ（80%を使用可能と仮定）
    available = gpu_info['total_gb'] * 0.8

    fits = total_required <= available

    # 推奨事項
    if fits:
        recommendation = f"OK: {total_required:.1f}GB required, {available:.1f}GB available"
    else:
        deficit = total_required - available
        recommendation = (
            f"WARNING: {total_required:.1f}GB required, {available:.1f}GB available. "
            f"Deficit: {deficit:.1f}GB. "
            f"Consider reducing train samples or using streaming."
        )

    return {
        'fits': fits,
        'total_required_gb': total_required,
        'available_gb': available,
        'train_cache_gb': train_cache_gb,
        'val_cache_gb': val_cache_gb,
        'model_size_gb': model_size_gb,
        'recommendation': recommendation
    }


def print_memory_report(
    train_tokens: int,
    val_tokens: int,
    num_layers: int,
    context_dim: int,
    embed_dim: int,
    num_input_tokens: int = 1
):
    """
    メモリレポートを表示
    """
    import sys

    def print_flush(msg):
        print(msg, flush=True)
        sys.stdout.flush()

    print_flush("\n" + "="*60)
    print_flush("MEMORY REPORT")
    print_flush("="*60)

    # GPU情報
    gpu_info = get_gpu_memory_info()
    if gpu_info['total_gb'] > 0:
        print_flush(f"GPU Total: {gpu_info['total_gb']:.1f}GB")
        print_flush(f"GPU Allocated: {gpu_info['allocated_gb']:.1f}GB")
        print_flush(f"GPU Free: {gpu_info['free_gb']:.1f}GB")
    else:
        print_flush("GPU: Not available (CPU mode)")

    print_flush("")

    # キャッシュ見積もり
    train_cache = estimate_cache_size(
        train_tokens, num_layers, context_dim, embed_dim, num_input_tokens
    )
    val_cache = estimate_cache_size(
        val_tokens, num_layers, context_dim, embed_dim, num_input_tokens
    )

    print_flush(f"Train tokens: {train_tokens:,}")
    print_flush(f"  Context cache: {train_cache['context_cache_mb']:.1f}MB")
    print_flush(f"  Token embeds: {train_cache['token_embeds_mb']:.1f}MB")
    print_flush(f"  Total: {train_cache['total_mb']:.1f}MB ({train_cache['total_gb']:.2f}GB)")

    print_flush(f"\nVal tokens: {val_tokens:,}")
    print_flush(f"  Context cache: {val_cache['context_cache_mb']:.1f}MB")
    print_flush(f"  Token embeds: {val_cache['token_embeds_mb']:.1f}MB")
    print_flush(f"  Total: {val_cache['total_mb']:.1f}MB ({val_cache['total_gb']:.2f}GB)")

    print_flush(f"\nTotal cache: {train_cache['total_gb'] + val_cache['total_gb']:.2f}GB")

    # 適合チェック
    check = can_fit_in_memory(
        train_tokens, val_tokens, num_layers, context_dim, embed_dim, num_input_tokens
    )
    print_flush(f"\n{check['recommendation']}")
    print_flush("="*60 + "\n")

    return check
