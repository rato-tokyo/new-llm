"""
Data Loading Utilities for Pythia (Pile Dataset)

Pileデータセットのダウンロードとキャッシュ機能。
Pythiaと同じデータ（Pile）を使用する。
"""

import time
from pathlib import Path
from typing import Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub.utils import HfHubHTTPError

from src.utils.io import print_flush


# デフォルトのキャッシュディレクトリ
DEFAULT_CACHE_DIR = Path("cache/pile_tokens")


def get_cache_path(num_tokens: int, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """キャッシュファイルのパスを生成"""
    return cache_dir / f"pile_{num_tokens}.pt"


def load_pile_tokens_cached(
    num_tokens: int,
    tokenizer_name: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    max_retries: int = 5,
    retry_delay: float = 30.0,
) -> torch.Tensor:
    """
    Pileデータセットからトークンをロード（キャッシュ付き）

    Args:
        num_tokens: 必要なトークン数
        tokenizer_name: トークナイザー名
        cache_dir: キャッシュディレクトリ
        max_retries: 429エラー時の最大リトライ回数
        retry_delay: リトライ間の待機時間（秒）

    Returns:
        tokens: [num_tokens] の1D tensor
    """
    cache_path = get_cache_path(num_tokens, cache_dir)

    # キャッシュが存在する場合はロード
    if cache_path.exists():
        print_flush(f"Loading cached tokens: {cache_path}")
        tokens = torch.load(cache_path, weights_only=True)
        print_flush(f"  Loaded {tokens.numel():,} tokens from cache")
        return tokens

    # キャッシュがない場合はダウンロード
    print_flush(f"Downloading Pile dataset: {num_tokens:,} tokens")

    # Load tokenizer
    print_flush(f"  Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Pile dataset (streaming)
    print_flush("  Loading dataset (streaming)...")
    dataset = load_dataset(
        "monology/pile-uncopyrighted",
        split="train",
        streaming=True,
    )

    # Collect tokens with retry logic
    all_tokens: list[int] = []
    retry_count = 0

    print_flush("  Tokenizing...")

    dataset_iter = iter(dataset)
    while len(all_tokens) < num_tokens:
        try:
            example = next(dataset_iter)
            text = example["text"]
            if not text or len(text) < 100:
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)

            if len(all_tokens) % 100000 == 0 and len(all_tokens) > 0:
                print_flush(f"    Collected {len(all_tokens):,} tokens...")

            retry_count = 0  # Reset on success

        except StopIteration:
            break
        except HfHubHTTPError as e:
            if "429" in str(e) and retry_count < max_retries:
                retry_count += 1
                print_flush(f"    Rate limited (429). Retry {retry_count}/{max_retries} after {retry_delay}s...")
                time.sleep(retry_delay)
                dataset = load_dataset(
                    "monology/pile-uncopyrighted",
                    split="train",
                    streaming=True,
                )
                dataset_iter = iter(dataset)
                skip_count = len(all_tokens) // 500
                for _ in range(skip_count):
                    try:
                        next(dataset_iter)
                    except StopIteration:
                        break
            else:
                raise

    # Truncate to exact size
    all_tokens = all_tokens[:num_tokens]
    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)

    # Save to cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tokens_tensor, cache_path)
    print_flush(f"  Saved {tokens_tensor.numel():,} tokens to cache: {cache_path}")

    return tokens_tensor


def prepare_pythia_phase1_data(
    num_tokens: int,
    val_split: float,
    tokenizer_name: str,
    device: torch.device,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pythia Phase 1用のデータを準備（キャッシュ付き）

    Args:
        num_tokens: 必要なトークン数
        val_split: 検証データの割合
        tokenizer_name: トークナイザー名
        device: デバイス

    Returns:
        train_ids: [num_train_tokens]
        val_ids: [num_val_tokens]
    """
    # Train/Val分割
    val_tokens = max(1000, int(num_tokens * val_split))
    train_tokens = num_tokens - val_tokens
    total_tokens = num_tokens

    print_flush(f"Preparing Pythia Phase 1 data: {num_tokens:,} tokens")
    print_flush(f"  Train: {train_tokens:,} tokens")
    print_flush(f"  Val: {val_tokens:,} tokens")

    # キャッシュ付きでトークンをロード
    tokens = load_pile_tokens_cached(
        num_tokens=total_tokens,
        tokenizer_name=tokenizer_name,
        cache_dir=cache_dir,
    )

    # Split into train/val
    train_ids = tokens[:train_tokens].to(device)
    val_ids = tokens[train_tokens:].to(device)

    return train_ids, val_ids


def prepare_pythia_phase2_data(
    num_samples: int,
    seq_length: int,
    val_split: float,
    tokenizer_name: str,
    device: torch.device,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pythia Phase 2用のデータを準備（キャッシュ付き）

    Args:
        num_samples: サンプル数
        seq_length: シーケンス長
        val_split: 検証データの割合
        tokenizer_name: トークナイザー名
        device: デバイス

    Returns:
        train_inputs: [num_train, seq_length]
        train_targets: [num_train, seq_length]
        val_inputs: [num_val, seq_length]
        val_targets: [num_val, seq_length]
    """
    # 必要なトークン数 (input + target のため seq_length + 1)
    total_tokens = num_samples * (seq_length + 1)

    # Train/Val分割
    val_samples = max(1, int(num_samples * val_split))
    train_samples = num_samples - val_samples

    print_flush(f"Preparing Pythia Phase 2 data: {num_samples:,} samples, seq_len={seq_length}")
    print_flush(f"  Train: {train_samples:,} samples")
    print_flush(f"  Val: {val_samples:,} samples")

    # キャッシュ付きでトークンをロード
    tokens = load_pile_tokens_cached(
        num_tokens=total_tokens,
        tokenizer_name=tokenizer_name,
        cache_dir=cache_dir,
    )

    # Reshape to [num_samples, seq_length + 1]
    all_data = tokens.view(num_samples, seq_length + 1).to(device)

    # Split into inputs and targets
    inputs = all_data[:, :-1]  # [num_samples, seq_length]
    targets = all_data[:, 1:]  # [num_samples, seq_length]

    # Split into train/val
    train_inputs = inputs[:train_samples]
    train_targets = targets[:train_samples]
    val_inputs = inputs[train_samples:]
    val_targets = targets[train_samples:]

    return train_inputs, train_targets, val_inputs, val_targets
