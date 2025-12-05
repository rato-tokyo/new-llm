"""
Data Loading Utilities for Pythia (Pile Dataset)

Pileデータセットのダウンロードとキャッシュ機能。
Pythiaと同じデータ（Pile）を使用する。
"""

import time
from pathlib import Path

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
