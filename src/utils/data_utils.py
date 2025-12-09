"""
Data Loading Utilities

日本語Wikipediaデータセットのダウンロードとキャッシュ機能。
"""

from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.utils.io import print_flush


# デフォルトのキャッシュディレクトリ
DEFAULT_CACHE_DIR = Path("cache/wiki_ja_tokens")


def get_cache_path(num_tokens: int, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """キャッシュファイルのパスを生成"""
    return cache_dir / f"wiki_ja_{num_tokens}.pt"


def load_wiki_ja_tokens_cached(
    num_tokens: int,
    tokenizer_name: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> torch.Tensor:
    """
    日本語Wikipediaデータセットからトークンをロード（キャッシュ付き）

    Args:
        num_tokens: 必要なトークン数
        tokenizer_name: トークナイザー名
        cache_dir: キャッシュディレクトリ

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
    print_flush(f"Downloading Japanese Wikipedia: {num_tokens:,} tokens")

    # Load tokenizer
    print_flush(f"  Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Japanese Wikipedia dataset (streaming)
    print_flush("  Loading dataset (streaming)...")
    dataset = load_dataset(
        "wikipedia",
        "20231101.ja",
        split="train",
        streaming=True,
    )

    # Collect tokens
    all_tokens: list[int] = []

    print_flush("  Tokenizing...")

    for example in dataset:
        text = example["text"]
        if not text or len(text) < 100:
            continue

        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)

        if len(all_tokens) % 100000 == 0 and len(all_tokens) > 0:
            print_flush(f"    Collected {len(all_tokens):,} tokens...")

        if len(all_tokens) >= num_tokens:
            break

    # Truncate to exact size
    all_tokens = all_tokens[:num_tokens]
    tokens_tensor = torch.tensor(all_tokens, dtype=torch.long)

    # Save to cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(tokens_tensor, cache_path)
    print_flush(f"  Saved {tokens_tensor.numel():,} tokens to cache: {cache_path}")

    return tokens_tensor


def load_long_documents_from_wiki_ja(
    tokenizer: PreTrainedTokenizer,
    num_docs: int,
    tokens_per_doc: int,
) -> list[torch.Tensor]:
    """
    日本語Wikipediaデータセットから長文ドキュメントをロード

    Args:
        tokenizer: トークナイザー
        num_docs: ドキュメント数
        tokens_per_doc: 各ドキュメントのトークン数

    Returns:
        documents: List of [tokens_per_doc] tensors
    """
    print_flush(f"Loading {num_docs} long documents ({tokens_per_doc} tokens each)...")

    dataset = load_dataset(
        "wikipedia",
        "20231101.ja",
        split="train",
        streaming=True,
    )

    documents: list[torch.Tensor] = []
    current_tokens: list[int] = []

    for example in dataset:
        text = example["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        current_tokens.extend(tokens)

        while len(current_tokens) >= tokens_per_doc:
            doc = current_tokens[:tokens_per_doc]
            documents.append(torch.tensor(doc, dtype=torch.long))
            current_tokens = current_tokens[tokens_per_doc:]

            if len(documents) >= num_docs:
                break

        if len(documents) >= num_docs:
            break

    print_flush(f"Loaded {len(documents)} documents")
    return documents
