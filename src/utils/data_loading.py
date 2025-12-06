"""
Unified Data Loading Utilities

各種データセットのロード機能を統一。
- Pile: 長文ドキュメント
- WikiText-2: 標準ベンチマーク
"""

import torch
from datasets import load_dataset
from transformers import PreTrainedTokenizer

from src.utils.io import print_flush


def load_long_documents_from_pile(
    tokenizer: PreTrainedTokenizer,
    num_docs: int,
    tokens_per_doc: int,
) -> list[torch.Tensor]:
    """
    Pileデータセットから長文ドキュメントをロード

    Args:
        tokenizer: トークナイザー
        num_docs: ドキュメント数
        tokens_per_doc: 各ドキュメントのトークン数

    Returns:
        documents: List of [tokens_per_doc] tensors
    """
    print_flush(f"Loading {num_docs} long documents ({tokens_per_doc} tokens each)...")

    dataset = load_dataset(
        "monology/pile-uncopyrighted",
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


def load_wikitext2(
    tokenizer: PreTrainedTokenizer,
    split: str = "test",
) -> torch.Tensor:
    """
    WikiText-2データセットをロード

    Args:
        tokenizer: トークナイザー
        split: "train", "validation", or "test"

    Returns:
        tokens: 1D tensor of all tokens
    """
    print_flush(f"Loading WikiText-2 ({split})...")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # Concatenate all text
    all_text = "\n".join(dataset["text"])

    # Tokenize
    tokens = tokenizer.encode(all_text, add_special_tokens=False)
    tokens = torch.tensor(tokens, dtype=torch.long)

    print_flush(f"Loaded {len(tokens):,} tokens")
    return tokens


def split_documents(
    documents: list[torch.Tensor],
    val_ratio: float = 0.1,
    min_val_size: int = 10,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    ドキュメントリストをtrain/valに分割

    Args:
        documents: ドキュメントリスト
        val_ratio: 検証用の割合
        min_val_size: 最小検証データ数

    Returns:
        (train_docs, val_docs)
    """
    val_size = max(min_val_size, int(len(documents) * val_ratio))
    train_docs = documents[:-val_size]
    val_docs = documents[-val_size:]
    print_flush(f"Train: {len(train_docs)}, Val: {len(val_docs)}")
    return train_docs, val_docs


def split_tokens(
    tokens: torch.Tensor,
    val_ratio: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    トークン列をtrain/valに分割

    Args:
        tokens: 1D tensor
        val_ratio: 検証用の割合

    Returns:
        (train_tokens, val_tokens)
    """
    val_size = int(len(tokens) * val_ratio)
    train_tokens = tokens[:-val_size]
    val_tokens = tokens[-val_size:]
    print_flush(f"Train: {len(train_tokens):,}, Val: {len(val_tokens):,}")
    return train_tokens, val_tokens
