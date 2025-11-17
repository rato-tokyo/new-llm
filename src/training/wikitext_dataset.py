"""WikiText dataset handling for language model pre-training"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import random


class WikiTextDataset(Dataset):
    """Dataset for WikiText language modeling

    WikiTextはWikipediaの記事から抽出されたテキストデータ。
    実際の文章の流れと文脈を学習するのに適している。
    """

    def __init__(self, texts: List[str], tokenizer, max_length: int = 64):
        """
        Args:
            texts: List of text strings from WikiText
            tokenizer: Tokenizer instance (SimpleTokenizer)
            max_length: Maximum sequence length (WikiTextは長いので64に拡張)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []

        # WikiTextは段落単位で処理
        for text in texts:
            # 空行や短すぎるテキストをスキップ
            if not text.strip() or len(text.split()) < 5:
                continue

            tokens = tokenizer.encode(text.strip())

            # 長いテキストは分割して複数のサンプルに
            if len(tokens) > max_length:
                # スライディングウィンドウで分割（オーバーラップあり）
                for i in range(0, len(tokens) - max_length + 1, max_length // 2):
                    chunk = tokens[i:i + max_length]
                    if len(chunk) == max_length:
                        self.sequences.append(chunk)
            else:
                # 短いテキストはパディング
                padded = tokens + [0] * (max_length - len(tokens))
                self.sequences.append(padded[:max_length])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Input is all tokens except last, target is all tokens except first
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])


def load_wikitext_data(config, wikitext_split='train') -> Tuple[WikiTextDataset, WikiTextDataset, object]:
    """Load WikiText-2 dataset using HuggingFace datasets

    Args:
        config: Configuration object with vocab_size, max_seq_length, etc.
        wikitext_split: 'train', 'validation', or 'test'

    Returns:
        train_dataset, val_dataset, tokenizer
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library required. Install with:\n"
            "pip install datasets"
        )

    # WikiText-2をロード（小規模版、約2MB）
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    # 訓練データとバリデーションデータを取得
    train_texts = [item['text'] for item in dataset['train']]
    val_texts = [item['text'] for item in dataset['validation']]

    # 空行を除去
    train_texts = [t for t in train_texts if t.strip()]
    val_texts = [t for t in val_texts if t.strip()]

    print(f"Loaded {len(train_texts)} training texts")
    print(f"Loaded {len(val_texts)} validation texts")

    # 既存のSimpleTokenizerを使用
    from .dataset import SimpleTokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(train_texts)

    # データセット作成
    train_dataset = WikiTextDataset(train_texts, tokenizer, config.max_seq_length)
    val_dataset = WikiTextDataset(val_texts, tokenizer, config.max_seq_length)

    print(f"Created {len(train_dataset)} training sequences")
    print(f"Created {len(val_dataset)} validation sequences")
    print(f"Vocabulary size: {len(tokenizer.word2idx)}")

    return train_dataset, val_dataset, tokenizer


def load_wikitext_103_data(config) -> Tuple[WikiTextDataset, WikiTextDataset, object]:
    """Load WikiText-103 dataset (larger version, ~181MB)

    WikiText-2で十分な結果が出た後に試す大規模版
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets library required. Install with:\n"
            "pip install datasets"
        )

    print("Loading WikiText-103 dataset (this may take a while)...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")

    train_texts = [item['text'] for item in dataset['train'] if item['text'].strip()]
    val_texts = [item['text'] for item in dataset['validation'] if item['text'].strip()]

    print(f"Loaded {len(train_texts)} training texts")
    print(f"Loaded {len(val_texts)} validation texts")

    from .dataset import SimpleTokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(train_texts)

    train_dataset = WikiTextDataset(train_texts, tokenizer, config.max_seq_length)
    val_dataset = WikiTextDataset(val_texts, tokenizer, config.max_seq_length)

    print(f"Created {len(train_dataset)} training sequences")
    print(f"Created {len(val_dataset)} validation sequences")

    return train_dataset, val_dataset, tokenizer
