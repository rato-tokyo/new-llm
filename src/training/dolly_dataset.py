"""Dolly-15k dataset handling for instruction finetuning"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from datasets import load_dataset


class DollyDataset(Dataset):
    """Dataset for Dolly-15k instruction finetuning

    Format:
        Input: "Instruction: {instruction}\nContext: {context}\nResponse: {response}"
    """

    def __init__(self, data, tokenizer, max_length: int = 128):
        """
        Args:
            data: HuggingFace dataset split (train or validation)
            tokenizer: SimpleTokenizer instance
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []

        for example in data:
            text = self._format_example(example)
            tokens = tokenizer.encode(text)

            # Truncate or pad to max_length
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                # Pad with 0 (PAD token)
                tokens = tokens + [0] * (max_length - len(tokens))

            self.sequences.append(tokens)

    def _format_example(self, example):
        """Format Dolly example as instruction-response string

        Format:
            Instruction: {instruction}
            Context: {context}  (if present)
            Response: {response}
        """
        parts = [f"Instruction: {example['instruction']}"]

        # Add context if present and not empty
        if example.get('context', '').strip():
            parts.append(f"Context: {example['context']}")

        parts.append(f"Response: {example['response']}")

        return " ".join(parts)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Input is all tokens except last, target is all tokens except first
        # This is for language modeling (predict next token)
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])


def load_dolly_data(config) -> Tuple[DollyDataset, DollyDataset, object]:
    """Load Dolly-15k dataset using HuggingFace datasets

    Args:
        config: Configuration object with vocab_size, max_seq_length, etc.

    Returns:
        train_dataset, val_dataset, tokenizer
    """
    print("Loading Dolly-15k dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k")

    # Dolly-15k has only 'train' split, so we split it ourselves
    # 90% train, 10% validation
    train_test_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
    train_data = train_test_split['train']
    val_data = train_test_split['test']

    print(f"Loaded {len(train_data)} training examples")
    print(f"Loaded {len(val_data)} validation examples")

    # Build vocabulary from training data
    # We need to extract all text first
    train_texts = []
    for example in train_data:
        # Format example
        parts = [example['instruction']]
        if example.get('context', '').strip():
            parts.append(example['context'])
        parts.append(example['response'])
        train_texts.append(" ".join(parts))

    # Use SimpleTokenizer from WikiText-2
    from .dataset import SimpleTokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(train_texts)

    print(f"Built vocabulary: {len(tokenizer.word2idx)} words")

    # Create datasets
    train_dataset = DollyDataset(train_data, tokenizer, config.max_seq_length)
    val_dataset = DollyDataset(val_data, tokenizer, config.max_seq_length)

    print(f"Created {len(train_dataset)} training sequences")
    print(f"Created {len(val_dataset)} validation sequences")

    return train_dataset, val_dataset, tokenizer
