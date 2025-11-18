"""Anthropic HH-RLHF dataset handling for dialog training

HH-RLHF (Human-Human RLHF) contains high-quality human conversations
with human feedback for helpful and harmless AI assistants.

Dataset: Anthropic/hh-rlhf
- Helpful dialogues: ~43k samples
- Harmless dialogues: ~42k samples
- English only
- Multi-turn conversations
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from datasets import load_dataset


class HHRLHFDataset(Dataset):
    """Dataset for Anthropic HH-RLHF dialog training

    Format:
        "Human: {question}\n\nAssistant: {response}\n\nHuman: ..."

    Uses "chosen" responses (human-preferred, high-quality)
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
            # Use "chosen" response (human-preferred)
            text = example['chosen']

            # Clean up text (remove extra whitespace)
            text = ' '.join(text.split())

            tokens = tokenizer.encode(text)

            # Truncate or pad to max_length
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                # Pad with 0 (PAD token)
                tokens = tokens + [0] * (max_length - len(tokens))

            self.sequences.append(tokens)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Input is all tokens except last, target is all tokens except first
        # This is for language modeling (predict next token)
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])


def load_hh_rlhf_data(config, subset='helpful') -> Tuple[HHRLHFDataset, HHRLHFDataset, object]:
    """Load Anthropic HH-RLHF dataset using HuggingFace datasets

    Args:
        config: Configuration object with vocab_size, max_seq_length, etc.
        subset: 'helpful' or 'harmless' (default: 'helpful')

    Returns:
        train_dataset, val_dataset, tokenizer
    """
    print(f"Loading HH-RLHF dataset (subset: {subset})...")

    # Load HH-RLHF dataset
    # Note: Different subsets available
    # - helpful-base: Basic helpful conversations
    # - helpful-online: Online helpful conversations
    # - helpful-rejection-sampled: With rejection sampling
    # - harmless-base: Harmless conversations

    dataset_name = f"Anthropic/hh-rlhf"

    # For simplicity, we'll use the full dataset which has train/test splits
    dataset = load_dataset(dataset_name)

    # HH-RLHF structure:
    # - train split
    # - test split (use as validation)
    train_data = dataset['train']
    val_data = dataset['test']

    print(f"Loaded {len(train_data)} training examples")
    print(f"Loaded {len(val_data)} validation examples")

    # Build vocabulary from training data
    # Extract text from "chosen" responses
    train_texts = []
    for example in train_data:
        # Use chosen response
        text = example['chosen']
        text = ' '.join(text.split())  # Clean whitespace
        train_texts.append(text)

    # Use SimpleTokenizer from existing code
    from .dataset import SimpleTokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(train_texts)

    print(f"Built vocabulary: {len(tokenizer.word2idx)} words")

    # Create datasets
    train_dataset = HHRLHFDataset(train_data, tokenizer, config.max_seq_length)
    val_dataset = HHRLHFDataset(val_data, tokenizer, config.max_seq_length)

    print(f"Created {len(train_dataset)} training sequences")
    print(f"Created {len(val_dataset)} validation sequences")

    return train_dataset, val_dataset, tokenizer
