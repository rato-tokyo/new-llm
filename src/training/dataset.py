"""Dataset handling for new-llm experiments"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import random


class SimpleTokenizer:
    """Simple word-level tokenizer"""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}
        self.next_idx = 4

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from texts"""
        word_freq = {}
        for text in texts:
            for word in text.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and take top vocab_size - 4 words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.vocab_size - 4]:
            if word not in self.word2idx:
                self.word2idx[word] = self.next_idx
                self.idx2word[self.next_idx] = word
                self.next_idx += 1

    def encode(self, text: str) -> List[int]:
        """Encode text to token indices"""
        tokens = [self.word2idx.get(word.lower(), 1) for word in text.split()]
        return [2] + tokens + [3]  # Add BOS and EOS

    def decode(self, indices: List[int]) -> str:
        """Decode token indices to text"""
        words = [self.idx2word.get(idx, "<UNK>") for idx in indices]
        return " ".join(words)


class TextDataset(Dataset):
    """Dataset for language modeling"""

    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, max_length: int = 32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []

        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) <= max_length:
                # Pad if necessary
                padded = tokens + [0] * (max_length - len(tokens))
                self.sequences.append(padded[:max_length])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        # Input is all tokens except last, target is all tokens except first
        return torch.tensor(seq[:-1]), torch.tensor(seq[1:])


def load_data(file_path: str, config) -> Tuple[TextDataset, TextDataset, SimpleTokenizer]:
    """Load and split data into train/val sets"""
    with open(file_path, 'r') as f:
        texts = [line.strip() for line in f if line.strip()]

    # Shuffle and split
    random.seed(config.random_seed)
    random.shuffle(texts)
    split_idx = int(len(texts) * config.train_split)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    # Build tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(train_texts)

    # Create datasets
    train_dataset = TextDataset(train_texts, tokenizer, config.max_seq_length)
    val_dataset = TextDataset(val_texts, tokenizer, config.max_seq_length)

    return train_dataset, val_dataset, tokenizer
