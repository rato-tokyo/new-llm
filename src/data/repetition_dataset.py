"""Repetition Dataset Generator for Context Vector Convergence Training

This module generates special training data where the same phrase is repeated
many times to test the hypothesis: context(n) ≈ context(n+1) for large n.

Example:
    Input:  context("red" * 100) + token("red")
    Output: context("red" * 101) ≈ context("red" * 100)

The goal is to train the context update mechanism to reach a stable fixed point.
"""

import torch
from torch.utils.data import Dataset
from typing import List, Tuple


class RepetitionDataset(Dataset):
    """Dataset that repeats a phrase to test context vector convergence

    Args:
        phrases: List of phrases to repeat (e.g., ["red", "red apple"])
        repetitions: Number of times to repeat each phrase
        tokenizer: Tokenizer to encode phrases
        max_length: Maximum sequence length
    """

    def __init__(
        self,
        phrases: List[str],
        repetitions: int = 100,
        tokenizer=None,
        max_length: int = 512
    ):
        self.phrases = phrases
        self.repetitions = repetitions
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Generate all sequences
        self.sequences = self._generate_sequences()

    def _generate_sequences(self) -> List[List[int]]:
        """Generate repeated sequences for all phrases"""
        sequences = []

        for phrase in self.phrases:
            # Tokenize the phrase
            phrase_tokens = self.tokenizer.encode(phrase)
            phrase_len = len(phrase_tokens)

            # Repeat the phrase
            # Create sequence: [phrase] * repetitions
            repeated_tokens = phrase_tokens * self.repetitions

            # Truncate to max_length if necessary
            if len(repeated_tokens) > self.max_length:
                repeated_tokens = repeated_tokens[:self.max_length]

            sequences.append(repeated_tokens)

        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        """Return a sequence of repeated tokens"""
        tokens = self.sequences[idx]
        return torch.tensor(tokens, dtype=torch.long)


def generate_repetition_phrases(
    num_tokens: int = 1,
    vocabulary: List[str] = None
) -> List[str]:
    """Generate phrases of specified token length

    Args:
        num_tokens: Number of tokens in each phrase (1, 2, 3, ...)
        vocabulary: List of words to use. If None, uses default set.

    Returns:
        List of phrases

    Examples:
        num_tokens=1: ["red", "blue", "green"]
        num_tokens=2: ["red apple", "blue sky", "green tree"]
        num_tokens=3: ["red apple tree", "blue sky cloud", ...]
    """
    if vocabulary is None:
        # Default vocabulary (simple, high-frequency words)
        vocabulary = [
            "red", "blue", "green", "yellow", "black", "white",
            "apple", "sky", "tree", "cloud", "river", "mountain",
            "cat", "dog", "bird", "fish", "lion", "tiger",
            "big", "small", "fast", "slow", "hot", "cold"
        ]

    if num_tokens == 1:
        # Single tokens
        return vocabulary[:10]  # Use first 10 words

    elif num_tokens == 2:
        # Two-token phrases
        adjectives = ["red", "blue", "green", "big", "small"]
        nouns = ["apple", "sky", "tree", "cat", "dog"]
        return [f"{adj} {noun}" for adj in adjectives for noun in nouns][:10]

    elif num_tokens == 3:
        # Three-token phrases
        adjectives = ["red", "blue", "green"]
        nouns1 = ["apple", "sky", "tree"]
        nouns2 = ["tree", "cloud", "mountain"]
        return [f"{adj} {n1} {n2}"
                for adj in adjectives
                for n1 in nouns1
                for n2 in nouns2][:10]

    else:
        # For longer phrases, just concatenate words
        phrases = []
        for i in range(10):
            phrase_words = vocabulary[i:i+num_tokens]
            if len(phrase_words) < num_tokens:
                phrase_words = vocabulary[:num_tokens]
            phrases.append(" ".join(phrase_words))
        return phrases


def create_staged_datasets(
    tokenizer,
    max_stage: int = 5,
    repetitions: int = 100,
    max_length: int = 512
) -> List[Tuple[int, RepetitionDataset]]:
    """Create datasets for staged training (1-token → multi-token)

    Args:
        tokenizer: Tokenizer instance
        max_stage: Maximum number of tokens per phrase
        repetitions: How many times to repeat each phrase
        max_length: Maximum sequence length

    Returns:
        List of (stage_num, dataset) tuples

    Example:
        Stage 1: ["red"] * 100
        Stage 2: ["red apple"] * 100
        Stage 3: ["red apple tree"] * 100
        ...
    """
    datasets = []

    for stage in range(1, max_stage + 1):
        # Generate phrases for this stage
        phrases = generate_repetition_phrases(num_tokens=stage)

        # Create dataset
        dataset = RepetitionDataset(
            phrases=phrases,
            repetitions=repetitions,
            tokenizer=tokenizer,
            max_length=max_length
        )

        datasets.append((stage, dataset))

    return datasets


def collate_fn(batch):
    """Collate function for DataLoader

    Pads sequences to same length within batch
    """
    # batch is a list of tensors
    max_len = max(seq.size(0) for seq in batch)

    padded = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, seq in enumerate(batch):
        padded[i, :seq.size(0)] = seq

    return padded
