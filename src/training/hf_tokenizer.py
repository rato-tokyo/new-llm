"""
HuggingFace Tokenizer creation for New-LLM

Uses industry-standard BPE tokenization instead of custom implementation.
This eliminates tokenizer-related bugs and enables:
- Proper subword tokenization
- Better handling of rare words
- Automatic serialization/deserialization
- Integration with HF ecosystem
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast


def create_tokenizer(
    texts,
    vocab_size=10000,
    min_frequency=2,
    special_tokens=None
):
    """Create a BPE tokenizer trained on the given texts

    Args:
        texts: List of text strings to train on
        vocab_size: Target vocabulary size (default: 10000)
        min_frequency: Minimum frequency for a token (default: 2)
        special_tokens: List of special tokens (default: standard set)

    Returns:
        PreTrainedTokenizerFast: HuggingFace tokenizer ready to use
    """
    if special_tokens is None:
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    # Create BPE tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))

    # Use whitespace pre-tokenizer
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Create trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True
    )

    # Train tokenizer
    print(f"Training BPE tokenizer on {len(texts):,} texts...")
    print(f"  Vocab size: {vocab_size}")
    print(f"  Min frequency: {min_frequency}")

    tokenizer.train_from_iterator(texts, trainer)

    # Add post-processor for special tokens
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<BOS> $A <EOS>",
        special_tokens=[
            ("<BOS>", tokenizer.token_to_id("<BOS>")),
            ("<EOS>", tokenizer.token_to_id("<EOS>")),
        ],
    )

    # Wrap in HuggingFace tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<UNK>",
        pad_token="<PAD>",
        bos_token="<BOS>",
        eos_token="<EOS>",
        model_max_length=512
    )

    print(f"✓ Tokenizer created: {len(hf_tokenizer)} tokens")

    return hf_tokenizer


def load_tokenizer(tokenizer_path):
    """Load a saved tokenizer

    Args:
        tokenizer_path: Path to tokenizer directory

    Returns:
        PreTrainedTokenizerFast: Loaded tokenizer
    """
    return PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
