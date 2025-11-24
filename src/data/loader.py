"""
Unified data loading for New-LLM

Supports multiple data sources:
- UltraChat dataset
- Text files
- Text directories
- Manual validation data
"""

import torch
import os
from transformers import AutoTokenizer


def load_data(config):
    """
    Load training and validation data based on configuration.

    Args:
        config: Configuration object with data settings

    Returns:
        tuple: (train_token_ids, val_token_ids) as torch tensors
    """
    print_flush("Loading training data...")
    train_token_ids = load_data_source(
        source_type=config.train_data_source,
        config=config,
        is_training=True
    )

    print_flush("Loading validation data...")
    if config.val_data_source == "auto_split":
        raise ValueError(
            "❌ CRITICAL ERROR: auto_split is STRICTLY FORBIDDEN for validation data!\n"
            "\n"
            "Correct specification:\n"
            "  - Validation data MUST contain ONLY tokens that appear in training data\n"
            "  - Use val_data_source='text_file' with manually generated validation text\n"
            "  - Generate validation data using: python3 scripts/generate_validation_data.py\n"
            "\n"
            "Why auto_split is forbidden:\n"
            "  - It splits training data randomly, creating arbitrary train/val sets\n"
            "  - Does NOT guarantee validation uses only training vocabulary\n"
            "  - Violates the fundamental requirement of vocabulary consistency\n"
            "\n"
            "Required config.py setting:\n"
            "  val_data_source = 'text_file'\n"
            "  val_text_file = './data/example_val.txt'\n"
        )
    else:
        val_token_ids = load_data_source(
            source_type=config.val_data_source,
            config=config,
            is_training=False
        )

    print_flush(f"  Train: {len(train_token_ids)} tokens")
    print_flush(f"  Val:   {len(val_token_ids)} tokens ({config.val_data_source})")

    return train_token_ids, val_token_ids


def load_data_source(source_type, config, is_training=True):
    """
    Load data from a specific source type.

    Args:
        source_type: Type of data source ("ultrachat", "text_file", "text_dir", "manual")
        config: Configuration object
        is_training: Whether loading training data (for file paths)

    Returns:
        torch.Tensor: Token IDs
    """
    if source_type == "ultrachat":
        return load_ultrachat_data(config)
    elif source_type == "text_file":
        file_path = config.train_text_file if is_training else config.val_text_file
        return load_text_file(file_path, config)
    elif source_type == "text_dir":
        dir_path = config.train_text_dir if is_training else config.val_text_dir
        return load_text_directory(dir_path, config)
    elif source_type == "manual":
        return load_manual_validation(config)
    else:
        raise ValueError(f"Unknown data source: {source_type}")


def load_ultrachat_data(config):
    """Load data from UltraChat dataset"""
    from datasets import load_dataset

    print_flush(f"Loading {config.num_samples} samples from UltraChat...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        cache_dir=os.path.join(config.cache_dir, "tokenizer")
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Check for cached data
    cache_file = os.path.join(
        config.cache_dir,
        f"ultrachat_{config.num_samples}samples_{config.max_seq_length}len.pt"
    )

    if os.path.exists(cache_file):
        print_flush(f"  Loading from cache: {cache_file}")
        token_ids = torch.load(cache_file)
    else:
        # Load dataset
        dataset = load_dataset(
            config.dataset_name,
            split=config.dataset_split,
            cache_dir=os.path.join(config.cache_dir, "datasets")
        )

        # Process samples
        all_token_ids = []
        for idx in range(min(config.num_samples, len(dataset))):
            messages = dataset[idx]["messages"]
            text = "\n".join([msg["content"] for msg in messages])

            # Tokenize
            tokens = tokenizer(
                text,
                max_length=config.max_seq_length,
                truncation=True,
                return_tensors="pt"
            )
            all_token_ids.append(tokens["input_ids"].squeeze(0))

        # Concatenate all tokens
        token_ids = torch.cat(all_token_ids)

        # Save cache
        os.makedirs(config.cache_dir, exist_ok=True)
        torch.save(token_ids, cache_file)
        print_flush(f"  Cached to: {cache_file}")

    print_flush(f"  Loaded {len(all_token_ids) if 'all_token_ids' in locals() else config.num_samples} text segments → {len(token_ids)} tokens")
    return token_ids


def load_text_file(file_path, config):
    """Load data from a text file"""
    print_flush(f"Loading text file: {file_path}")

    if not os.path.exists(file_path):
        print_flush(f"  Warning: File not found, creating empty data")
        return torch.zeros(100, dtype=torch.long)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        cache_dir=os.path.join(config.cache_dir, "tokenizer")
    )

    # Read and tokenize
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # No truncation for validation data - we want all tokens
    tokens = tokenizer(
        text,
        truncation=False,
        return_tensors="pt"
    )

    token_ids = tokens["input_ids"].squeeze(0)
    print_flush(f"  Loaded {len(token_ids)} tokens")
    return token_ids


def load_text_directory(dir_path, config):
    """Load data from all text files in a directory"""
    print_flush(f"Loading text files from: {dir_path}")

    if not os.path.exists(dir_path):
        print_flush(f"  Warning: Directory not found, creating empty data")
        return torch.zeros(100, dtype=torch.long)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",
        cache_dir=os.path.join(config.cache_dir, "tokenizer")
    )

    # Process all text files
    all_token_ids = []
    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith('.txt'):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            tokens = tokenizer(
                text,
                max_length=config.max_seq_length,
                truncation=True,
                return_tensors="pt"
            )
            all_token_ids.append(tokens["input_ids"].squeeze(0))

    if not all_token_ids:
        print_flush(f"  Warning: No text files found, creating empty data")
        return torch.zeros(100, dtype=torch.long)

    token_ids = torch.cat(all_token_ids)
    print_flush(f"  Loaded {len(all_token_ids)} files → {len(token_ids)} tokens")
    return token_ids


def load_manual_validation(config):
    """Load manually created validation data"""
    manual_path = config.manual_val_path

    if os.path.exists(manual_path):
        print_flush(f"Loading manual validation data: {manual_path}")
        token_ids = torch.load(manual_path)
        print_flush(f"  Loaded {len(token_ids)} validation tokens")
        return token_ids
    else:
        print_flush(f"  Warning: Manual validation file not found: {manual_path}")
        print_flush(f"  Creating small default validation data")
        return torch.randint(0, 1000, (100,))


def print_flush(msg):
    """Print with immediate flush"""
    print(msg, flush=True)