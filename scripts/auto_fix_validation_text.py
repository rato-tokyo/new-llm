"""
Automatic Validation Text Fixer

Iteratively fixes validation text by replacing unknown tokens with placeholders,
then uses Claude to suggest replacements from training vocabulary.

Usage:
    python3 scripts/auto_fix_validation_text.py
"""

import os
import sys
import torch
from tokenizers import Tokenizer

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import ResidualConfig


def load_training_vocabulary(config):
    """Load training data and extract vocabulary"""
    tokenizer_path = os.path.join(config.cache_dir, "tokenizer", "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    train_data_path = os.path.join(config.cache_dir, "ultrachat_5samples_128len.pt")
    train_token_ids = torch.load(train_data_path)

    train_vocab_set = set(train_token_ids.tolist())

    return tokenizer, train_vocab_set


def check_and_mark_unknown_tokens(text, tokenizer, train_vocab_set):
    """
    Check text for unknown tokens and mark them with [UNKNOWN_XXX]

    Returns:
        marked_text: Text with unknown tokens marked
        unknown_words: List of (word, token_id) tuples
        is_valid: True if no unknown tokens
    """
    # Encode text
    encoding = tokenizer.encode(text)
    token_ids = encoding.ids
    tokens = encoding.tokens

    # Find unknown tokens
    unknown_tokens = []
    for i, (token_id, token_str) in enumerate(zip(token_ids, tokens)):
        if token_id not in train_vocab_set:
            unknown_tokens.append((i, token_id, token_str))

    if not unknown_tokens:
        return text, [], True

    # Mark unknown tokens in text
    marked_text = text
    unknown_words = []

    for idx, token_id, token_str in unknown_tokens:
        # Find unique words that decode to this token
        word = token_str.replace('Ġ', ' ').strip()
        if word and word not in [uw[0] for uw in unknown_words]:
            unknown_words.append((word, token_id))
            # Replace in text (case-sensitive)
            marked_text = marked_text.replace(word, f"[UNKNOWN_{len(unknown_words)}]")

    return marked_text, unknown_words, False


def save_replacement_guide(unknown_words, tokenizer, train_vocab_set, output_path):
    """
    Save a guide file for Claude to help with replacements

    Returns:
        guide_path: Path to the guide file
    """
    # Get some common words from training vocab for reference
    common_tokens = list(train_vocab_set)[:100]
    common_words = [tokenizer.decode([tid]) for tid in common_tokens]

    guide_content = f"""# Validation Text Replacement Guide

## Unknown Words to Replace

The following words are NOT in the training vocabulary and need replacement:

"""

    for i, (word, token_id) in enumerate(unknown_words, 1):
        guide_content += f"{i}. [UNKNOWN_{i}] = '{word}' (token_id: {token_id})\n"

    guide_content += f"""

## Instructions for Replacement

For each [UNKNOWN_X] placeholder, suggest a replacement word/phrase from the training vocabulary that:
1. Has similar meaning to the original word
2. Fits grammatically in the sentence
3. Uses ONLY tokens from the training vocabulary

## Available Common Words (Sample)

Here are some common words available in the training vocabulary:

{', '.join([f'"{w.strip()}"' for w in common_words[:50] if w.strip()])}

## Full Vocabulary Reference

See: cache/train_vocab.txt for the complete list of 371 available tokens.

## Output Format

Provide replacements in this format:
[UNKNOWN_1] -> "suggested replacement"
[UNKNOWN_2] -> "another replacement"
...

"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(guide_content)

    return output_path


def apply_replacements(marked_text, replacements):
    """
    Apply replacement suggestions to marked text

    Args:
        marked_text: Text with [UNKNOWN_X] markers
        replacements: Dict mapping [UNKNOWN_X] to replacement text

    Returns:
        fixed_text: Text with replacements applied
    """
    fixed_text = marked_text
    for marker, replacement in replacements.items():
        fixed_text = fixed_text.replace(marker, replacement)

    return fixed_text


def main():
    """Main fixing function"""

    print(f"\n{'='*70}")
    print("Automatic Validation Text Fixer")
    print(f"{'='*70}\n")

    # Load configuration
    config = ResidualConfig()

    # Load training vocabulary
    print("Loading training vocabulary...")
    tokenizer, train_vocab_set = load_training_vocabulary(config)
    print(f"  Training vocabulary: {len(train_vocab_set)} unique tokens\n")

    # Check if validation text exists
    val_text_path = os.path.join(project_root, "data", "example_val.txt")

    if not os.path.exists(val_text_path):
        print(f"❌ Validation text not found: {val_text_path}")
        print("Please create a draft validation text first.\n")
        return

    # Load validation text
    with open(val_text_path, 'r', encoding='utf-8') as f:
        original_text = f.read()

    print(f"Original validation text ({len(original_text)} chars):")
    print(f"{original_text[:200]}...\n")

    # Check and mark unknown tokens
    print("Checking for unknown tokens...")
    marked_text, unknown_words, is_valid = check_and_mark_unknown_tokens(
        original_text, tokenizer, train_vocab_set
    )

    if is_valid:
        print("✅ Validation text is already compatible!")
        print("All tokens are in the training vocabulary.\n")
        return

    print(f"❌ Found {len(unknown_words)} unknown words:\n")
    for i, (word, token_id) in enumerate(unknown_words, 1):
        print(f"  {i}. '{word}' (token_id: {token_id})")

    # Save marked text
    marked_text_path = os.path.join(project_root, "data", "example_val_marked.txt")
    with open(marked_text_path, 'w', encoding='utf-8') as f:
        f.write(marked_text)

    print(f"\n💾 Saved marked text to: {marked_text_path}")

    # Save replacement guide
    guide_path = os.path.join(project_root, "data", "replacement_guide.txt")
    save_replacement_guide(unknown_words, tokenizer, train_vocab_set, guide_path)

    print(f"💾 Saved replacement guide to: {guide_path}")

    print(f"\n{'='*70}")
    print("Next Steps")
    print(f"{'='*70}\n")

    print("1. Review the marked text:")
    print(f"   {marked_text_path}")
    print(f"\n2. Review the replacement guide:")
    print(f"   {guide_path}")
    print(f"\n3. Ask Claude to suggest replacements for each [UNKNOWN_X]")
    print(f"\n4. Apply replacements manually or via script")
    print(f"\n5. Run this script again to verify (repeat until valid)")

    print(f"\nMarked text preview:")
    print(f"{marked_text[:300]}...\n")


if __name__ == "__main__":
    main()
