"""
Validation Data Generator

This script generates validation text data using the same vocabulary as training data.

Workflow:
1. Load training data and extract unique tokens
2. Decode training data to understand content themes
3. Generate new validation text using the same vocabulary
4. Encode and save validation token IDs
5. Verify no unseen tokens in validation data

Usage:
    python3 scripts/generate_validation_data.py
"""

import os
import sys
import torch
from tokenizers import Tokenizer
from collections import Counter

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import ResidualConfig


def load_training_data(config):
    """Load training data and tokenizer"""
    # Load tokenizer
    tokenizer_path = os.path.join(config.cache_dir, "tokenizer", "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Run train.py first.")

    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Load training data
    train_data_path = os.path.join(config.cache_dir, "ultrachat_5samples_128len.pt")
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data not found at {train_data_path}. Run train.py first.")

    train_token_ids = torch.load(train_data_path)

    return tokenizer, train_token_ids


def analyze_training_vocabulary(train_token_ids, tokenizer):
    """Analyze training data vocabulary"""
    # Convert to list
    token_list = train_token_ids.tolist()

    # Get unique tokens
    unique_tokens = sorted(set(token_list))

    # Token frequency
    token_counts = Counter(token_list)

    # Decode training text
    train_text = tokenizer.decode(token_list)

    print(f"\n{'='*70}")
    print("Training Data Analysis")
    print(f"{'='*70}\n")
    print(f"Total tokens: {len(token_list):,}")
    print(f"Unique tokens: {len(unique_tokens):,}")
    print(f"\nSample text (first 500 chars):")
    print(f"{train_text[:500]}\n...")

    return unique_tokens, token_counts, train_text


def generate_validation_text_instructions(unique_tokens, token_counts, train_text):
    """Generate instructions for validation text creation"""

    # Get most common tokens
    most_common = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:30]

    instructions = f"""
VALIDATION TEXT GENERATION INSTRUCTIONS
{'='*70}

Goal: Generate English text suitable for validation that uses ONLY the tokens from training data.

Vocabulary Constraints:
- Total unique tokens available: {len(unique_tokens)}
- Must use ONLY these tokens (no new vocabulary)

Content Guidelines:
- Topic: Mix of technology, landmarks, sustainability, and general knowledge
- Length: Approximately 160 tokens (~800 characters)
- Style: Similar to training data (informative, varied topics)
- Requirement: Must be DIFFERENT from training text (natural diversity)

Training Data Sample Themes:
{train_text[:800]}

Most Common Tokens to Use Frequently:
"""

    for token_id, count in most_common[:20]:
        # This is a workaround since we can't directly use tokenizer here
        instructions += f"  Token {token_id} ({count}x in train)\n"

    instructions += """

Output Format:
- Plain English text
- Natural sentences and paragraphs
- No special formatting or markup
- Focus on coherent, readable content

Example Topics to Cover:
1. Technology features or innovations
2. Travel destinations or landmarks
3. Environmental sustainability
4. General knowledge or advice

IMPORTANT: The text will be tokenized and checked - it MUST only contain tokens from the training vocabulary!
"""

    return instructions


def save_validation_text_manual(output_path):
    """Save instructions for manual validation text creation"""

    manual_instructions = """# Validation Text Generation - Manual Mode

Since automatic text generation with vocabulary constraints is complex,
please manually create validation text following these guidelines:

## Requirements:
1. Write 3-4 short paragraphs (approximately 160 tokens total)
2. Topics: Technology, travel, sustainability, or general knowledge
3. Style: Informative and varied (similar to training data)
4. Avoid copying training text exactly

## Content Ideas:
- Describe a technology feature or tool
- Recommend tourist attractions in a city
- Discuss environmental initiatives
- Give advice on a topic

## Save Location:
Save your text to: data/example_val.txt

## Verification:
After saving, run this script again to verify token compatibility.
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(manual_instructions)

    print(f"\n✅ Saved manual instructions to: {output_path}")


def encode_and_verify_validation_data(tokenizer, train_token_ids, val_text_path, config):
    """Encode validation text and verify compatibility"""

    if not os.path.exists(val_text_path):
        print(f"\n⚠️  Validation text file not found: {val_text_path}")
        print("Please create the file first using the manual instructions.")
        return None

    # Load validation text
    with open(val_text_path, 'r', encoding='utf-8') as f:
        val_text = f.read()

    print(f"\n{'='*70}")
    print("Validation Text Verification")
    print(f"{'='*70}\n")

    # Encode
    encoding = tokenizer.encode(val_text)
    val_token_ids = torch.tensor(encoding.ids, dtype=torch.long)

    print(f"Validation text length: {len(val_text)} characters")
    print(f"Validation tokens: {len(val_token_ids)}")

    # Check compatibility
    train_set = set(train_token_ids.tolist())
    val_set = set(val_token_ids.tolist())

    unseen_tokens = val_set - train_set

    if unseen_tokens:
        print(f"\n❌ INCOMPATIBLE: Found {len(unseen_tokens)} tokens not in training data!")
        print(f"Unseen token IDs: {sorted(list(unseen_tokens))[:20]}")
        print("\nPlease revise the validation text to use only training vocabulary.")
        return None

    # Calculate overlap
    overlap_ratio = len(val_set & train_set) / len(val_set) * 100

    print(f"\n✅ COMPATIBLE: All validation tokens exist in training data!")
    print(f"Unique tokens in validation: {len(val_set)}")
    print(f"Overlap with training: {overlap_ratio:.1f}%")

    # Save encoded validation data
    val_output_path = os.path.join(config.cache_dir, "manual_val_tokens.pt")
    torch.save(val_token_ids, val_output_path)

    print(f"\n💾 Saved validation token IDs to: {val_output_path}")

    return val_token_ids


def main():
    """Main generation function"""

    print(f"\n{'='*70}")
    print("Validation Data Generator")
    print(f"{'='*70}\n")

    # Load configuration
    config = ResidualConfig()

    # Load training data
    try:
        tokenizer, train_token_ids = load_training_data(config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # Analyze vocabulary
    unique_tokens, token_counts, train_text = analyze_training_vocabulary(
        train_token_ids, tokenizer
    )

    # Generate instructions
    instructions = generate_validation_text_instructions(
        unique_tokens, token_counts, train_text
    )

    # Save instructions
    instructions_path = os.path.join(project_root, "data", "validation_generation_guide.txt")
    os.makedirs(os.path.dirname(instructions_path), exist_ok=True)

    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(instructions)

    print(f"\n✅ Saved generation instructions to: {instructions_path}")

    # Check if validation text exists
    val_text_path = os.path.join(project_root, "data", "example_val.txt")

    if os.path.exists(val_text_path):
        # Verify existing validation data
        val_token_ids = encode_and_verify_validation_data(
            tokenizer, train_token_ids, val_text_path, config
        )

        if val_token_ids is not None:
            print(f"\n{'='*70}")
            print("✅ Validation data is ready to use!")
            print(f"{'='*70}\n")
            print("Update config.py:")
            print('  val_data_source = "text_file"')
            print('  val_text_file = "./data/example_val.txt"')
    else:
        print(f"\n{'='*70}")
        print("Next Steps")
        print(f"{'='*70}\n")
        print("1. Review the generation instructions:")
        print(f"   {instructions_path}")
        print(f"\n2. Create validation text file:")
        print(f"   {val_text_path}")
        print(f"\n3. Run this script again to verify compatibility")
        print(f"\n4. Update config.py to use the validation data")


if __name__ == "__main__":
    main()
