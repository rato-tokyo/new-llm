"""UltraChat Dataset Loader with ChatML Formatting

Loads UltraChat dataset and converts to ChatML format:

<|im_start|>user
User message<|im_end|>
<|im_start|>assistant
Assistant response<|im_end|>

This format allows model to distinguish between user and assistant messages.
"""

from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import os


class UltraChatLoader:
    """
    Load UltraChat dataset and convert to ChatML format

    ChatML (Chat Markup Language):
    - <|im_start|>user\nUser message<|im_end|>
    - <|im_start|>assistant\nAssistant response<|im_end|>
    """

    def __init__(self, config):
        """
        Initialize loader

        Args:
            config: Configuration with:
                - dataset_name: HuggingFace dataset name
                - dataset_split: Split to use (train_sft, test_sft)
                - cache_dir: Cache directory
                - max_seq_length: Maximum sequence length
        """
        self.config = config
        self.cache_dir = config.cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load tokenizer (use standard tokenizer)
        # Using microsoft/DialoGPT-small as base tokenizer
        tokenizer_cache_path = os.path.join(self.cache_dir, "tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/DialoGPT-small",
            cache_dir=tokenizer_cache_path
        )

        # Add special tokens for ChatML
        special_tokens = {
            "additional_special_tokens": ["<|im_start|>", "<|im_end|>"]
        }
        self.tokenizer.add_special_tokens(special_tokens)

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Update vocab size in config
        config.vocab_size = len(self.tokenizer)

        print(f"Tokenizer loaded. Vocab size: {config.vocab_size}")

    def load_dataset(self, max_samples=None):
        """
        Load UltraChat dataset

        Args:
            max_samples: Maximum number of samples to load (None = all)

        Returns:
            dataset: HuggingFace dataset
        """
        print(f"Loading dataset: {self.config.dataset_name} ({self.config.dataset_split})")

        # Load dataset with caching
        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            cache_dir=self.cache_dir
        )

        # Limit samples if specified
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        print(f"Dataset loaded: {len(dataset)} samples")
        return dataset

    def convert_to_chatml(self, messages):
        """
        Convert messages to ChatML format

        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."}

        Returns:
            chatml_text: Formatted string
        """
        chatml_parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Format: <|im_start|>role\ncontent<|im_end|>
            chatml_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        return "\n".join(chatml_parts)

    def tokenize_dialogue(self, messages, max_length=None):
        """
        Convert messages to token IDs

        Args:
            messages: List of {"role": "user"/"assistant", "content": "..."}
            max_length: Maximum sequence length (None = use config)

        Returns:
            input_ids: Token IDs tensor [seq_len]
            user_mask: Mask indicating user tokens (for training) [seq_len]
        """
        if max_length is None:
            max_length = self.config.max_seq_length

        # Convert to ChatML
        chatml_text = self.convert_to_chatml(messages)

        # Tokenize
        encoding = self.tokenizer(
            chatml_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)  # [seq_len]

        # Create mask for assistant tokens (we only train on assistant responses)
        # This requires identifying which tokens belong to assistant
        user_mask = self._create_assistant_mask(messages, input_ids)

        return input_ids, user_mask

    def _create_assistant_mask(self, messages, input_ids):
        """
        Create mask indicating which tokens belong to assistant

        We only train on assistant tokens, not user tokens.

        Args:
            messages: Original messages
            input_ids: Tokenized IDs

        Returns:
            mask: Boolean tensor [seq_len] (True = train on this token)
        """
        # Reconstruct text with markers
        full_text = ""
        assistant_ranges = []  # [(start_char, end_char), ...]

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            marker_start = f"<|im_start|>{role}\n"
            marker_end = "<|im_end|>\n"

            start_pos = len(full_text) + len(marker_start)
            full_text += marker_start + content + marker_end
            end_pos = len(full_text) - len(marker_end)

            if role == "assistant":
                assistant_ranges.append((start_pos, end_pos))

        # Create character-level mask
        char_mask = [False] * len(full_text)
        for start, end in assistant_ranges:
            for i in range(start, end):
                char_mask[i] = True

        # Convert to token-level mask using tokenizer's char_to_token mapping
        # For simplicity, we'll use a heuristic:
        # If a token overlaps with assistant content, mark it as True
        mask = torch.zeros(len(input_ids), dtype=torch.bool)

        # Decode each token and check if it's in assistant range
        for i in range(len(input_ids)):
            token_text = self.tokenizer.decode([input_ids[i]])
            # Check if this token appears in assistant content
            # (This is a simplified heuristic)
            for msg in messages:
                if msg["role"] == "assistant" and token_text.strip() in msg["content"]:
                    mask[i] = True
                    break

        return mask

    def get_sample(self, dataset, idx):
        """
        Get a single processed sample

        Args:
            dataset: HuggingFace dataset
            idx: Sample index

        Returns:
            input_ids: Token IDs [seq_len]
            assistant_mask: Mask for assistant tokens [seq_len]
            messages: Original messages
        """
        sample = dataset[idx]
        messages = sample["messages"]

        input_ids, assistant_mask = self.tokenize_dialogue(messages)

        return input_ids, assistant_mask, messages

    def collate_batch(self, samples):
        """
        Collate multiple samples into a batch

        Args:
            samples: List of (input_ids, assistant_mask, messages)

        Returns:
            batch_input_ids: [batch, max_seq_len]
            batch_masks: [batch, max_seq_len]
        """
        # Find max length in batch
        max_len = max(len(sample[0]) for sample in samples)

        # Pad to max length
        batch_input_ids = []
        batch_masks = []

        for input_ids, mask, _ in samples:
            # Pad input_ids
            pad_len = max_len - len(input_ids)
            padded_ids = torch.cat([
                input_ids,
                torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)
            ])

            # Pad mask
            padded_mask = torch.cat([
                mask,
                torch.zeros(pad_len, dtype=torch.bool)
            ])

            batch_input_ids.append(padded_ids)
            batch_masks.append(padded_mask)

        batch_input_ids = torch.stack(batch_input_ids)
        batch_masks = torch.stack(batch_masks)

        return batch_input_ids, batch_masks
