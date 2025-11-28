"""
実験用データローダー
"""

from typing import List, Optional

import torch

from src.experiments.utils import print_flush


class UltraChatDataLoader:
    """
    UltraChatデータセットのローダー

    サンプル単位でデータを管理し、訓練/検証分割を柔軟に行う。
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.sample_tokens: List[torch.Tensor] = []
        self.sample_boundaries: List[int] = []
        self.all_tokens: Optional[torch.Tensor] = None
        self._loaded_samples = 0

    def load_all(self, total_samples: int) -> None:
        """
        全サンプルをロード

        Args:
            total_samples: ロードするサンプル数（train + val）
        """
        print_flush(f"\n  Loading {total_samples} samples from UltraChat...")

        from datasets import load_dataset
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

        self.sample_tokens = []
        self.sample_boundaries = [0]

        for i in range(total_samples):
            text = dataset[i]["messages"][0]["content"]
            tokens = tokenizer(text, return_tensors="pt")
            self.sample_tokens.append(tokens["input_ids"].squeeze(0))
            self.sample_boundaries.append(
                self.sample_boundaries[-1] + len(self.sample_tokens[-1])
            )

        self.all_tokens = torch.cat(self.sample_tokens).to(self.device)
        self._loaded_samples = total_samples
        print_flush(f"  Total tokens: {len(self.all_tokens):,}")
        print_flush(f"  Samples loaded: {total_samples}")

    def get_train(self, num_samples: int) -> torch.Tensor:
        """
        訓練データを取得

        Args:
            num_samples: 訓練に使用するサンプル数

        Returns:
            訓練トークン列
        """
        if self.all_tokens is None:
            raise RuntimeError("Data not loaded. Call load_all() first.")
        end_idx = self.sample_boundaries[num_samples]
        return self.all_tokens[:end_idx]

    def get_val(self, train_samples: int, val_samples: int) -> torch.Tensor:
        """
        検証データを取得

        Args:
            train_samples: 訓練サンプル数（検証データの開始位置）
            val_samples: 検証サンプル数

        Returns:
            検証トークン列
        """
        if self.all_tokens is None:
            raise RuntimeError("Data not loaded. Call load_all() first.")
        start_idx = self.sample_boundaries[train_samples]
        end_idx = self.sample_boundaries[train_samples + val_samples]
        return self.all_tokens[start_idx:end_idx]

    @property
    def total_tokens(self) -> int:
        """総トークン数"""
        if self.all_tokens is None:
            return 0
        return len(self.all_tokens)

    @property
    def loaded_samples(self) -> int:
        """ロード済みサンプル数"""
        return self._loaded_samples
