"""
MemoryDataProvider - メモリ展開型データプロバイダー

全データをメモリに展開（小〜中規模向け）
サンプルシャッフル機能付き
"""

import os
from typing import Tuple, Optional, List
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from .base import DataProvider, print_flush


class MemoryDataProvider(DataProvider):
    """メモリ展開型データプロバイダー（シャッフル機能付き）"""

    def __init__(self, config, shuffle_samples: bool = False, shuffle_seed: int = 42):
        """
        Args:
            config: ResidualConfig
            shuffle_samples: サンプルをシャッフルするか
            shuffle_seed: シャッフル用乱数シード
        """
        self.config = config
        self.shuffle_samples = shuffle_samples
        self.shuffle_seed = shuffle_seed

        self._train_token_ids: Optional[torch.Tensor] = None
        self._val_token_ids: Optional[torch.Tensor] = None
        self._sample_order: Optional[List[int]] = None
        self._loaded = False

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._loaded:
            return self._train_token_ids, self._val_token_ids

        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            cache_dir=os.path.join(self.config.cache_dir, "tokenizer")
        )
        tokenizer.pad_token = tokenizer.eos_token

        print_flush("Loading training data...")
        self._train_token_ids, self._sample_order = self._load_train_data(tokenizer)

        print_flush("Loading validation data...")
        self._val_token_ids = self._load_val_data(tokenizer)

        print_flush(f"  Train: {len(self._train_token_ids)} tokens")
        print_flush(f"  Val:   {len(self._val_token_ids)} tokens")
        if self.shuffle_samples:
            print_flush(f"  Shuffle: enabled (seed={self.shuffle_seed})")

        self._loaded = True
        return self._train_token_ids, self._val_token_ids

    def _load_train_data(self, tokenizer) -> Tuple[torch.Tensor, List[int]]:
        """訓練データをロード（UltraChat）"""
        shuffle_suffix = f"_shuffle{self.shuffle_seed}" if self.shuffle_samples else ""
        cache_file = os.path.join(
            self.config.cache_dir,
            f"ultrachat_{self.config.num_samples}samples_{self.config.max_seq_length}len{shuffle_suffix}.pt"
        )

        if os.path.exists(cache_file):
            print_flush(f"  Loading from cache: {cache_file}")
            cached = torch.load(cache_file)
            if isinstance(cached, dict):
                return cached['token_ids'], cached['sample_order']
            else:
                return cached, list(range(self.config.num_samples))

        print_flush(f"  Loading {self.config.num_samples} samples from UltraChat...")
        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            cache_dir=os.path.join(self.config.cache_dir, "datasets")
        )

        num_samples = min(self.config.num_samples, len(dataset))
        sample_order = list(range(num_samples))

        if self.shuffle_samples:
            import random
            rng = random.Random(self.shuffle_seed)
            rng.shuffle(sample_order)
            print_flush(f"  Shuffled {num_samples} samples with seed {self.shuffle_seed}")

        all_token_ids = []
        for idx in sample_order:
            messages = dataset[idx]["messages"]
            text = "\n".join([msg["content"] for msg in messages])

            tokens = tokenizer(
                text,
                max_length=self.config.max_seq_length,
                truncation=True,
                return_tensors="pt"
            )
            all_token_ids.append(tokens["input_ids"].squeeze(0))

        token_ids = torch.cat(all_token_ids)

        os.makedirs(self.config.cache_dir, exist_ok=True)
        torch.save({'token_ids': token_ids, 'sample_order': sample_order}, cache_file)
        print_flush(f"  Cached to: {cache_file}")

        return token_ids, sample_order

    def _load_val_data(self, tokenizer) -> torch.Tensor:
        """検証データをロード（テキストファイル）"""
        if self.config.val_data_source == "auto_split":
            raise ValueError(
                "auto_split is FORBIDDEN for validation data!\n"
                "Use val_data_source='text_file' with val_text_file path."
            )

        file_path = self.config.val_text_file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Validation file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        tokens = tokenizer(text, truncation=False, return_tensors="pt")
        return tokens["input_ids"].squeeze(0)

    def get_sample_order(self) -> Optional[List[int]]:
        """現在のサンプル順序を取得"""
        return self._sample_order

    def reshuffle(self, new_seed: Optional[int] = None) -> torch.Tensor:
        """
        サンプルを再シャッフルして訓練データを再生成

        Args:
            new_seed: 新しいシード（Noneの場合は現在のシード+1）

        Returns:
            再シャッフル後の訓練トークンID
        """
        if new_seed is not None:
            self.shuffle_seed = new_seed
        else:
            self.shuffle_seed += 1

        self.shuffle_samples = True
        self._loaded = False
        self._train_token_ids = None
        self._sample_order = None

        # キャッシュをバイパスして再ロード
        tokenizer = AutoTokenizer.from_pretrained(
            "gpt2",
            cache_dir=os.path.join(self.config.cache_dir, "tokenizer")
        )
        tokenizer.pad_token = tokenizer.eos_token

        print_flush(f"Reshuffling with seed {self.shuffle_seed}...")
        self._train_token_ids, self._sample_order = self._load_train_data(tokenizer)

        return self._train_token_ids

    def get_num_train_tokens(self) -> int:
        if not self._loaded:
            self.load_data()
        return len(self._train_token_ids)

    def get_num_val_tokens(self) -> int:
        if not self._loaded:
            self.load_data()
        return len(self._val_token_ids)

    def get_all_train_tokens(self, device: torch.device) -> torch.Tensor:
        if not self._loaded:
            self.load_data()
        return self._train_token_ids.to(device)

    def get_all_val_tokens(self, device: torch.device) -> torch.Tensor:
        if not self._loaded:
            self.load_data()
        return self._val_token_ids.to(device)

    @property
    def is_streaming(self) -> bool:
        return False

    def close(self):
        self._train_token_ids = None
        self._val_token_ids = None
        self._sample_order = None
        self._loaded = False
