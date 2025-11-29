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
        self._sample_boundaries: Optional[List[Tuple[int, int]]] = None  # サンプル境界情報
        self._loaded = False

    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._loaded:
            # _loaded が True なら必ず値が設定されている
            assert self._train_token_ids is not None
            assert self._val_token_ids is not None
            return self._train_token_ids, self._val_token_ids

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name,
            cache_dir=os.path.join(self.config.cache_dir, "tokenizer")
        )
        tokenizer.pad_token = tokenizer.eos_token

        print_flush("Loading training data...")
        self._train_token_ids, self._sample_order, self._sample_boundaries = self._load_train_data(tokenizer)

        print_flush("Loading validation data...")
        self._val_token_ids = self._load_val_data(tokenizer)

        print_flush(f"  Train: {len(self._train_token_ids)} tokens ({len(self._sample_boundaries)} samples)")
        print_flush(f"  Val:   {len(self._val_token_ids)} tokens")
        if self.shuffle_samples:
            print_flush(f"  Shuffle: enabled (seed={self.shuffle_seed})")

        self._loaded = True
        return self._train_token_ids, self._val_token_ids

    def _load_train_data(self, tokenizer) -> Tuple[torch.Tensor, List[int], List[Tuple[int, int]]]:
        """訓練データをロード（UltraChat）"""
        shuffle_suffix = f"_shuffle{self.shuffle_seed}" if self.shuffle_samples else ""
        cache_file = os.path.join(
            self.config.cache_dir,
            f"ultrachat_{self.config.num_samples}samples_full{shuffle_suffix}.pt"
        )

        if os.path.exists(cache_file):
            print_flush(f"  Loading from cache: {cache_file}")
            cached = torch.load(cache_file)
            if isinstance(cached, dict):
                token_ids = cached['token_ids']
                sample_order = cached['sample_order']
                # サンプル境界情報（キャッシュに含まれていない場合は再計算が必要）
                if 'sample_boundaries' in cached:
                    sample_boundaries = cached['sample_boundaries']
                else:
                    # 古いキャッシュの場合、境界情報なし → 再生成が必要
                    print_flush("  Warning: Cache missing sample boundaries, regenerating...")
                    os.remove(cache_file)
                    return self._load_train_data(tokenizer)
                return token_ids, sample_order, sample_boundaries
            else:
                # 非常に古いキャッシュ形式
                print_flush("  Warning: Old cache format, regenerating...")
                os.remove(cache_file)
                return self._load_train_data(tokenizer)

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
        sample_boundaries = []  # 各サンプルの(start, end)インデックス
        current_pos = 0

        for idx in sample_order:
            messages = dataset[idx]["messages"]
            text = "\n".join([msg["content"] for msg in messages])

            tokens = tokenizer(
                text,
                truncation=False,
                return_tensors="pt"
            )
            sample_tokens = tokens["input_ids"].squeeze(0)
            all_token_ids.append(sample_tokens)

            # サンプル境界を記録
            sample_len = len(sample_tokens)
            sample_boundaries.append((current_pos, current_pos + sample_len))
            current_pos += sample_len

        token_ids = torch.cat(all_token_ids)

        os.makedirs(self.config.cache_dir, exist_ok=True)
        torch.save({
            'token_ids': token_ids,
            'sample_order': sample_order,
            'sample_boundaries': sample_boundaries
        }, cache_file)
        print_flush(f"  Cached to: {cache_file}")

        return token_ids, sample_order, sample_boundaries

    def _load_val_data(self, tokenizer) -> torch.Tensor:
        """検証データをロード（テキストファイル、存在しない場合は自動生成）"""
        if self.config.val_data_source == "auto_split":
            raise ValueError(
                "auto_split is FORBIDDEN for validation data!\n"
                "Use val_data_source='text_file' with val_text_file path."
            )

        file_path = self.config.val_text_file
        if not os.path.exists(file_path):
            self._generate_val_file(file_path, tokenizer)

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        tokens = tokenizer(text, truncation=False, return_tensors="pt")
        return tokens["input_ids"].squeeze(0)

    def _generate_val_file(self, file_path: str, tokenizer) -> None:
        """検証データファイルを自動生成（UltraChatのサンプル1000-1020）"""
        print_flush(f"  Validation file not found, generating: {file_path}")

        # ディレクトリ作成
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)

        # UltraChatから検証データを生成
        dataset = load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            cache_dir=os.path.join(self.config.cache_dir, "datasets")
        )

        # サンプル1000-1020を使用（訓練データ0-999と重複しない）
        val_start_idx = 1000
        val_end_idx = min(1020, len(dataset))

        val_texts = []
        for idx in range(val_start_idx, val_end_idx):
            messages = dataset[idx]["messages"]
            text = "\n".join([msg["content"] for msg in messages])
            val_texts.append(text)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(val_texts))

        print_flush(f"  Generated {len(val_texts)} validation samples (indices {val_start_idx}-{val_end_idx-1})")

    def get_sample_order(self) -> Optional[List[int]]:
        """現在のサンプル順序を取得"""
        return self._sample_order

    def get_sample_boundaries(self) -> Optional[List[Tuple[int, int]]]:
        """サンプル境界情報を取得

        Returns:
            List of (start, end) tuples for each sample
            例: [(0, 128), (128, 256), ...]
        """
        if not self._loaded:
            self.load_data()
        return self._sample_boundaries

    def get_split_token_ids(self, split_id: int, num_splits: int, device: torch.device) -> torch.Tensor:
        """
        指定されたsplit用のトークンIDを取得

        サンプル単位で分割: sample_id % num_splits == split_id のサンプルのみ

        Args:
            split_id: 分割ID (0 to num_splits-1)
            num_splits: 総分割数
            device: デバイス

        Returns:
            該当するサンプルのトークンID
        """
        if not self._loaded:
            self.load_data()

        if self._sample_boundaries is None:
            raise ValueError("Sample boundaries not available")
        if self._train_token_ids is None:
            raise ValueError("Train token IDs not available")

        # このsplitに属するサンプルを抽出
        split_tokens = []
        for sample_idx, (start, end) in enumerate(self._sample_boundaries):
            if sample_idx % num_splits == split_id:
                split_tokens.append(self._train_token_ids[start:end])

        if len(split_tokens) == 0:
            raise ValueError(f"No samples found for split_id={split_id}")

        return torch.cat(split_tokens).to(device)

    def reshuffle(self, new_seed: Optional[int] = None) -> torch.Tensor:
        """
        サンプルを再シャッフルして訓練データを再生成

        シンプルに現在ロード済みのトークンIDを並び替えるだけ。
        サンプル単位ではなく、全体のトークン順序をシャッフル。

        Args:
            new_seed: 新しいシード（Noneの場合は現在のシード+1）

        Returns:
            再シャッフル後の訓練トークンID
        """
        if not self._loaded:
            self.load_data()

        if new_seed is not None:
            self.shuffle_seed = new_seed
        else:
            self.shuffle_seed += 1

        # 現在のトークンIDをシャッフル
        import random
        rng = random.Random(self.shuffle_seed)

        if self._train_token_ids is None:
            raise ValueError("Train token IDs not available")

        indices = list(range(len(self._train_token_ids)))
        rng.shuffle(indices)
        self._train_token_ids = self._train_token_ids[indices]
        self._sample_order = indices  # シャッフル順序を記録

        print_flush(f"Reshuffled tokens with seed {self.shuffle_seed}")
        return self._train_token_ids

    def get_num_train_tokens(self) -> int:
        if not self._loaded:
            self.load_data()
        if self._train_token_ids is None:
            raise ValueError("Train token IDs not available")
        return len(self._train_token_ids)

    def get_num_val_tokens(self) -> int:
        if not self._loaded:
            self.load_data()
        if self._val_token_ids is None:
            raise ValueError("Val token IDs not available")
        return len(self._val_token_ids)

    def get_all_train_tokens(self, device: torch.device) -> torch.Tensor:
        if not self._loaded:
            self.load_data()
        if self._train_token_ids is None:
            raise ValueError("Train token IDs not available")
        return self._train_token_ids.to(device)

    def get_all_val_tokens(self, device: torch.device) -> torch.Tensor:
        if not self._loaded:
            self.load_data()
        if self._val_token_ids is None:
            raise ValueError("Val token IDs not available")
        return self._val_token_ids.to(device)

    @property
    def is_streaming(self) -> bool:
        return False

    def close(self):
        self._train_token_ids = None
        self._val_token_ids = None
        self._sample_order = None
        self._sample_boundaries = None
        self._loaded = False
