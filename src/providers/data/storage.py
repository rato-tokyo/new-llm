"""
StorageDataProvider - ストレージベース型データプロバイダー

mmapでディスクから直接読み込み（大規模向け）
"""

import os
import json
from typing import Tuple, Optional, Dict, Any
import torch
import numpy as np

from .base import DataProvider, print_flush


class StorageDataProvider(DataProvider):
    """ストレージベース型データプロバイダー"""

    def __init__(self, config):
        self.config = config
        self.storage_dir = config.disk_offload_dir
        self.use_bf16 = getattr(config, 'use_bf16', True)

        self._metadata: Optional[Dict[str, Any]] = None
        self._token_mmap: Optional[np.memmap] = None
        self._opened = False

        self.metadata_path = os.path.join(self.storage_dir, "metadata.json")
        self.token_path = os.path.join(self.storage_dir, "tokens", "token_ids.bin")

    def is_prepared(self) -> bool:
        return os.path.exists(self.metadata_path)

    def prepare(self, device: torch.device = None):
        """データを準備（ストリーミングダウンロード）"""
        from src.utils.streaming_loader import StreamingDataLoader

        if device is None:
            device = torch.device(self.config.device)

        loader = StreamingDataLoader(
            output_dir=self.storage_dir,
            num_samples=self.config.full_ultrachat_samples,
            use_bf16=self.use_bf16,
            chunk_size=getattr(self.config, 'streaming_chunk_size', 10_000)
        )
        return loader.prepare(device)

    def _load_metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            if not os.path.exists(self.metadata_path):
                raise FileNotFoundError(
                    f"Metadata not found: {self.metadata_path}\n"
                    "Run prepare() first to download and prepare data."
                )
            with open(self.metadata_path, 'r') as f:
                self._metadata = json.load(f)
        return self._metadata

    def _open_mmap(self):
        if self._opened:
            return

        metadata = self._load_metadata()
        num_tokens = metadata['num_tokens']

        self._token_mmap = np.memmap(
            self.token_path,
            dtype=np.int64,
            mode='r',
            shape=(num_tokens,)
        )
        self._opened = True

    def load_data(self) -> Tuple[None, None]:
        self._load_metadata()
        self._open_mmap()
        return None, None

    def get_num_train_tokens(self) -> int:
        metadata = self._load_metadata()
        split_ratio = getattr(self.config, 'train_val_split_ratio', 0.9)
        return int(metadata['num_tokens'] * split_ratio)

    def get_num_val_tokens(self) -> int:
        metadata = self._load_metadata()
        return metadata['num_tokens'] - self.get_num_train_tokens()

    def get_all_train_tokens(self, device: torch.device) -> torch.Tensor:
        self._open_mmap()
        num_train = self.get_num_train_tokens()
        chunk_np = self._token_mmap[:num_train].copy()
        return torch.from_numpy(chunk_np).to(device)

    def get_all_val_tokens(self, device: torch.device) -> torch.Tensor:
        self._open_mmap()
        num_train = self.get_num_train_tokens()
        metadata = self._load_metadata()
        num_total = metadata['num_tokens']
        chunk_np = self._token_mmap[num_train:num_total].copy()
        return torch.from_numpy(chunk_np).to(device)

    @property
    def is_streaming(self) -> bool:
        return True

    def close(self):
        if self._token_mmap is not None:
            del self._token_mmap
            self._token_mmap = None
        self._opened = False
