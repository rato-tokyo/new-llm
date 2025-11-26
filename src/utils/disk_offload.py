"""
ディスクオフロードユーティリティ

大規模データセットのPhase 1訓練用。
メモリマップファイルとダブルバッファリングを管理。
"""

import os
import numpy as np
import torch
from typing import Optional, Tuple


class ContextSwapper:
    """
    ダブルバッファリングによるコンテキスト管理。

    2つのmmapファイル（a, b）を交互に使用し、
    イテレーション間でポインタを交換することで物理コピーを回避。
    """

    def __init__(
        self,
        storage_dir: str,
        num_tokens: int,
        context_dim: int,
        use_bf16: bool = True
    ):
        """
        Args:
            storage_dir: mmapファイルを保存するディレクトリ
            num_tokens: 総トークン数
            context_dim: コンテキスト次元数
            use_bf16: bf16精度を使用するか
        """
        self.storage_dir = storage_dir
        self.num_tokens = num_tokens
        self.context_dim = context_dim
        self.use_bf16 = use_bf16

        # bf16はnumpyでサポートされていないため、float16を使用
        # PyTorchでbf16に変換する
        self.np_dtype = np.float16 if use_bf16 else np.float32
        self.torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

        self.contexts_dir = os.path.join(storage_dir, "contexts")
        os.makedirs(self.contexts_dir, exist_ok=True)

        self.file_a = os.path.join(self.contexts_dir, "contexts_a.bin")
        self.file_b = os.path.join(self.contexts_dir, "contexts_b.bin")

        self._mmap_a: Optional[np.memmap] = None
        self._mmap_b: Optional[np.memmap] = None
        self._current_is_a = True

    def create_storage(self) -> Tuple[str, str]:
        """
        ストレージファイルを作成（ゼロ初期化）。

        Returns:
            (file_a_path, file_b_path)
        """
        shape = (self.num_tokens, self.context_dim)
        bytes_per_element = 2 if self.use_bf16 else 4
        total_bytes = self.num_tokens * self.context_dim * bytes_per_element

        print(f"Creating context storage:")
        print(f"  Shape: {shape}")
        print(f"  Dtype: {'bf16' if self.use_bf16 else 'float32'}")
        print(f"  Size per file: {total_bytes / 1e9:.2f} GB")
        print(f"  Total: {total_bytes * 2 / 1e9:.2f} GB")

        for path in [self.file_a, self.file_b]:
            if not os.path.exists(path):
                mmap = np.memmap(path, dtype=self.np_dtype, mode='w+', shape=shape)
                mmap[:] = 0
                mmap.flush()
                del mmap
                print(f"  Created: {path}")
            else:
                print(f"  Exists: {path}")

        return self.file_a, self.file_b

    def open(self, mode: str = 'r+'):
        """
        mmapファイルを開く。

        Args:
            mode: 'r' (読み取り), 'r+' (読み書き), 'w+' (新規作成)
        """
        shape = (self.num_tokens, self.context_dim)
        self._mmap_a = np.memmap(self.file_a, dtype=self.np_dtype, mode=mode, shape=shape)
        self._mmap_b = np.memmap(self.file_b, dtype=self.np_dtype, mode=mode, shape=shape)

    def close(self):
        """mmapファイルを閉じる。"""
        if self._mmap_a is not None:
            self._mmap_a.flush()
            del self._mmap_a
            self._mmap_a = None
        if self._mmap_b is not None:
            self._mmap_b.flush()
            del self._mmap_b
            self._mmap_b = None

    def swap(self):
        """現在と前回のバッファを交換（ポインタ交換のみ、コピーなし）。"""
        self._current_is_a = not self._current_is_a

    def get_current_mmap(self) -> np.memmap:
        """現在のイテレーションの出力用mmapを取得。"""
        return self._mmap_a if self._current_is_a else self._mmap_b

    def get_previous_mmap(self) -> np.memmap:
        """前回のイテレーションの出力mmapを取得（CVFP損失計算用）。"""
        return self._mmap_b if self._current_is_a else self._mmap_a

    def get_chunk(
        self,
        start_idx: int,
        end_idx: int,
        from_previous: bool = False,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        指定範囲のコンテキストをTensorとして取得。

        Args:
            start_idx: 開始インデックス
            end_idx: 終了インデックス
            from_previous: 前回のイテレーションから取得するか
            device: 出力デバイス

        Returns:
            コンテキストテンソル [chunk_size, context_dim]
        """
        mmap = self.get_previous_mmap() if from_previous else self.get_current_mmap()

        # numpy -> torch変換
        # float16をbfloat16に変換（numpyはbf16非サポートのため）
        chunk_np = mmap[start_idx:end_idx].copy()
        chunk = torch.from_numpy(chunk_np)

        if self.use_bf16:
            chunk = chunk.to(torch.bfloat16)

        if device is not None:
            chunk = chunk.to(device)

        return chunk

    def set_chunk(
        self,
        start_idx: int,
        contexts: torch.Tensor
    ):
        """
        指定範囲にコンテキストを書き込み。

        Args:
            start_idx: 開始インデックス
            contexts: 書き込むコンテキスト [chunk_size, context_dim]
        """
        mmap = self.get_current_mmap()

        # torch -> numpy変換
        # bfloat16をfloat16に変換して保存
        contexts_np = contexts.detach().cpu()
        if self.use_bf16:
            contexts_np = contexts_np.to(torch.float16)
        contexts_np = contexts_np.numpy()

        mmap[start_idx:start_idx + len(contexts)] = contexts_np

    def flush(self):
        """変更をディスクに書き込み。"""
        if self._mmap_a is not None:
            self._mmap_a.flush()
        if self._mmap_b is not None:
            self._mmap_b.flush()


class EmbeddingCache:
    """
    トークン埋め込みのディスクキャッシュ。

    GPT-2埋め込みを事前計算してmmapファイルに保存。
    訓練時はmmapから直接読み込み。
    """

    def __init__(
        self,
        storage_dir: str,
        num_tokens: int,
        embed_dim: int,
        use_bf16: bool = True
    ):
        self.storage_dir = storage_dir
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim
        self.use_bf16 = use_bf16

        self.np_dtype = np.float16 if use_bf16 else np.float32
        self.torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

        self.embeddings_dir = os.path.join(storage_dir, "embeddings")
        os.makedirs(self.embeddings_dir, exist_ok=True)

        self.file_path = os.path.join(self.embeddings_dir, "token_embeddings.bin")
        self._mmap: Optional[np.memmap] = None

    def exists(self) -> bool:
        """キャッシュファイルが存在するか。"""
        return os.path.exists(self.file_path)

    def create(self) -> str:
        """キャッシュファイルを作成。"""
        shape = (self.num_tokens, self.embed_dim)
        bytes_per_element = 2 if self.use_bf16 else 4
        total_bytes = self.num_tokens * self.embed_dim * bytes_per_element

        print(f"Creating embedding cache:")
        print(f"  Shape: {shape}")
        print(f"  Size: {total_bytes / 1e9:.2f} GB")

        mmap = np.memmap(self.file_path, dtype=self.np_dtype, mode='w+', shape=shape)
        mmap.flush()
        del mmap

        print(f"  Created: {self.file_path}")
        return self.file_path

    def open(self, mode: str = 'r'):
        """mmapファイルを開く。"""
        shape = (self.num_tokens, self.embed_dim)
        self._mmap = np.memmap(self.file_path, dtype=self.np_dtype, mode=mode, shape=shape)

    def close(self):
        """mmapファイルを閉じる。"""
        if self._mmap is not None:
            if hasattr(self._mmap, 'flush'):
                self._mmap.flush()
            del self._mmap
            self._mmap = None

    def get_chunk(
        self,
        start_idx: int,
        end_idx: int,
        device: torch.device = None
    ) -> torch.Tensor:
        """
        指定範囲の埋め込みをTensorとして取得。
        """
        chunk_np = self._mmap[start_idx:end_idx].copy()
        chunk = torch.from_numpy(chunk_np)

        if self.use_bf16:
            chunk = chunk.to(torch.bfloat16)

        if device is not None:
            chunk = chunk.to(device)

        return chunk

    def set_chunk(
        self,
        start_idx: int,
        embeddings: torch.Tensor
    ):
        """
        指定範囲に埋め込みを書き込み。
        """
        embeddings_np = embeddings.detach().cpu()
        if self.use_bf16:
            embeddings_np = embeddings_np.to(torch.float16)
        embeddings_np = embeddings_np.numpy()

        self._mmap[start_idx:start_idx + len(embeddings)] = embeddings_np

    def flush(self):
        """変更をディスクに書き込み。"""
        if self._mmap is not None:
            self._mmap.flush()


class TokenIDCache:
    """
    トークンIDのディスクキャッシュ。
    """

    def __init__(self, storage_dir: str, num_tokens: int):
        self.storage_dir = storage_dir
        self.num_tokens = num_tokens

        self.tokens_dir = os.path.join(storage_dir, "tokens")
        os.makedirs(self.tokens_dir, exist_ok=True)

        self.file_path = os.path.join(self.tokens_dir, "token_ids.bin")
        self._mmap: Optional[np.memmap] = None

    def exists(self) -> bool:
        return os.path.exists(self.file_path)

    def create(self) -> str:
        shape = (self.num_tokens,)
        total_bytes = self.num_tokens * 8  # int64

        print(f"Creating token ID cache:")
        print(f"  Shape: {shape}")
        print(f"  Size: {total_bytes / 1e9:.2f} GB")

        mmap = np.memmap(self.file_path, dtype=np.int64, mode='w+', shape=shape)
        mmap.flush()
        del mmap

        print(f"  Created: {self.file_path}")
        return self.file_path

    def open(self, mode: str = 'r'):
        shape = (self.num_tokens,)
        self._mmap = np.memmap(self.file_path, dtype=np.int64, mode=mode, shape=shape)

    def close(self):
        if self._mmap is not None:
            if hasattr(self._mmap, 'flush'):
                self._mmap.flush()
            del self._mmap
            self._mmap = None

    def get_chunk(self, start_idx: int, end_idx: int) -> torch.Tensor:
        chunk_np = self._mmap[start_idx:end_idx].copy()
        return torch.from_numpy(chunk_np)

    def set_chunk(self, start_idx: int, token_ids: torch.Tensor):
        self._mmap[start_idx:start_idx + len(token_ids)] = token_ids.numpy()

    def flush(self):
        if self._mmap is not None:
            self._mmap.flush()
