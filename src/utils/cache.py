"""
Context Cache Collection Utilities

Phase 2 Prep用のコンテキストキャッシュ収集関数
GPUメモリを効率的に使用するため、チャンク処理を行う

重要: 全データを一度にGPUにロードすると7GB+のメモリを消費する。
必ずチャンク単位で処理すること。
"""

import os
import time
from typing import Protocol, Optional, Tuple, List, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.utils.io import print_flush
from src.utils.device import clear_gpu_cache

# デフォルトチャンクサイズ
DEFAULT_CONTEXT_CHUNK_SIZE = 10000  # context cache収集用
DEFAULT_EMBED_CHUNK_SIZE = 50000    # token embedding収集用


class EmbeddingModel(Protocol):
    """Token embedding収集に必要なモデルインターフェース"""
    embed_dim: int
    token_embedding: nn.Embedding
    embed_norm: nn.LayerNorm


class ContextModel(Protocol):
    """コンテキストキャッシュ収集に必要なモデルインターフェース"""
    context_dim: int
    token_embedding: nn.Embedding
    embed_norm: nn.LayerNorm

    def forward_context(self, context: torch.Tensor, token_embeds: torch.Tensor) -> torch.Tensor:
        ...


def collect_context_cache_sequential(
    model: ContextModel,
    token_ids: torch.Tensor,
    device: torch.device,
    chunk_size: int = 10000,
    progress_interval: int = 100000,
) -> torch.Tensor:
    """
    Phase 2 Prep: コンテキストキャッシュを順次処理で収集

    正確なRNN動作を再現するため、トークンを1つずつ処理する。
    GPUメモリ効率のため、token_embedsをチャンク単位で処理。

    Args:
        model: ContextModel プロトコルを満たすモデル
        token_ids: トークンID [num_tokens]
        device: デバイス
        chunk_size: 一度にGPUに転送するトークン数（デフォルト10000）
        progress_interval: 進捗表示間隔

    Returns:
        context_cache: [num_tokens-1, context_dim] on CPU
    """
    model.eval()  # type: ignore
    num_tokens = len(token_ids)
    context_dim = model.context_dim

    # 結果格納（CPU）
    context_cache = torch.zeros(num_tokens - 1, context_dim, device='cpu')

    # 初期context
    prev_context = torch.zeros(1, context_dim, device=device)

    # チャンク単位で処理
    with torch.no_grad():
        for chunk_start in range(0, num_tokens - 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_tokens - 1)

            # このチャンクのtoken_idsをGPUに転送
            chunk_token_ids = token_ids[chunk_start:chunk_end + 1].to(device)

            # embedding計算（チャンク分のみ）
            chunk_embeds = model.token_embedding(chunk_token_ids)
            chunk_embeds = model.embed_norm(chunk_embeds)

            # チャンク内を順次処理
            for i in range(chunk_end - chunk_start):
                token_embed = chunk_embeds[i:i+1]
                new_context = model.forward_context(prev_context, token_embed)

                # キャッシュに保存（CPUへ）
                context_cache[chunk_start + i] = new_context.cpu()

                # 計算グラフを切断して次へ
                prev_context = new_context.detach()

            # チャンク完了後、GPUメモリを解放
            del chunk_token_ids, chunk_embeds
            clear_gpu_cache(device)

            # 進捗表示
            processed = chunk_end
            if processed % progress_interval < chunk_size or processed == num_tokens - 1:
                print_flush(f"      {processed:,}/{num_tokens-1:,} tokens processed...")

    return context_cache


def collect_context_cache_sequential_multiblock(
    model: nn.Module,
    token_ids: torch.Tensor,
    device: torch.device,
    num_blocks: int,
    chunk_size: int = 10000,
    progress_interval: int = 100000,
) -> list[torch.Tensor]:
    """
    Phase 2 Prep: 複数ContextBlock用のコンテキストキャッシュ収集

    N-block構成での順次処理。各ブロックのコンテキストを個別に収集。

    Args:
        model: forward_context(block_idx, context, token_embeds) を持つモデル
        token_ids: トークンID [num_tokens]
        device: デバイス
        num_blocks: ContextBlockの数
        chunk_size: 一度にGPUに転送するトークン数
        progress_interval: 進捗表示間隔

    Returns:
        list of context_cache: 各ブロックの [num_tokens-1, context_dim] on CPU
    """
    model.eval()
    num_tokens = len(token_ids)
    context_dim = model.context_dim  # type: ignore

    # 結果格納（各ブロック用、CPU）
    context_caches = [
        torch.zeros(num_tokens - 1, context_dim, device='cpu')
        for _ in range(num_blocks)
    ]

    # 初期context（各ブロック）
    prev_contexts = [
        torch.zeros(1, context_dim, device=device)
        for _ in range(num_blocks)
    ]

    # チャンク単位で処理
    with torch.no_grad():
        for chunk_start in range(0, num_tokens - 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_tokens - 1)

            # このチャンクのtoken_idsをGPUに転送
            chunk_token_ids = token_ids[chunk_start:chunk_end + 1].to(device)

            # embedding計算（チャンク分のみ）
            chunk_embeds = model.token_embedding(chunk_token_ids)  # type: ignore
            chunk_embeds = model.embed_norm(chunk_embeds)  # type: ignore

            # チャンク内を順次処理
            for i in range(chunk_end - chunk_start):
                token_embed = chunk_embeds[i:i+1]

                # 各ブロックを順番に処理
                for block_idx in range(num_blocks):
                    new_context = model.forward_context(  # type: ignore
                        block_idx, prev_contexts[block_idx], token_embed
                    )

                    # キャッシュに保存（CPUへ）
                    context_caches[block_idx][chunk_start + i] = new_context.cpu()

                    # 計算グラフを切断
                    prev_contexts[block_idx] = new_context.detach()

            # チャンク完了後、GPUメモリを解放
            del chunk_token_ids, chunk_embeds
            clear_gpu_cache(device)

            # 進捗表示
            processed = chunk_end
            if processed % progress_interval < chunk_size or processed == num_tokens - 1:
                print_flush(f"      {processed:,}/{num_tokens-1:,} tokens processed...")

    return context_caches


def collect_token_embeds_chunked(
    model: EmbeddingModel,
    token_ids: torch.Tensor,
    device: torch.device,
    chunk_size: int = DEFAULT_EMBED_CHUNK_SIZE,
) -> torch.Tensor:
    """
    Token embeddingsをチャンク単位で収集（GPUメモリ節約）

    全token_idsを一度にGPUにロードすると7GB+のメモリを消費する。
    チャンク単位で処理することで、最大メモリ使用量を150MB程度に抑える。

    Args:
        model: EmbeddingModel プロトコルを満たすモデル
        token_ids: トークンID [num_tokens]
        device: デバイス
        chunk_size: 一度にGPUに転送するトークン数（デフォルト50000）

    Returns:
        token_embeds: [num_tokens-1, embed_dim] on CPU
    """
    num_tokens = len(token_ids)
    embed_dim = model.embed_dim

    # 結果格納（CPU）
    token_embeds = torch.zeros(num_tokens - 1, embed_dim, device='cpu')

    with torch.no_grad():
        for chunk_start in range(0, num_tokens - 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_tokens - 1)

            # チャンク分だけGPUに転送
            chunk_ids = token_ids[chunk_start:chunk_end].to(device)
            chunk_embeds = model.token_embedding(chunk_ids)
            chunk_embeds = model.embed_norm(chunk_embeds)

            # CPUに保存
            token_embeds[chunk_start:chunk_end] = chunk_embeds.cpu()

            # GPUメモリ解放
            del chunk_ids, chunk_embeds

        clear_gpu_cache(device)

    return token_embeds


# ============================================================
# ファイルベースのキャッシュ管理（大規模データ用）
# ============================================================

DEFAULT_FILE_CHUNK_SIZE = 100000  # ファイル保存用チャンクサイズ


class MultiBlockModel(Protocol):
    """複数ContextBlockを持つモデルのインターフェース"""
    context_dims: List[int]  # 各ブロックのcontext_dim
    num_context_blocks: int
    token_embedding: nn.Embedding
    embed_norm: nn.LayerNorm

    def forward_context(
        self, block_idx: int, context: torch.Tensor, token_embeds: torch.Tensor
    ) -> torch.Tensor:
        ...


def collect_multiblock_cache_to_files(
    model: MultiBlockModel,
    token_ids: torch.Tensor,
    device: torch.device,
    cache_dir: str,
    prefix: str = "cache",
    chunk_size: int = DEFAULT_FILE_CHUNK_SIZE,
    embed_chunk_size: int = DEFAULT_CONTEXT_CHUNK_SIZE,
    progress_interval: int = 100000,
    prev_context_steps: int = 0,
) -> Tuple[int, int, List[str]]:
    """
    N-block用: チャンク単位でファイル保存しながらコンテキストキャッシュを収集

    GPUメモリ効率が良い（~17GB → ~2GB）。
    token_embedsもチャンク単位で処理し、全データをGPUにロードしない。

    Args:
        model: MultiBlockModel プロトコルを満たすモデル
        token_ids: トークンID [num_tokens]
        device: デバイス
        cache_dir: キャッシュ保存ディレクトリ
        prefix: ファイル名プレフィックス（"train" or "val"）
        chunk_size: ファイルに保存するチャンクサイズ（デフォルト100,000トークン）
        embed_chunk_size: GPUに転送するトークン数（デフォルト10,000）
        progress_interval: 進捗表示間隔
        prev_context_steps: 前のトークン時のcontextも連結する数（0で無効）
            例: prev_context_steps=1 の場合、出力は:
            [context_0[i], context_1[i], context_0[i-1], context_1[i-1]]
            combined_dim = sum(context_dims) * (1 + prev_context_steps)

    Returns:
        (num_tokens, combined_dim, chunk_files): トークン数、結合次元、チャンクファイルリスト
    """
    model.eval()  # type: ignore
    num_tokens = len(token_ids) - 1
    context_dims = model.context_dims  # List[int]
    num_blocks = model.num_context_blocks
    # prev_context_steps=0: 現在のみ、=1: 現在+1つ前、=2: 現在+1つ前+2つ前
    combined_dim = sum(context_dims) * (1 + prev_context_steps)

    os.makedirs(cache_dir, exist_ok=True)

    prev_info = f", prev_steps={prev_context_steps}" if prev_context_steps > 0 else ""
    print_flush(f"    Collecting {prefix} cache ({num_tokens:,} tokens, "
                f"{num_blocks} blocks{prev_info}, chunk={chunk_size:,})...")
    collect_start = time.time()

    chunk_files: List[str] = []
    num_file_chunks = (num_tokens + chunk_size - 1) // chunk_size

    # 初期context（各ブロック用、ブロックごとに異なる次元）
    prev_contexts = [
        torch.zeros(1, context_dims[block_idx], device=device)
        for block_idx in range(num_blocks)
    ]

    # prev_context_steps > 0 の場合、履歴を保持するリングバッファ
    # context_history[step][block_idx] = context tensor (CPU)
    # step=0 が最新、step=1 が1つ前、...
    context_history: List[List[torch.Tensor]] = []
    if prev_context_steps > 0:
        for _ in range(prev_context_steps):
            context_history.append([
                torch.zeros(1, context_dims[block_idx], device='cpu')
                for block_idx in range(num_blocks)
            ])

    with torch.no_grad():
        # ファイルチャンク単位で処理
        for file_chunk_idx in range(num_file_chunks):
            file_chunk_start = file_chunk_idx * chunk_size
            file_chunk_end = min((file_chunk_idx + 1) * chunk_size, num_tokens)
            current_file_chunk_size = file_chunk_end - file_chunk_start

            # このファイルチャンク用のキャッシュ（CPU）
            file_context_cache = torch.zeros(
                current_file_chunk_size, combined_dim, device='cpu'
            )
            file_token_embeds = torch.zeros(
                current_file_chunk_size, model.embed_dim, device='cpu'  # type: ignore
            )

            # GPUメモリ効率のため、embed_chunk_size単位で処理
            for embed_chunk_start in range(file_chunk_start, file_chunk_end, embed_chunk_size):
                embed_chunk_end = min(embed_chunk_start + embed_chunk_size, file_chunk_end)

                # このチャンクのtoken_idsをGPUに転送
                chunk_token_ids = token_ids[embed_chunk_start:embed_chunk_end + 1].to(device)

                # embedding計算（チャンク分のみ）
                chunk_embeds = model.token_embedding(chunk_token_ids)
                chunk_embeds = model.embed_norm(chunk_embeds)

                # チャンク内を順次処理
                for i in range(embed_chunk_end - embed_chunk_start):
                    global_idx = embed_chunk_start + i
                    local_idx = global_idx - file_chunk_start

                    token_embed = chunk_embeds[i:i+1]

                    # token_embedsを保存
                    file_token_embeds[local_idx] = token_embed.cpu().squeeze(0)

                    # 全ブロックを処理して現在のcontextを取得
                    current_contexts: List[torch.Tensor] = []
                    for block_idx in range(num_blocks):
                        new_context = model.forward_context(
                            block_idx, prev_contexts[block_idx], token_embed
                        )
                        current_contexts.append(new_context.cpu())
                        prev_contexts[block_idx] = new_context.detach()

                    # 結合リスト: [現在のcontext] + [履歴context]
                    all_contexts = current_contexts.copy()
                    if prev_context_steps > 0:
                        for step in range(prev_context_steps):
                            all_contexts.extend(context_history[step])

                    # 結合してキャッシュに保存
                    file_context_cache[local_idx] = torch.cat(
                        all_contexts, dim=-1
                    ).squeeze(0)

                    # 履歴を更新（シフト）
                    if prev_context_steps > 0:
                        # 古い履歴を1つずつ後ろにシフト
                        for step in range(prev_context_steps - 1, 0, -1):
                            context_history[step] = context_history[step - 1]
                        # 最新の履歴を更新
                        context_history[0] = current_contexts

                    del current_contexts, all_contexts

                # GPUメモリ解放
                del chunk_token_ids, chunk_embeds
                clear_gpu_cache(device)

                # 進捗表示
                processed = embed_chunk_end
                if processed % progress_interval < embed_chunk_size or processed == num_tokens:
                    print_flush(f"      {processed:,}/{num_tokens:,} tokens processed...")

            # ファイルチャンクをファイルに保存
            chunk_file = os.path.join(
                cache_dir, f"{prefix}_chunk_{file_chunk_idx:04d}.pt"
            )
            torch.save({
                'context_cache': file_context_cache,
                'token_embeds': file_token_embeds,
                'chunk_start': file_chunk_start,
                'chunk_end': file_chunk_end,
            }, chunk_file)
            chunk_files.append(chunk_file)

            del file_context_cache, file_token_embeds

            print_flush(f"      Chunk {file_chunk_idx+1}/{num_file_chunks} saved "
                        f"({file_chunk_end:,}/{num_tokens:,} tokens)")

    print_flush(f"    Cache collected [{time.time() - collect_start:.1f}s] "
                f"-> {len(chunk_files)} chunks")

    return num_tokens, combined_dim, chunk_files


class ChunkedCacheDataset(Dataset):
    """
    チャンクファイルからデータを読み込むDataset

    Phase 2学習で使用。メモリ効率が良い。
    大規模データでも全データをメモリにロードせずに処理可能。
    """

    def __init__(self, chunk_files: List[str]):
        """
        Args:
            chunk_files: チャンクファイルパスのリスト
        """
        self.chunk_files = chunk_files

        # 全体のサイズを計算
        self.total_size = 0
        self.chunk_info: List[Tuple[int, int, str]] = []  # (start, end, file)

        for chunk_file in chunk_files:
            data = torch.load(chunk_file, weights_only=True)
            chunk_start = data['chunk_start']
            chunk_end = data['chunk_end']
            self.chunk_info.append((chunk_start, chunk_end, chunk_file))
            self.total_size = max(self.total_size, chunk_end)
            del data

        # 現在ロード中のチャンク
        self._current_chunk_idx: Optional[int] = None
        self._current_data: Optional[Dict[str, torch.Tensor]] = None

    def __len__(self) -> int:
        return self.total_size

    def _load_chunk(self, chunk_idx: int) -> None:
        """チャンクをロード"""
        if self._current_chunk_idx != chunk_idx:
            self._current_data = torch.load(
                self.chunk_info[chunk_idx][2], weights_only=True
            )
            self._current_chunk_idx = chunk_idx

    def _find_chunk(self, idx: int) -> int:
        """インデックスが属するチャンクを見つける"""
        for chunk_idx, (start, end, _) in enumerate(self.chunk_info):
            if start <= idx < end:
                return chunk_idx
        raise IndexError(f"Index {idx} out of range")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (context_cache[idx], token_embed[idx])
        """
        chunk_idx = self._find_chunk(idx)
        self._load_chunk(chunk_idx)

        assert self._current_data is not None
        local_idx = idx - self.chunk_info[chunk_idx][0]

        return (
            self._current_data['context_cache'][local_idx],
            self._current_data['token_embeds'][local_idx],
        )

    def get_all_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        全データをメモリにロード（小〜中規模データ用）

        Returns:
            (context_cache, token_embeds): 全データを結合したテンソル
        """
        all_context = []
        all_embeds = []

        for chunk_file in self.chunk_files:
            data = torch.load(chunk_file, weights_only=True)
            all_context.append(data['context_cache'])
            all_embeds.append(data['token_embeds'])
            del data

        return torch.cat(all_context, dim=0), torch.cat(all_embeds, dim=0)
