"""
Context Cache Collection Utilities

Phase 2 Prep用のコンテキストキャッシュ収集関数
GPUメモリを効率的に使用するため、チャンク処理を行う

重要: 全データを一度にGPUにロードすると7GB+のメモリを消費する。
必ずチャンク単位で処理すること。
"""

from typing import Protocol
import torch
import torch.nn as nn

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
