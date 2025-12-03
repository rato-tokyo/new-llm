#!/usr/bin/env python3
"""
Context-KV Attention 実験スクリプト

チャンク単位のcontextをKVキャッシュとして使用するモデルの実験。

Usage:
    python3 scripts/experiment_context_kv.py -s 200 --chunk-size 100
"""

import argparse
import gc
import os
import sys
import time
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from config.experiment import DataConfig
from src.config.wrappers import Phase1ConfigWrapper
from src.providers.data.memory import MemoryDataProvider
from src.evaluation.metrics import compute_effective_rank
from src.models.context_kv import ContextKVAttentionLLM, ContextKVWrapper
from src.trainers.phase1.memory import MemoryPhase1Trainer
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.device import clear_gpu_cache


def collect_val_cache_parallel(
    model: ContextKVWrapper,
    token_ids: torch.Tensor,
    device: torch.device,
    batch_size: int = 50000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Val dataのキャッシュを並列処理で収集

    Phase 1のtrainと同じshifted_prev_context方式を使用。

    Returns:
        context_cache: [num_tokens-1, context_dim]
        token_embeds: [num_tokens-1, embed_dim]
    """
    model.eval()
    num_tokens = len(token_ids) - 1  # 最後のトークンを除く
    context_dim = model.context_dim

    # Token embeddings計算
    with torch.no_grad():
        token_embeds_gpu = model.token_embedding(token_ids[:-1].unsqueeze(0).to(device))
        token_embeds_gpu = model.embed_norm(token_embeds_gpu).squeeze(0)
        token_embeds = token_embeds_gpu.cpu()
        del token_embeds_gpu
        clear_gpu_cache(device)

    # 結果テンソルを事前確保
    contexts = torch.zeros(num_tokens, context_dim)

    # 初期context（ゼロベクトル）
    init_ctx = torch.zeros(1, context_dim)

    # 最初のパス: ランダム初期化 (小さいスケールで初期化)
    previous_contexts = torch.randn(num_tokens, context_dim) * 0.01

    # 数回iterationして収束させる（簡易版）
    for iteration in range(3):
        # shifted_prev: [zero, ctx[0], ctx[1], ..., ctx[n-2]]
        shifted_prev = torch.cat([init_ctx, previous_contexts[:-1]], dim=0)

        for start_idx in range(0, num_tokens, batch_size):
            end_idx = min(start_idx + batch_size, num_tokens)
            batch_contexts = shifted_prev[start_idx:end_idx].to(device)
            batch_embeds = token_embeds[start_idx:end_idx].to(device)

            with torch.no_grad():
                batch_output = model.context_block(batch_contexts, batch_embeds)
            contexts[start_idx:end_idx] = batch_output.cpu()

            del batch_contexts, batch_embeds, batch_output
            clear_gpu_cache(device)

        # 次のiteration用に更新 (最後のiterationではclone不要)
        if iteration < 2:
            previous_contexts = contexts.clone()
        del shifted_prev

    # 不要な中間テンソルを解放
    del previous_contexts, init_ctx

    return contexts, token_embeds


def get_chunk_boundaries(context_cache: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    チャンク境界のcontextを抽出

    Args:
        context_cache: [num_tokens, context_dim]
        chunk_size: チャンクサイズ

    Returns:
        chunk_boundaries: [num_chunks, context_dim]
        各チャンク境界（position chunk_size-1, 2*chunk_size-1, ...）のcontext
    """
    # 境界インデックス: chunk_size-1, 2*chunk_size-1, ...
    boundary_indices = list(range(chunk_size - 1, len(context_cache), chunk_size))
    if len(boundary_indices) == 0:
        return torch.zeros(0, context_cache.shape[1])
    return context_cache[boundary_indices]


def build_batch_context_chunks(
    batch_indices: torch.Tensor,
    context_cache: torch.Tensor,
    chunk_boundaries: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """
    バッチのcontext chunksを動的に構築（ベクトル化版）

    Args:
        batch_indices: バッチ内の位置インデックス [batch_size]
        context_cache: 全context [num_tokens, context_dim]
        chunk_boundaries: チャンク境界のcontext [num_boundaries, context_dim]
        chunk_size: チャンクサイズ

    Returns:
        batch_chunks: [batch_size, batch_max_chunks, context_dim]
        batch_max_chunksはバッチ内の最大位置で決まる
    """
    batch_size = len(batch_indices)
    context_dim = context_cache.shape[1]
    num_boundaries = len(chunk_boundaries)

    # バッチ内の最大位置から必要なチャンク数を計算
    max_pos = int(batch_indices.max().item())
    batch_max_chunks = (max_pos + 1) // chunk_size + 1

    # 出力テンソル（バッチに必要な分だけ確保）
    batch_chunks = torch.zeros(batch_size, batch_max_chunks, context_dim)

    # 各サンプルのチャンク数を計算（ベクトル化）
    num_past_chunks = (batch_indices + 1) // chunk_size  # [batch_size]

    # チャンク境界をコピー（部分ベクトル化）
    if num_boundaries > 0:
        for c in range(min(batch_max_chunks, num_boundaries)):
            # このチャンクが必要なサンプルのマスク
            mask = num_past_chunks > c
            batch_chunks[mask, c] = chunk_boundaries[c]

    # 現在のcontextを最後のチャンク位置に設定（ベクトル化）
    batch_chunks[torch.arange(batch_size), num_past_chunks] = context_cache[batch_indices]

    return batch_chunks


def train_phase2(
    model: ContextKVAttentionLLM,
    train_context_cache: torch.Tensor,
    train_chunk_boundaries: torch.Tensor,
    train_token_embeds: torch.Tensor,
    train_targets: torch.Tensor,
    val_context_cache: torch.Tensor,
    val_chunk_boundaries: torch.Tensor,
    val_token_embeds: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
    chunk_size: int,
    num_epochs: int = 40,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 3,
) -> tuple[float, float, int]:
    """
    Phase 2: Context-KV Attention + FFN の学習（メモリ効率版）

    Returns:
        best_ppl, best_acc, best_epoch
    """
    model.freeze_all_context_blocks()

    # Embedding も freeze
    for param in model.token_embedding.parameters():
        param.requires_grad = False
    for param in model.embed_norm.parameters():
        param.requires_grad = False
    print_flush("✓ Embedding frozen")

    # 学習対象パラメータ
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())
    print_flush(f"✓ Training {trainable_params:,}/{total_params:,} parameters")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()

    num_train = len(train_targets)
    num_val = len(val_targets)

    best_val_ppl = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    no_improve = 0

    print_flush(f"\n[Phase 2] {num_train:,} train / {num_val:,} val tokens, {num_epochs} epochs")

    num_batches = (num_train + batch_size - 1) // batch_size

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0

        indices = torch.randperm(num_train)

        for batch_num, start in enumerate(range(0, num_train, batch_size)):
            end = min(start + batch_size, num_train)
            batch_idx = indices[start:end]

            # 進捗表示（最初のエポックのみ詳細表示、中間成績付き）
            if epoch == 1 and (batch_num % 50 == 0 or batch_num == num_batches - 1):
                processed = batch_num * batch_size
                if processed > 0:
                    interim_ppl = torch.exp(torch.tensor(train_loss / processed)).item()
                    interim_acc = train_correct / processed * 100
                    print_flush(f"    Epoch 1: batch {batch_num+1}/{num_batches} "
                                f"(ppl={interim_ppl:.1f}, acc={interim_acc:.1f}%)")
                else:
                    print_flush(f"    Epoch 1: batch {batch_num+1}/{num_batches}...")

            # 動的にcontext chunksを構築
            batch_context_chunks = build_batch_context_chunks(
                batch_idx, train_context_cache, train_chunk_boundaries,
                chunk_size
            ).to(device)
            batch_token_embeds = train_token_embeds[batch_idx].to(device)
            batch_targets = train_targets[batch_idx].to(device)

            optimizer.zero_grad()

            # Forward
            hidden = model.forward_attention(batch_token_embeds, batch_context_chunks)
            logits = model.forward_output(hidden)

            loss = criterion(logits, batch_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_idx)
            train_correct += int((logits.argmax(dim=-1) == batch_targets).sum().item())

            del batch_context_chunks, batch_token_embeds, batch_targets
            clear_gpu_cache(device)

        train_loss /= num_train
        train_ppl = torch.exp(torch.tensor(train_loss)).item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for start in range(0, num_val, batch_size):
                end = min(start + batch_size, num_val)
                batch_idx = torch.arange(start, end)

                batch_context_chunks = build_batch_context_chunks(
                    batch_idx, val_context_cache, val_chunk_boundaries,
                    chunk_size
                ).to(device)
                batch_token_embeds = val_token_embeds[start:end].to(device)
                batch_targets = val_targets[start:end].to(device)

                hidden = model.forward_attention(batch_token_embeds, batch_context_chunks)
                logits = model.forward_output(hidden)

                loss = criterion(logits, batch_targets)

                val_loss += loss.item() * len(batch_targets)
                val_correct += int((logits.argmax(dim=-1) == batch_targets).sum().item())

                del batch_context_chunks, batch_token_embeds, batch_targets
                clear_gpu_cache(device)

        val_loss /= num_val
        val_ppl = torch.exp(torch.tensor(val_loss)).item()
        val_acc = val_correct / num_val

        epoch_time = time.time() - epoch_start

        # Early stopping check
        improved = val_ppl < best_val_ppl
        if improved:
            best_val_ppl = val_ppl
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            marker = " *"
        else:
            no_improve += 1
            marker = ""

        print_flush(
            f"    Epoch {epoch}: train_ppl={train_ppl:.1f} val_ppl={val_ppl:.1f} "
            f"acc={val_acc*100:.1f}% [{epoch_time:.1f}s]{marker}"
        )

        if no_improve >= patience:
            print_flush(f"    → Early stop at epoch {epoch}")
            break

    print_flush(f"    Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}, acc={best_val_acc*100:.1f}%")

    return best_val_ppl, best_val_acc, best_epoch


def run_context_kv_experiment(
    num_samples: int = 200,
    context_dim: int = 256,
    chunk_size: int = 100,
    num_heads: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Context-KV Attention 実験を実行（1ブロック版）"""

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print_flush(f"Device: {device} ({gpu_name}, {gpu_mem:.1f}GB)")
    else:
        print_flush(f"Device: {device}")

    # 設定
    base_config = Config()

    # データロード
    print_flush("Loading data...")
    data_config = DataConfig.from_base(base_config, num_samples=num_samples)
    data_provider = MemoryDataProvider(data_config)
    train_token_ids, val_token_ids = data_provider.load_data()
    num_train_tokens = len(train_token_ids)
    print_flush(f"Data: {num_train_tokens:,} train, {len(val_token_ids):,} val tokens")

    # モデル作成（1ブロック）
    context_dims = [context_dim]
    print_flush(f"\nCreating ContextKVAttentionLLM (cd={context_dim}, heads={num_heads}, chunk={chunk_size})...")

    model = ContextKVAttentionLLM(
        vocab_size=base_config.vocab_size,
        embed_dim=base_config.embed_dim,
        context_dims=context_dims,
        num_heads=num_heads,
        chunk_size=chunk_size,
    )
    model.to(device)

    params = model.num_params()
    print_flush(f"Parameters: {params['total']:,} total")

    # ========== Phase 1: ContextBlock学習 ==========
    print_flush(f"\n[Phase 1] Training ContextBlock (cd={context_dim})...")
    wrapper = ContextKVWrapper(model, block_idx=0)
    config_wrapper = Phase1ConfigWrapper(base_config, context_dim)
    trainer = MemoryPhase1Trainer(wrapper, config_wrapper, device)

    phase1_start = time.time()
    train_result = trainer.train(
        train_token_ids,
        label="Context",
        return_all_layers=True,
    )
    phase1_time = time.time() - phase1_start

    # Phase 1完了後、trainerを解放してメモリ節約
    del trainer
    gc.collect()
    clear_gpu_cache(device)

    model.freeze_context_block(0)
    print_flush("✓ ContextBlock frozen")

    # ========== Phase 2 Prep: キャッシュとチャンク境界を準備 ==========
    print_flush("\n[Phase 2 Prep] Preparing cache and chunk boundaries...")
    cache_start = time.time()

    # Phase 1のキャッシュを使用（Trainデータ）
    assert train_result.cache is not None
    assert train_result.token_embeds is not None

    # メモリ効率化: 必要なデータのみ取り出し、train_resultを早期解放
    # Note: train_result.contextsは使用しない（cacheと重複するため）
    train_context_cache = train_result.cache
    train_token_embeds = train_result.token_embeds

    # train_result全体を解放（contexts含む）
    del train_result
    gc.collect()
    clear_gpu_cache(device)

    train_chunk_boundaries = get_chunk_boundaries(train_context_cache, chunk_size)
    train_targets = train_token_ids[1:].clone()

    # Val dataはPhase 1と同じ並列方式でキャッシュ収集
    print_flush("  Collecting val cache (parallel)...")
    val_context_cache, val_token_embeds = collect_val_cache_parallel(
        wrapper, val_token_ids, device, config_wrapper.phase1_batch_size
    )

    # wrapperとconfig_wrapperを解放（Phase 2では使わない）
    del wrapper, config_wrapper
    gc.collect()

    val_chunk_boundaries = get_chunk_boundaries(val_context_cache, chunk_size)
    val_targets = val_token_ids[1:].clone()

    # 元のtoken_idsはtargetsに変換済みなので解放
    del train_token_ids, val_token_ids
    gc.collect()

    cache_time = time.time() - cache_start
    print_flush(f"Cache preparation: {cache_time:.1f}s")
    print_flush(f"  Train: {train_context_cache.shape}, {len(train_chunk_boundaries)} chunks")
    print_flush(f"  Val: {val_context_cache.shape}, {len(val_chunk_boundaries)} chunks")

    # Effective Rank（最終位置のcontextで計算）
    val_er = compute_effective_rank(val_context_cache.cpu())
    val_er_pct = val_er / context_dim * 100
    print_flush(f"Effective Rank: Val={val_er_pct:.1f}%")

    clear_gpu_cache(device)

    # ========== Phase 2: Attention + FFN学習 ==========
    print_flush("\n[Phase 2] Training Context-KV Attention...")

    phase2_start = time.time()
    best_ppl, best_acc, best_epoch = train_phase2(
        model=model,
        train_context_cache=train_context_cache,
        train_chunk_boundaries=train_chunk_boundaries,
        train_token_embeds=train_token_embeds,
        train_targets=train_targets,
        val_context_cache=val_context_cache,
        val_chunk_boundaries=val_chunk_boundaries,
        val_token_embeds=val_token_embeds,
        val_targets=val_targets,
        device=device,
        chunk_size=chunk_size,
    )
    phase2_time = time.time() - phase2_start

    total_time = phase1_time + cache_time + phase2_time

    # ========== サマリー ==========
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY - Context-KV Attention Experiment")
    print_flush("=" * 70)
    print_flush(f"  Context dim: {context_dim}")
    print_flush(f"  Num heads: {num_heads}")
    print_flush(f"  Chunk size: {chunk_size}")
    print_flush(f"Parameters: {params['total']:,}")
    print_flush(f"Phase 1: {phase1_time:.1f}s")
    print_flush(f"Cache collection: {cache_time:.1f}s")
    print_flush(f"Phase 2: {phase2_time:.1f}s, epoch {best_epoch}")
    print_flush(f"Effective Rank: {val_er_pct:.1f}%")
    print_flush(f"Val PPL: {best_ppl:.1f}")
    print_flush(f"Val Acc: {best_acc*100:.1f}%")
    print_flush(f"Total time: {total_time:.1f}s")
    print_flush("=" * 70)

    return {
        'num_samples': num_samples,
        'context_dim': context_dim,
        'chunk_size': chunk_size,
        'num_heads': num_heads,
        'val_ppl': best_ppl,
        'val_acc': best_acc,
        'val_er_pct': val_er_pct,
        'phase1_time': phase1_time,
        'cache_time': cache_time,
        'phase2_time': phase2_time,
        'total_time': total_time,
        'best_epoch': best_epoch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Context-KV Attention Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # 200サンプル、チャンクサイズ100
  python3 scripts/experiment_context_kv.py -s 200 --chunk-size 100

  # 800サンプル、チャンクサイズ50
  python3 scripts/experiment_context_kv.py -s 800 --chunk-size 50
'''
    )
    parser.add_argument('-s', '--samples', type=int, default=200,
                        help='Number of samples (default: 200)')
    parser.add_argument('-c', '--context-dim', type=int, default=256,
                        help='Context dimension (default: 256)')
    parser.add_argument('--chunk-size', type=int, default=100,
                        help='Chunk size for context KV (default: 100)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    print_flush("=" * 70)
    print_flush("CONTEXT-KV ATTENTION EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {args.samples}")
    print_flush(f"Context dim: {args.context_dim}")
    print_flush(f"Chunk size: {args.chunk_size}")
    print_flush(f"Num heads: {args.num_heads}")
    print_flush("=" * 70)

    run_context_kv_experiment(
        num_samples=args.samples,
        context_dim=args.context_dim,
        chunk_size=args.chunk_size,
        num_heads=args.num_heads,
        seed=args.seed,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
