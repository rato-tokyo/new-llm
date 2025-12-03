#!/usr/bin/env python3
"""
Context-KV Attention 実験スクリプト

チャンク単位のcontextをKVキャッシュとして使用するモデルの実験。

Usage:
    python3 scripts/experiment_context_kv.py -s 200 --chunk-size 100
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Tuple

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
from src.trainers.phase1.base import Phase1Result
from src.trainers.phase1.memory import MemoryPhase1Trainer
from src.utils.io import print_flush
from src.utils.seed import set_seed
from src.utils.device import clear_gpu_cache


def collect_context_chunks(
    model: ContextKVAttentionLLM,
    token_ids: torch.Tensor,
    device: torch.device,
    chunk_size: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    チャンク単位でcontextを収集

    Args:
        model: ContextKVAttentionLLM
        token_ids: [num_tokens]
        device: 計算デバイス
        chunk_size: チャンクサイズ

    Returns:
        context_chunks: [num_tokens-1, max_chunks, combined_dim]
        token_embeds: [num_tokens-1, embed_dim]
        chunk_boundaries: チャンク境界位置のリスト
    """
    num_tokens = len(token_ids)
    num_blocks = model.num_context_blocks
    combined_dim = model.combined_context_dim

    # 出力テンソル
    # 最大チャンク数を計算
    max_chunks = (num_tokens - 1) // chunk_size + 1

    # 出力テンソル（CPU上に事前確保）
    context_chunks = torch.zeros(num_tokens - 1, max_chunks, combined_dim)
    token_embeds_out = torch.zeros(num_tokens - 1, model.embed_dim)

    # 順次処理でcontext収集
    contexts = [
        torch.zeros(model.context_dims[i], device=device)
        for i in range(num_blocks)
    ]

    # チャンク境界のcontext（CPU上）
    chunk_contexts_cpu: List[torch.Tensor] = []

    total_tokens = num_tokens - 1
    log_interval = max(1000, total_tokens // 10)  # 10%ごと または 1000ごと
    print_flush(f"    Collecting context chunks ({total_tokens:,} tokens, chunk_size={chunk_size})...")

    start_time = time.time()

    with torch.no_grad():
        for i in range(total_tokens):
            # Token embedding
            token_id = token_ids[i:i+1].to(device)
            token_embed = model.token_embedding(token_id)
            token_embed = model.embed_norm(token_embed).squeeze(0)

            # 全ContextBlockを更新
            new_contexts = []
            for block_idx in range(num_blocks):
                ctx = contexts[block_idx].unsqueeze(0)
                te = token_embed.unsqueeze(0)
                new_ctx = model.forward_context(block_idx, ctx, te).squeeze(0)
                new_contexts.append(new_ctx.detach())
            contexts = new_contexts

            # 現在のcontextを連結
            current_combined = torch.cat(contexts, dim=-1)

            # チャンク境界の場合、chunk_contextsに追加
            if (i + 1) % chunk_size == 0:
                chunk_contexts_cpu.append(current_combined.cpu())

            # この位置で利用可能なチャンク数
            num_available_chunks = len(chunk_contexts_cpu) + 1  # +1 for current

            # context_chunksに保存（CPU上）
            # 過去のチャンクはすでにCPUにあるので再変換不要
            for j, ctx in enumerate(chunk_contexts_cpu):
                context_chunks[i, j] = ctx
            context_chunks[i, num_available_chunks - 1] = current_combined.cpu()

            # token_embedsに保存（CPU上）
            token_embeds_out[i] = token_embed.cpu()

            if (i + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                speed = (i + 1) / elapsed
                eta = (total_tokens - i - 1) / speed if speed > 0 else 0
                print_flush(f"      {i+1:,}/{total_tokens:,} ({(i+1)/total_tokens*100:.0f}%) "
                           f"[{elapsed:.1f}s, {speed:.0f} tok/s, ETA {eta:.0f}s]")
                # GPUメモリ解放
                clear_gpu_cache(device)

    return context_chunks, token_embeds_out, chunk_contexts_cpu


def train_phase2(
    model: ContextKVAttentionLLM,
    train_context_chunks: torch.Tensor,
    train_token_embeds: torch.Tensor,
    train_targets: torch.Tensor,
    val_context_chunks: torch.Tensor,
    val_token_embeds: torch.Tensor,
    val_targets: torch.Tensor,
    device: torch.device,
    num_epochs: int = 40,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 3,
) -> Tuple[float, float, int, List[Dict[str, float]]]:
    """
    Phase 2: Context-KV Attention + FFN の学習

    Returns:
        best_ppl, best_acc, best_epoch, history
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
    history: List[Dict[str, float]] = []

    print_flush(f"\n[Phase 2] {num_train:,} train / {num_val:,} val tokens, {num_epochs} epochs")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0

        indices = torch.randperm(num_train)

        for start in range(0, num_train, batch_size):
            end = min(start + batch_size, num_train)
            batch_idx = indices[start:end]

            batch_context_chunks = train_context_chunks[batch_idx].to(device)
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

        train_loss /= num_train
        train_ppl = torch.exp(torch.tensor(train_loss)).item()
        _ = train_correct / num_train  # train_acc (not used in log)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for start in range(0, num_val, batch_size):
                end = min(start + batch_size, num_val)

                batch_context_chunks = val_context_chunks[start:end].to(device)
                batch_token_embeds = val_token_embeds[start:end].to(device)
                batch_targets = val_targets[start:end].to(device)

                hidden = model.forward_attention(batch_token_embeds, batch_context_chunks)
                logits = model.forward_output(hidden)

                loss = criterion(logits, batch_targets)

                val_loss += loss.item() * len(batch_targets)
                val_correct += int((logits.argmax(dim=-1) == batch_targets).sum().item())

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

        history.append({
            'epoch': epoch,
            'train_ppl': train_ppl,
            'val_ppl': val_ppl,
            'val_acc': val_acc,
        })

        if no_improve >= patience:
            print_flush(f"    → Early stop at epoch {epoch}")
            break

    print_flush(f"    Best: epoch {best_epoch}, ppl={best_val_ppl:.1f}, acc={best_val_acc*100:.1f}%")

    return best_val_ppl, best_val_acc, best_epoch, history


def run_context_kv_experiment(
    num_samples: int = 200,
    context_dims: List[int] = [256, 256],
    chunk_size: int = 100,
    num_heads: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    """Context-KV Attention 実験を実行"""

    set_seed(seed)
    num_blocks = len(context_dims)
    combined_dim = sum(context_dims)

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

    # データ分割（Phase 1用）
    train_data_splits: List[torch.Tensor] = []
    split_size = num_train_tokens // num_blocks
    for i in range(num_blocks):
        start_idx = i * split_size
        if i == num_blocks - 1:
            end_idx = num_train_tokens
        else:
            end_idx = (i + 1) * split_size + 1
        train_data_splits.append(train_token_ids[start_idx:end_idx])
    print_flush(f"Split: {', '.join([f'Block{i}={len(s)-1:,}' for i, s in enumerate(train_data_splits)])} tokens")

    # モデル作成
    dims_str = '+'.join(map(str, context_dims))
    print_flush(f"\nCreating ContextKVAttentionLLM (cd={dims_str}={combined_dim}, heads={num_heads}, chunk={chunk_size})...")

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
    print_flush(f"  ContextBlocks ({num_blocks}): {params['total_context_blocks']:,}")
    print_flush(f"  Context-KV Attention: {params['context_kv_attention']:,}")
    print_flush(f"  Output FFN: {params['output_ffn']:,}")

    # ========== Phase 1: 各ContextBlock学習 ==========
    phase1_times: List[float] = []
    phase1_stats: List[Phase1Result] = []

    for block_idx in range(num_blocks):
        print_flush(f"\n[Phase 1-{block_idx}] Training ContextBlock {block_idx} (cd={context_dims[block_idx]})...")
        wrapper = ContextKVWrapper(model, block_idx=block_idx)
        config_wrapper = Phase1ConfigWrapper(base_config, context_dims[block_idx])
        trainer = MemoryPhase1Trainer(wrapper, config_wrapper, device)

        phase_start = time.time()
        result = trainer.train(
            train_data_splits[block_idx],
            label=f"Context{block_idx}",
            return_all_layers=True,
        )
        phase_time = time.time() - phase_start

        phase1_times.append(phase_time)
        phase1_stats.append(result)

        model.freeze_context_block(block_idx)
        print_flush(f"✓ ContextBlock {block_idx} frozen")

    # ========== Phase 2 Prep: Context chunks収集 ==========
    print_flush("\n[Phase 2 Prep] Collecting context chunks...")
    cache_start = time.time()

    train_context_chunks, train_token_embeds, _ = collect_context_chunks(
        model, train_token_ids, device, chunk_size
    )
    train_targets = train_token_ids[1:].clone()

    val_context_chunks, val_token_embeds, _ = collect_context_chunks(
        model, val_token_ids, device, chunk_size
    )
    val_targets = val_token_ids[1:].clone()

    cache_time = time.time() - cache_start
    print_flush(f"Cache collection: {cache_time:.1f}s")
    print_flush(f"  Train: {train_context_chunks.shape}")
    print_flush(f"  Val: {val_context_chunks.shape}")

    # Effective Rank
    # 最後のチャンクのcontextで計算
    val_last_context = val_context_chunks[:, -1, :]  # [num_val, combined_dim]
    val_er = compute_effective_rank(val_last_context.cpu())
    val_er_pct = val_er / combined_dim * 100
    print_flush(f"Effective Rank: Val={val_er_pct:.1f}%")

    # CPUに移動（メモリ節約）
    train_context_chunks = train_context_chunks.cpu()
    train_token_embeds = train_token_embeds.cpu()
    val_context_chunks = val_context_chunks.cpu()
    val_token_embeds = val_token_embeds.cpu()

    clear_gpu_cache(device)

    # ========== Phase 2: Attention + FFN学習 ==========
    print_flush("\n[Phase 2] Training Context-KV Attention...")

    phase2_start = time.time()
    best_ppl, best_acc, best_epoch, history = train_phase2(
        model=model,
        train_context_chunks=train_context_chunks,
        train_token_embeds=train_token_embeds,
        train_targets=train_targets,
        val_context_chunks=val_context_chunks,
        val_token_embeds=val_token_embeds,
        val_targets=val_targets,
        device=device,
    )
    phase2_time = time.time() - phase2_start

    total_time = sum(phase1_times) + cache_time + phase2_time

    # ========== サマリー ==========
    print_flush("\n" + "=" * 70)
    print_flush(f"SUMMARY - Context-KV Attention Experiment ({num_blocks} blocks)")
    print_flush("=" * 70)
    print_flush("Architecture: ContextKVAttentionLLM")
    print_flush(f"  Context dims: {context_dims}")
    print_flush(f"  Combined dim: {combined_dim}")
    print_flush(f"  Num heads: {num_heads}")
    print_flush(f"  Chunk size: {chunk_size}")
    print_flush(f"Parameters: {params['total']:,}")
    for i in range(num_blocks):
        print_flush(f"Phase 1-{i}: {phase1_times[i]:.1f}s")
    print_flush(f"Cache collection: {cache_time:.1f}s")
    print_flush(f"Phase 2: {phase2_time:.1f}s, epoch {best_epoch}")
    print_flush(f"Effective Rank: {val_er_pct:.1f}%")
    print_flush(f"Val PPL: {best_ppl:.1f}")
    print_flush(f"Val Acc: {best_acc*100:.1f}%")
    print_flush(f"Total time: {total_time:.1f}s")
    print_flush("=" * 70)

    return {
        'num_samples': num_samples,
        'context_dims': context_dims,
        'chunk_size': chunk_size,
        'num_heads': num_heads,
        'val_ppl': best_ppl,
        'val_acc': best_acc,
        'val_er_pct': val_er_pct,
        'phase1_times': phase1_times,
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
    parser.add_argument('-c', '--context-dims', type=str, default='256,256',
                        help='Context dimensions (default: 256,256)')
    parser.add_argument('--chunk-size', type=int, default=100,
                        help='Chunk size for context KV (default: 100)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    args = parser.parse_args()

    # context_dimsのパース
    context_dims = [int(d.strip()) for d in args.context_dims.split(',')]

    print_flush("=" * 70)
    print_flush("CONTEXT-KV ATTENTION EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {args.samples}")
    print_flush(f"Context dims: {context_dims}")
    print_flush(f"Chunk size: {args.chunk_size}")
    print_flush(f"Num heads: {args.num_heads}")
    print_flush("=" * 70)

    run_context_kv_experiment(
        num_samples=args.samples,
        context_dims=context_dims,
        chunk_size=args.chunk_size,
        num_heads=args.num_heads,
        seed=args.seed,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
