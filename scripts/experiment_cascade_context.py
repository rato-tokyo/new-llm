#!/usr/bin/env python3
"""
Multi Context 実験スクリプト（N分割方式）

N個のContextBlockを異なるデータで学習し、異なる表現を獲得する。

アーキテクチャ（1層固定、Nブロック）:
  Phase 1[i]: ContextBlock[i] を i 番目のデータ区間で学習
    - 初期入力: ゼロベクトル
    - データ: tokens[i*split:(i+1)*split]

  Phase 2:
    - 順次処理で全データのコンテキストキャッシュを収集
    - concat(context[0], ..., context[N-1]) で TokenBlock を学習

使用方法:
  python3 scripts/experiment_cascade_context.py -s 2000 -n 2  # 2ブロック（デフォルト）
  python3 scripts/experiment_cascade_context.py -s 2000 -n 4  # 4ブロック
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Dict, Optional, Any, List

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import Config
from src.models.cascade import CascadeContextLLM, SingleContextWrapper
from src.trainers.phase1.memory import MemoryPhase1Trainer
from src.trainers.phase2.cascade import CascadePhase2Trainer
from src.evaluation.metrics import analyze_fixed_points
from src.providers.data import MemoryDataProvider
from src.utils.io import print_flush
from src.utils.device import clear_gpu_cache
from src.utils.seed import set_seed
from src.utils.cache import collect_multiblock_cache_to_files, ChunkedCacheDataset
from src.config.wrappers import Phase1ConfigWrapper, Phase2ConfigWrapper
from config.experiment import DataConfig


def run_cascade_context_experiment(
    num_samples: int = 2000,
    context_dim: int = 500,
    num_context_blocks: int = 2,
    seed: int = 42,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Cascade Context 実験を実行

    Args:
        num_samples: サンプル数
        context_dim: 各ContextBlockの出力次元
        num_context_blocks: ContextBlockの数
        seed: 乱数シード
        output_dir: 出力ディレクトリ
    """

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print_flush(f"Device: {device} ({gpu_name}, {gpu_mem:.1f}GB)")
    else:
        print_flush(f"Device: {device}")

    base_config = Config()

    # データ読み込み
    print_flush("Loading data...")
    data_config = DataConfig.from_base(base_config, num_samples=num_samples)
    data_provider = MemoryDataProvider(data_config)
    train_token_ids, val_token_ids = data_provider.load_data()
    train_token_ids = train_token_ids.to(device)
    val_token_ids = val_token_ids.to(device)

    num_train_tokens = len(train_token_ids)
    num_val_tokens = len(val_token_ids)
    print_flush(f"Data: {num_train_tokens:,} train, {num_val_tokens:,} val tokens")

    # N分割方式: データをN等分
    train_data_splits: List[torch.Tensor] = []
    split_size = num_train_tokens // num_context_blocks
    split_info = []
    for i in range(num_context_blocks):
        start_idx = i * split_size
        if i == num_context_blocks - 1:
            end_idx = num_train_tokens
        else:
            end_idx = (i + 1) * split_size + 1
        train_data_splits.append(train_token_ids[start_idx:end_idx])
        split_info.append(f"Block{i}={len(train_data_splits[-1])-1:,}")
    print_flush(f"Split: {', '.join(split_info)} tokens")

    # モデル作成
    combined_dim = context_dim * num_context_blocks
    print_flush(f"\nCreating CascadeContextLLM (cd={context_dim}x{num_context_blocks}={combined_dim})...")
    model = CascadeContextLLM(
        vocab_size=base_config.vocab_size,
        embed_dim=base_config.embed_dim,
        context_dim=context_dim,
        num_context_blocks=num_context_blocks,
    )
    model.to(device)

    params = model.num_params()
    print_flush(f"Parameters: {params['total']:,} total")
    print_flush(f"  ContextBlocks ({num_context_blocks}): {params['total_context_blocks']:,}")
    for i in range(num_context_blocks):
        print_flush(f"    Block {i}: {params[f'context_block_{i}']:,}")
    print_flush(f"  TokenBlock: {params['token_block']:,}")

    config_wrapper = Phase1ConfigWrapper(base_config, context_dim)

    # ========== Phase 1: N分割方式で各ブロックを学習 ==========
    train_context_caches: List[torch.Tensor] = []
    phase1_times: List[float] = []
    phase1_stats: List[Dict[str, Any]] = []

    for block_idx in range(num_context_blocks):
        print_flush(f"\n[Phase 1-{block_idx}] Training ContextBlock {block_idx} on split {block_idx}...")
        wrapper = SingleContextWrapper(model, block_idx=block_idx)
        trainer = MemoryPhase1Trainer(wrapper, config_wrapper, device)

        phase_start = time.time()
        result = trainer.train(
            train_data_splits[block_idx],
            label=f"Context{block_idx}",
            return_all_layers=True,
        )
        phase_time = time.time() - phase_start

        assert result.cache is not None
        train_context_caches.append(result.cache)

        stats = trainer._training_stats
        phase1_times.append(phase_time)
        phase1_stats.append(stats)

        print_flush(f"Phase 1-{block_idx}: {phase_time:.1f}s, {stats.get('iterations', 0)} iter, "
                    f"conv={stats.get('convergence_rate', 0)*100:.0f}%")

        model.freeze_context_block(block_idx)
        print_flush(f"✓ ContextBlock {block_idx} frozen")

    phase1_total_time = sum(phase1_times)

    # Phase 1で使ったキャッシュを解放
    del train_context_caches
    clear_gpu_cache(device)

    # ========== Phase 2 Prep: チャンク単位でキャッシュ収集 ==========
    if output_dir is None:
        output_dir = f"importants/logs/temp_cache_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cache_dir = os.path.join(output_dir, "cache_chunks")
    print_flush(f"\n[Phase 2 Prep] Collecting multi context cache (chunked, {num_context_blocks} blocks)...")
    cache_start = time.time()

    train_num_tokens, _, train_chunk_files = collect_multiblock_cache_to_files(
        model, train_token_ids, device, cache_dir, prefix="train"
    )
    val_num_tokens, _, val_chunk_files = collect_multiblock_cache_to_files(
        model, val_token_ids, device, cache_dir, prefix="val"
    )

    cache_time = time.time() - cache_start
    print_flush(f"Cache collection: {cache_time:.1f}s")
    print_flush(f"  Train: {train_num_tokens:,} tokens, {len(train_chunk_files)} chunks")
    print_flush(f"  Val: {val_num_tokens:,} tokens, {len(val_chunk_files)} chunks")

    # Effective Rank計算
    val_dataset = ChunkedCacheDataset(val_chunk_files)
    val_context_cache, val_token_embeds = val_dataset.get_all_data()
    val_metrics = analyze_fixed_points(val_context_cache, label="Val", verbose=False)
    val_er = val_metrics['effective_rank']
    val_er_pct = val_er / combined_dim * 100
    print_flush(f"Effective Rank: Val={val_er_pct:.1f}%")

    # ========== Phase 2: TokenBlock の学習 ==========
    print_flush(f"\n[Phase 2] Training TokenBlock with concatenated context (cd={combined_dim})...")

    train_dataset = ChunkedCacheDataset(train_chunk_files)
    train_context_cache, train_token_embeds = train_dataset.get_all_data()

    phase2_config = Phase2ConfigWrapper(base_config)
    phase2_trainer = CascadePhase2Trainer(model, phase2_config, device)

    phase2_start = time.time()
    history = phase2_trainer.train(
        train_token_ids=train_token_ids,
        val_token_ids=val_token_ids,
        train_context_cache=train_context_cache,
        train_token_embeds=train_token_embeds,
        val_context_cache=val_context_cache,
        val_token_embeds=val_token_embeds,
    )
    phase2_time = time.time() - phase2_start

    # キャッシュを解放
    del train_context_cache, train_token_embeds
    del val_context_cache, val_token_embeds
    del train_dataset, val_dataset
    clear_gpu_cache(device)

    best_epoch = history['best_epoch']
    best_ppl = history['val_ppl'][best_epoch - 1]
    best_acc = history['val_acc'][best_epoch - 1]

    total_time = phase1_total_time + cache_time + phase2_time

    print_flush(f"\nPhase 2: {phase2_time:.1f}s, Best epoch {best_epoch}")
    print_flush(f"Result: PPL={best_ppl:.1f}, Acc={best_acc*100:.1f}%")
    print_flush(f"Total time: {total_time:.1f}s")

    # ========== 結果サマリー ==========
    print_flush("\n" + "=" * 70)
    print_flush(f"SUMMARY - Multi Context Experiment ({num_context_blocks} blocks)")
    print_flush("=" * 70)
    print_flush(f"Architecture: CascadeContextLLM ({num_context_blocks} blocks, 1L each)")
    for i in range(num_context_blocks):
        block_tokens = len(train_data_splits[i]) - 1
        print_flush(f"  ContextBlock {i}: cd={context_dim}, trained on {block_tokens:,} tokens")
    print_flush(f"  TokenBlock: cd={combined_dim} (concatenated)")
    print_flush(f"Parameters: {params['total']:,}")
    for i in range(num_context_blocks):
        print_flush(f"Phase 1-{i}: {phase1_times[i]:.1f}s, conv={phase1_stats[i].get('convergence_rate', 0)*100:.0f}%")
    print_flush(f"Cache collection: {cache_time:.1f}s")
    print_flush(f"Phase 2: {phase2_time:.1f}s, epoch {best_epoch}")
    print_flush(f"Effective Rank: {val_er_pct:.1f}% (of {combined_dim})")
    print_flush(f"Val PPL: {best_ppl:.1f}")
    print_flush(f"Val Acc: {best_acc*100:.1f}%")
    print_flush(f"Total time: {total_time:.1f}s")
    print_flush("=" * 70)

    # キャッシュディレクトリ削除
    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)

    data_provider.close()

    return {
        'val_ppl': best_ppl,
        'val_acc': best_acc,
        'val_er': val_er,
        'val_er_pct': val_er_pct,
        'phase1_times': phase1_times,
        'phase2_time': phase2_time,
        'total_time': total_time,
        'history': history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Cascade Context Experiment')
    parser.add_argument('-s', '--samples', type=int, default=2000,
                        help='Number of samples (default: 2000)')
    parser.add_argument('-n', '--num-blocks', type=int, default=2,
                        help='Number of context blocks (default: 2)')
    parser.add_argument('-c', '--context-dim', type=int, default=256,
                        help='Context dimension per block (default: 256)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory')

    args = parser.parse_args()

    print_flush("=" * 70)
    print_flush("CASCADE CONTEXT EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Samples: {args.samples}")
    print_flush(f"Blocks: {args.num_blocks}")
    print_flush(f"Context dim per block: {args.context_dim}")
    print_flush(f"Combined context dim: {args.context_dim * args.num_blocks}")
    print_flush("=" * 70)

    run_cascade_context_experiment(
        num_samples=args.samples,
        context_dim=args.context_dim,
        num_context_blocks=args.num_blocks,
        seed=args.seed,
        output_dir=args.output,
    )

    print_flush("\nDONE")


if __name__ == "__main__":
    main()
