#!/usr/bin/env python3
"""
Multi-Block Sample Size Search Experiment

複数ブロック（カスケード連結）でのサンプル数探索実験。
最後に指数減衰モデルでのフィッティングも自動実行。

使用方法:
  python3 scripts/experiment_multiblock_sample_search.py --start 200 --end 1600 -c 256
  python3 scripts/experiment_multiblock_sample_search.py --start 200 --end 3200 -c 256 -p 1
"""

import os
import sys
import argparse
import time
from datetime import datetime
from typing import Dict, Optional, Any, List

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.optimize import curve_fit

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


# ============================================================
# Exponential Decay Fitting
# ============================================================

def fit_exp_decay(samples: np.ndarray, ppls: np.ndarray) -> Dict[str, Any]:
    """指数減衰モデルでフィッティング"""

    def exp_decay(n: np.ndarray, ppl_min: float, A: float, b: float, c: float) -> np.ndarray:
        return ppl_min + A * np.exp(-b * (n ** c))

    try:
        popt, _ = curve_fit(
            exp_decay, samples, ppls,
            p0=[100, 300, 0.01, 0.5],
            bounds=([0, 0, 0, 0], [200, 2000, 1, 1]),
            maxfev=10000
        )
        ppl_min, A, b, c = popt

        pred = exp_decay(samples, *popt)
        ss_res = np.sum((ppls - pred) ** 2)
        ss_tot = np.sum((ppls - np.mean(ppls)) ** 2)
        r2 = 1 - ss_res / ss_tot

        return {
            'success': True,
            'ppl_min': ppl_min,
            'A': A,
            'b': b,
            'c': c,
            'r2': r2,
            'predictions': pred,
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ============================================================
# Main Search Function
# ============================================================

def run_multiblock_sample_search(
    context_dims: List[int] = [256, 256],
    start_samples: int = 200,
    end_samples: int = 1600,
    seed: int = 42,
    output_dir: Optional[str] = None,
    prev_context_steps: int = 0,
) -> Dict[str, Any]:
    """Multi-block sample size search"""

    set_seed(seed)
    num_blocks = len(context_dims)

    # サンプル数リスト（倍増）
    sample_sizes: List[int] = []
    current = start_samples
    while current <= end_samples:
        sample_sizes.append(current)
        current *= 2

    # prev_context_steps=0: 現在のみ, =1: 現在+1つ前, など
    combined_dim = sum(context_dims) * (1 + prev_context_steps)
    dims_str = '+'.join(map(str, context_dims))

    print_flush("=" * 70)
    print_flush("MULTI-BLOCK SAMPLE SIZE SEARCH")
    print_flush("=" * 70)
    print_flush(f"Context dims: {context_dims}")
    print_flush(f"Num blocks: {num_blocks}")
    print_flush(f"Prev context steps: {prev_context_steps}")
    print_flush(f"Combined context dim: {dims_str}={combined_dim}")
    print_flush(f"Sample sizes: {sample_sizes}")
    if output_dir:
        print_flush(f"Output: {output_dir}")
    print_flush("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print_flush(f"Device: {device} ({gpu_name}, {gpu_mem:.1f}GB)")
    else:
        print_flush(f"Device: {device}")

    base_config = Config()

    results: List[Dict[str, Any]] = []
    best_ppl = float('inf')
    best_samples = start_samples

    for num_samples in sample_sizes:
        print_flush(f"\n{'='*70}")
        print_flush(f"[SAMPLES={num_samples}] Starting experiment ({num_blocks} blocks)...")
        print_flush("=" * 70)

        experiment_start = time.time()

        # データ読み込み
        print_flush(f"\nLoading data ({num_samples} samples)...")
        data_config = DataConfig.from_base(base_config, num_samples=num_samples)
        data_provider = MemoryDataProvider(data_config)
        train_token_ids, val_token_ids = data_provider.load_data()
        train_token_ids = train_token_ids.to(device)
        val_token_ids = val_token_ids.to(device)

        num_train_tokens = len(train_token_ids)
        num_val_tokens = len(val_token_ids)
        print_flush(f"Data: {num_train_tokens:,} train, {num_val_tokens:,} val tokens")

        # N分割
        train_data_splits: List[torch.Tensor] = []
        split_size = num_train_tokens // num_blocks
        for i in range(num_blocks):
            start_idx = i * split_size
            end_idx = num_train_tokens if i == num_blocks - 1 else (i + 1) * split_size + 1
            train_data_splits.append(train_token_ids[start_idx:end_idx])

        # モデル作成
        dim_info = f"cd={dims_str}"
        if prev_context_steps > 0:
            dim_info += f"x(1+{prev_context_steps})"
        dim_info += f"={combined_dim}"
        print_flush(f"\nCreating CascadeContextLLM ({dim_info})...")
        model = CascadeContextLLM(
            vocab_size=base_config.vocab_size,
            embed_dim=base_config.embed_dim,
            context_dims=context_dims,
            prev_context_steps=prev_context_steps,
        )
        model.to(device)

        # Phase 1: 各ブロック学習
        phase1_times = []
        phase1_convs = []

        for block_idx in range(num_blocks):
            print_flush(f"\n[Phase 1-{block_idx}] Training ContextBlock {block_idx} (cd={context_dims[block_idx]})...")
            wrapper = SingleContextWrapper(model, block_idx=block_idx)
            config_wrapper = Phase1ConfigWrapper(base_config, context_dims[block_idx], patience=2)
            trainer = MemoryPhase1Trainer(wrapper, config_wrapper, device)

            phase_start = time.time()
            _ = trainer.train(
                train_data_splits[block_idx],
                label=f"Context{block_idx}",
                return_all_layers=True,
            )
            phase_time = time.time() - phase_start

            stats = trainer._training_stats
            phase1_times.append(phase_time)
            phase1_convs.append(stats.get('convergence_rate', 0))

            iters = stats.get('iterations', 0)
            conv = stats.get('convergence_rate', 0)
            print_flush(f"Phase 1-{block_idx}: {phase_time:.1f}s, {iters} iter, conv={conv*100:.0f}%")

            model.freeze_context_block(block_idx)

        phase1_total_time = sum(phase1_times)

        # Phase 2 Prep
        print_flush("\n[Phase 2 Prep] Collecting context cache...")
        cache_start = time.time()

        cache_dir = f"/tmp/multiblock_cache_{num_samples}_{datetime.now().strftime('%H%M%S')}"
        train_num, _, train_chunks = collect_multiblock_cache_to_files(
            model, train_token_ids, device, cache_dir, prefix="train",
            prev_context_steps=prev_context_steps,
        )
        val_num, _, val_chunks = collect_multiblock_cache_to_files(
            model, val_token_ids, device, cache_dir, prefix="val",
            prev_context_steps=prev_context_steps,
        )

        train_dataset = ChunkedCacheDataset(train_chunks)
        val_dataset = ChunkedCacheDataset(val_chunks)
        train_context_cache, train_token_embeds = train_dataset.get_all_data()
        val_context_cache, val_token_embeds = val_dataset.get_all_data()

        cache_time = time.time() - cache_start
        print_flush(f"Cache collection: {cache_time:.1f}s")

        # Effective Rank
        er_analysis = analyze_fixed_points(val_context_cache, label="Val", verbose=False)
        val_er = er_analysis.get('effective_rank', 0)
        val_er_pct = val_er / combined_dim * 100
        print_flush(f"Effective Rank: Val={val_er_pct:.1f}% ({val_er:.1f}/{combined_dim})")

        # Phase 2
        print_flush("\n[Phase 2] Training TokenBlock...")
        phase2_config = Phase2ConfigWrapper(base_config)
        phase2_trainer = CascadePhase2Trainer(model, phase2_config, device)

        phase2_start = time.time()
        history = phase2_trainer.train(
            train_token_ids, val_token_ids,
            train_context_cache, train_token_embeds,
            val_context_cache, val_token_embeds,
        )
        phase2_time = time.time() - phase2_start

        total_time = time.time() - experiment_start

        val_ppl = history['best_val_ppl']
        val_acc = history['val_acc'][history['best_epoch'] - 1]

        result = {
            'num_samples': num_samples,
            'num_train_tokens': num_train_tokens,
            'val_ppl': val_ppl,
            'val_acc': val_acc,
            'val_er': val_er,
            'val_er_pct': val_er_pct,
            'phase1_time': phase1_total_time,
            'phase2_time': phase2_time,
            'total_time': total_time,
        }
        results.append(result)

        print_flush(f"\n[Result] samples={num_samples}: PPL={val_ppl:.1f}, Acc={val_acc*100:.1f}%, "
                    f"ER={val_er_pct:.1f}%, Time={total_time:.1f}s")

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_samples = num_samples
            print_flush(f"  ★ New best! samples={num_samples}, PPL={val_ppl:.1f}")

        # メモリ解放
        del model, train_context_cache, val_context_cache
        del train_token_embeds, val_token_embeds
        del train_dataset, val_dataset
        del train_token_ids, val_token_ids
        data_provider.close()
        clear_gpu_cache(device)

        # キャッシュディレクトリ削除
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)

    # ========== 指数減衰フィッティング ==========
    print_flush("\n" + "=" * 70)
    print_flush("EXPONENTIAL DECAY MODEL FITTING")
    print_flush("=" * 70)

    samples_arr = np.array([r['num_samples'] for r in results])
    ppls_arr = np.array([r['val_ppl'] for r in results])

    fit_result = fit_exp_decay(samples_arr, ppls_arr)

    if fit_result['success']:
        print_flush("\nModel: PPL = PPL_min + A × exp(-b × n^c)")
        print_flush(f"  PPL_min = {fit_result['ppl_min']:.2f}")
        print_flush(f"  A = {fit_result['A']:.2f}")
        print_flush(f"  b = {fit_result['b']:.6f}")
        print_flush(f"  c = {fit_result['c']:.4f}")
        print_flush(f"  R² = {fit_result['r2']:.6f}")

        print_flush("\nPrediction vs Actual:")
        for s, actual, pred in zip(samples_arr, ppls_arr, fit_result['predictions']):
            err = (pred - actual) / actual * 100
            print_flush(f"  {int(s):>6} samples: actual={actual:.1f}, pred={pred:.1f} ({err:+.1f}%)")

        print_flush(f"\n★ Theoretical limit (PPL_min): {fit_result['ppl_min']:.1f}")
    else:
        print_flush(f"Fitting failed: {fit_result['error']}")

    # 最終サマリー
    print_flush("\n" + "=" * 70)
    print_flush(f"SUMMARY - Multi-Block Sample Size Search ({num_blocks} blocks)")
    print_flush("=" * 70)
    print_flush(f"Context dims: {context_dims}")
    print_flush(f"Num blocks: {num_blocks}")
    print_flush(f"Prev context steps: {prev_context_steps}")
    print_flush(f"Combined context dim: {dims_str}={combined_dim}")
    print_flush(f"Sample sizes: {sample_sizes}")
    print_flush("\nResults:")
    print_flush(f"{'Samples':>8} | {'Tokens':>12} | {'PPL':>8} | {'Acc':>6} | {'ER%':>6} | {'Time':>8}")
    print_flush("-" * 60)
    for r in results:
        marker = " ★" if r['num_samples'] == best_samples else ""
        print_flush(f"{r['num_samples']:>8} | {r['num_train_tokens']:>12,} | {r['val_ppl']:>8.1f} | "
                    f"{r['val_acc']*100:>5.1f}% | {r['val_er_pct']:>5.1f}% | {r['total_time']:>7.1f}s{marker}")
    print_flush("-" * 60)
    print_flush(f"\n★ Best: samples={best_samples}, PPL={best_ppl:.1f}")

    if fit_result['success']:
        print_flush(f"\n★ Exp Decay PPL_min: {fit_result['ppl_min']:.1f} (R²={fit_result['r2']:.4f})")

    print_flush("=" * 70)

    # 結果を保存
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result_file = os.path.join(output_dir, "results.txt")
        with open(result_file, 'w') as f:
            f.write(f"Multi-Block Sample Size Search Results ({num_blocks} blocks)\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Context dims: {context_dims}\n")
            f.write(f"Combined context dim: {dims_str}={combined_dim}\n")
            f.write(f"Sample sizes: {sample_sizes}\n\n")
            f.write(f"{'Samples':>8} | {'Tokens':>12} | {'PPL':>8} | {'Acc':>6} | {'ER%':>6} | {'Time':>8}\n")
            f.write("-" * 60 + "\n")
            for r in results:
                marker = " *" if r['num_samples'] == best_samples else ""
                f.write(f"{r['num_samples']:>8} | {r['num_train_tokens']:>12,} | {r['val_ppl']:>8.1f} | "
                        f"{r['val_acc']*100:>5.1f}% | {r['val_er_pct']:>5.1f}% | {r['total_time']:>7.1f}s{marker}\n")
            f.write("-" * 60 + "\n")
            f.write(f"\nBest: samples={best_samples}, PPL={best_ppl:.1f}\n")

            if fit_result['success']:
                f.write("\n\nExponential Decay Fitting:\n")
                f.write("  PPL = PPL_min + A × exp(-b × n^c)\n")
                f.write(f"  PPL_min = {fit_result['ppl_min']:.2f}\n")
                f.write(f"  A = {fit_result['A']:.2f}\n")
                f.write(f"  b = {fit_result['b']:.6f}\n")
                f.write(f"  c = {fit_result['c']:.4f}\n")
                f.write(f"  R² = {fit_result['r2']:.6f}\n")

        print_flush(f"\nResults saved to: {result_file}")

    return {
        'results': results,
        'best_samples': best_samples,
        'best_ppl': best_ppl,
        'fit_result': fit_result,
    }


def parse_context_dims(value: str) -> List[int]:
    """context_dims引数をパース"""
    parts = value.split(',')
    return [int(p.strip()) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Multi-Block Sample Size Search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # 2ブロック、各256次元（従来の使用法）
  python3 scripts/experiment_multiblock_sample_search.py --start 200 --end 1600 -c 256 -n 2

  # 2ブロック、異なる次元（256, 128）
  python3 scripts/experiment_multiblock_sample_search.py --start 200 --end 1600 -c 256,128

  # 3ブロック、異なる次元
  python3 scripts/experiment_multiblock_sample_search.py --start 200 --end 1600 -c 256,128,64
'''
    )
    parser.add_argument('--context-dims', '-c', type=str, default='256,256',
                        help='Context dimensions: single int "256" or list "256,128" (default: 256,256)')
    parser.add_argument('--num-blocks', '-n', type=int, default=None,
                        help='Number of context blocks (ignored if -c is a list)')
    parser.add_argument('--start', type=int, default=200,
                        help='Starting sample size (default: 200)')
    parser.add_argument('--end', type=int, default=1600,
                        help='Ending sample size (default: 1600)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--prev-context', '-p', type=int, default=0,
                        help='Number of previous context steps to concatenate (0=disabled)')

    args = parser.parse_args()

    # context_dimsのパース
    context_dims = parse_context_dims(args.context_dims)

    # -n が指定された場合、同じ次元をn個に拡張（後方互換性）
    if args.num_blocks is not None and len(context_dims) == 1:
        context_dims = context_dims * args.num_blocks

    num_blocks = len(context_dims)

    # 出力ディレクトリ
    if args.output:
        output_dir = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"importants/logs/{timestamp}_multiblock_{num_blocks}b_sample_search"

    run_multiblock_sample_search(
        context_dims=context_dims,
        start_samples=args.start,
        end_samples=args.end,
        seed=args.seed,
        output_dir=output_dir,
        prev_context_steps=args.prev_context,
    )

    print_flush("\n" + "=" * 70)
    print_flush("DONE")
    print_flush("=" * 70)


if __name__ == "__main__":
    main()
