#!/usr/bin/env python3
"""
多様性損失アルゴリズム比較実験（Phase 1 + Phase 2 + α値分析）

2つのアルゴリズム（MCDL, ODCM）で Phase 1 + Phase 2 を実行し、
α値（スケーリング指数）を比較する。

使用方法:
  # デフォルト: context_dim=1000, layer=1, samples=50,100,200
  python3 scripts/diversity_full_experiment.py

  # 特定のアルゴリズムのみ実行
  python3 scripts/diversity_full_experiment.py -a MCDL ODCM

  # サンプルサイズを指定
  python3 scripts/diversity_full_experiment.py -s 50 100 200

  # context_dimを指定
  python3 scripts/diversity_full_experiment.py -c 1000

Colab実行用:
  !cd /content/new-llm && python3 scripts/diversity_full_experiment.py
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Callable, Dict, Any, List

import numpy as np
from scipy import stats

import torch

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ResidualConfig
from src.models import LLM
from src.trainers.phase1 import FlexibleDiversityTrainer
from src.trainers.phase2 import Phase2Trainer
from src.evaluation.metrics import analyze_fixed_points
from src.experiments.config import DataConfig, Phase1Config, Phase2Config
from src.providers.data import MemoryDataProvider
from src.utils.io import print_flush
from src.utils.device import clear_gpu_cache
from src.utils.seed import set_seed
from src.losses.diversity import (
    DIVERSITY_ALGORITHMS,
    ALGORITHM_DESCRIPTIONS,
)


# =============================================================================
# スケーリング則計算
# =============================================================================

def calculate_scaling_law(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """スケーリング則を計算: PPL = A × tokens^α"""
    if len(results) < 2:
        return {'alpha': None, 'A': None, 'r_squared': None}

    tokens = np.array([r['train_tokens'] for r in results])
    ppl = np.array([r['val_ppl'] for r in results])

    # NaNやInfを除外
    valid_mask = np.isfinite(tokens) & np.isfinite(ppl) & (ppl > 0)
    if valid_mask.sum() < 2:
        return {'alpha': None, 'A': None, 'r_squared': None}

    tokens = tokens[valid_mask]
    ppl = ppl[valid_mask]

    # 対数変換
    log_tokens = np.log(tokens)
    log_ppl = np.log(ppl)

    # 線形回帰
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_tokens, log_ppl)

    return {
        'alpha': slope,
        'A': np.exp(intercept),
        'r_squared': r_value ** 2,
        'p_value': p_value,
        'std_err': std_err,
    }


# =============================================================================
# 単一実験実行
# =============================================================================

def run_single_experiment(
    algorithm_name: str,
    diversity_fn: Callable[[torch.Tensor], torch.Tensor],
    num_samples: int,
    base_config: ResidualConfig,
    device: torch.device,
    seed: int = 42,
    context_dim: int = 1000
) -> Dict[str, Any]:
    """単一の実験を実行（Phase 1 + Phase 2）"""

    set_seed(seed)

    # データ読み込み用設定（共有設定クラスを使用）
    data_config = DataConfig.from_base(base_config, num_samples=num_samples)

    # データプロバイダー
    data_provider = MemoryDataProvider(data_config)
    train_token_ids, val_token_ids = data_provider.load_data()
    train_token_ids = train_token_ids.to(device)
    val_token_ids = val_token_ids.to(device)

    num_train_tokens = len(train_token_ids)
    num_val_tokens = len(val_token_ids)

    print_flush(f"    Data: {num_train_tokens:,} train, {num_val_tokens:,} val tokens")

    # モデル作成
    set_seed(seed)
    model = LLM(
        vocab_size=base_config.vocab_size,
        embed_dim=base_config.embed_dim,
        context_dim=context_dim,
        num_layers=base_config.num_layers,
        num_input_tokens=base_config.num_input_tokens,
        use_pretrained_embeddings=base_config.use_pretrained_embeddings,
        use_weight_tying=base_config.use_weight_tying,
        config=base_config
    )
    model.to(device)

    # Phase 1用設定（共有設定クラスを使用）
    # dist_reg_weight = 0.9 で多様性90%, CVFP 10%（前回実験と同じ設定）
    phase1_config = Phase1Config.from_base(
        base_config, device,
        context_dim=context_dim,
        dist_reg_weight=0.9,  # 多様性90%, CVFP 10%
    )

    # Phase 1 トレーナー作成（FlexibleDiversityTrainer使用）
    phase1_trainer = FlexibleDiversityTrainer(
        model, phase1_config, device,
        diversity_fn=diversity_fn,
        algorithm_name=algorithm_name
    )

    # Phase 1 実行
    phase1_start = time.time()
    train_result = phase1_trainer.train(
        train_token_ids,
        label=algorithm_name,
        return_all_layers=True,
        val_token_ids=val_token_ids
    )
    phase1_time = time.time() - phase1_start

    # train_resultの型チェック
    if isinstance(train_result, tuple):
        train_contexts, train_context_cache, train_token_embeds = train_result
    else:
        raise ValueError("Expected tuple from train() with return_all_layers=True")

    # Phase 1 統計を取得
    phase1_stats = phase1_trainer._training_stats
    phase1_iterations = phase1_stats.get('iterations', 0)
    best_val_er = phase1_stats.get('best_val_er', 0.0)

    # 検証データのキャッシュ収集
    val_result = phase1_trainer.evaluate(val_token_ids, return_all_layers=True)
    assert isinstance(val_result, tuple), "evaluate with return_all_layers=True must return tuple"
    val_contexts, val_context_cache, val_token_embeds = val_result

    # Effective Rank計算
    train_metrics = analyze_fixed_points(train_contexts, label="Train", verbose=False)
    val_metrics = analyze_fixed_points(val_contexts, label="Val", verbose=False)
    train_er = train_metrics['effective_rank']
    val_er = val_metrics['effective_rank']
    train_er_pct = train_er / context_dim * 100
    val_er_pct = val_er / context_dim * 100
    best_val_er_pct = best_val_er / context_dim * 100

    print_flush(f"    Phase 1: {phase1_time:.1f}s, {phase1_iterations} iter, "
                f"ER={train_er_pct:.1f}%/{best_val_er_pct:.1f}%")

    # Phase 2用設定（共有設定クラスを使用）
    phase2_config = Phase2Config.from_base(
        base_config, device,
        context_dim=context_dim,
    )

    # Phase 2 トレーナー作成
    phase2_trainer = Phase2Trainer(model, phase2_config)

    # Phase 2 実行
    phase2_start = time.time()
    history = phase2_trainer.train_full(
        train_token_ids=train_token_ids,
        val_token_ids=val_token_ids,
        device=device,
        train_context_cache=train_context_cache,
        train_token_embeds=train_token_embeds,
        val_context_cache=val_context_cache,
        val_token_embeds=val_token_embeds
    )
    phase2_time = time.time() - phase2_start

    best_epoch = history['best_epoch']
    best_ppl = history['val_ppl'][best_epoch - 1]
    best_acc = history['val_acc'][best_epoch - 1]
    best_train_ppl = history['train_ppl'][best_epoch - 1]

    print_flush(f"    Phase 2: {phase2_time:.1f}s, PPL={best_ppl:.1f}, Acc={best_acc*100:.1f}%")

    total_time = phase1_time + phase2_time

    # メモリ解放
    del model, phase1_trainer, phase2_trainer
    del train_contexts, val_contexts
    del train_context_cache, val_context_cache
    del train_token_embeds, val_token_embeds
    data_provider.close()
    clear_gpu_cache(device)

    return {
        'algorithm': algorithm_name,
        'context_dim': context_dim,
        'num_samples': num_samples,
        'train_tokens': num_train_tokens,
        'val_tokens': num_val_tokens,
        'phase1_iterations': phase1_iterations,
        'phase1_time': phase1_time,
        'train_er': train_er,
        'train_er_pct': train_er_pct,
        'val_er': val_er,
        'val_er_pct': val_er_pct,
        'best_val_er': best_val_er,
        'best_val_er_pct': best_val_er_pct,
        'phase2_time': phase2_time,
        'best_epoch': best_epoch,
        'train_ppl': best_train_ppl,
        'val_ppl': best_ppl,
        'val_acc': best_acc,
        'total_time': total_time,
    }


# =============================================================================
# 全実験実行
# =============================================================================

def run_all_experiments(
    algorithms: List[str],
    sample_sizes: List[int],
    context_dim: int,
    base_config: ResidualConfig,
    device: torch.device,
    output_dir: str
) -> Dict[str, List[Dict[str, Any]]]:
    """全実験を実行"""

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    total_experiments = len(algorithms) * len(sample_sizes)
    current = 0

    for algorithm_name in algorithms:
        if algorithm_name not in DIVERSITY_ALGORITHMS:
            print_flush(f"Unknown algorithm: {algorithm_name}, skipping")
            continue

        diversity_fn = DIVERSITY_ALGORITHMS[algorithm_name]
        print_flush(f"\n{'='*70}")
        print_flush(f"Algorithm: {algorithm_name} - {ALGORITHM_DESCRIPTIONS.get(algorithm_name, '')}")
        print_flush(f"{'='*70}")

        algo_results = []

        for num_samples in sample_sizes:
            current += 1
            print_flush(f"\n[{current}/{total_experiments}] {algorithm_name} | "
                       f"ctx_dim={context_dim} | {num_samples} samples")

            try:
                result = run_single_experiment(
                    algorithm_name=algorithm_name,
                    diversity_fn=diversity_fn,
                    num_samples=num_samples,
                    base_config=base_config,
                    device=device,
                    context_dim=context_dim
                )
                algo_results.append(result)

            except Exception as e:
                print_flush(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                algo_results.append({
                    'algorithm': algorithm_name,
                    'context_dim': context_dim,
                    'num_samples': num_samples,
                    'error': str(e),
                })

        all_results[algorithm_name] = algo_results

        # アルゴリズムごとの結果を保存
        algo_dir = os.path.join(output_dir, algorithm_name)
        os.makedirs(algo_dir, exist_ok=True)
        with open(os.path.join(algo_dir, 'results.json'), 'w') as f:
            json.dump(algo_results, f, indent=2)

    return all_results


# =============================================================================
# 結果表示
# =============================================================================

def print_results_table(all_results: Dict[str, List[Dict[str, Any]]], context_dim: int):
    """結果をテーブル形式で表示"""

    print_flush("\n" + "=" * 130)
    print_flush("FULL EXPERIMENT RESULTS (Phase 1 + Phase 2)")
    print_flush("=" * 130)

    # 全アルゴリズムの結果を統合
    all_data = []
    for algo_name, results in all_results.items():
        for r in results:
            if 'error' not in r:
                all_data.append(r)

    if not all_data:
        print_flush("No valid results.")
        return

    # ヘッダー
    header = (f"{'Algo':<6} {'Samples':>7} {'Tokens':>10} {'P1 Iter':>7} "
              f"{'Train ER':>9} {'Val ER':>7} {'BestValER':>9} "
              f"{'T.PPL':>7} {'V.PPL':>7} {'Acc':>6} {'Time':>6}")
    print_flush(header)
    print_flush("-" * 130)

    for r in all_data:
        print_flush(
            f"{r['algorithm']:<6} "
            f"{r['num_samples']:>7} "
            f"{r['train_tokens']:>10,} "
            f"{r['phase1_iterations']:>7} "
            f"{r['train_er_pct']:>8.1f}% "
            f"{r['val_er_pct']:>6.1f}% "
            f"{r['best_val_er_pct']:>8.1f}% "
            f"{r['train_ppl']:>7.1f} "
            f"{r['val_ppl']:>7.1f} "
            f"{r['val_acc']*100:>5.1f}% "
            f"{r['total_time']:>5.0f}s"
        )

    print_flush("=" * 130)


def print_scaling_analysis(all_results: Dict[str, List[Dict[str, Any]]], output_dir: str):
    """スケーリング分析を表示・保存"""

    print_flush("\n" + "=" * 100)
    print_flush("SCALING LAW ANALYSIS (PPL = A × tokens^α)")
    print_flush("=" * 100)

    header = f"{'Algorithm':<10} {'α (PPL)':>10} {'A':>12} {'R²':>8} {'Best Val PPL':>12} {'Best Acc':>10}"
    print_flush(header)
    print_flush("-" * 100)

    scaling_data = []

    for algo_name, results in all_results.items():
        valid_results = [r for r in results if 'error' not in r]
        if not valid_results:
            continue

        # スケーリング則計算
        scaling = calculate_scaling_law(valid_results)

        # 最良結果（最大サンプル数）
        best_result = max(valid_results, key=lambda r: r['num_samples'])

        alpha_str = f"{scaling['alpha']:.5f}" if scaling['alpha'] else "N/A"
        A_str = f"{scaling['A']:.2e}" if scaling['A'] else "N/A"
        r_sq_str = f"{scaling['r_squared']:.4f}" if scaling['r_squared'] else "N/A"

        print_flush(
            f"{algo_name:<10} "
            f"{alpha_str:>10} "
            f"{A_str:>12} "
            f"{r_sq_str:>8} "
            f"{best_result['val_ppl']:>12.1f} "
            f"{best_result['val_acc']*100:>9.1f}%"
        )

        scaling_data.append({
            'algorithm': algo_name,
            'alpha': scaling['alpha'],
            'A': scaling['A'],
            'r_squared': scaling['r_squared'],
            'best_val_ppl': best_result['val_ppl'],
            'best_val_acc': best_result['val_acc'],
            'best_val_er_pct': best_result['best_val_er_pct'],
        })

    print_flush("=" * 100)

    # α値でソート（負の値が大きいほど良い）
    print_flush("\n--- Ranking by α (more negative = better scaling) ---")
    sorted_data = sorted(
        [d for d in scaling_data if d['alpha'] is not None],
        key=lambda x: x['alpha']
    )
    for i, d in enumerate(sorted_data, 1):
        print_flush(f"  {i}. {d['algorithm']}: α={d['alpha']:.5f}, PPL={d['best_val_ppl']:.1f}")

    # PPLでソート
    print_flush("\n--- Ranking by Val PPL (lower = better) ---")
    sorted_by_ppl = sorted(scaling_data, key=lambda x: x['best_val_ppl'])
    for i, d in enumerate(sorted_by_ppl, 1):
        print_flush(f"  {i}. {d['algorithm']}: PPL={d['best_val_ppl']:.1f}, Acc={d['best_val_acc']*100:.1f}%")

    # 保存
    with open(os.path.join(output_dir, 'scaling_analysis.json'), 'w') as f:
        json.dump(scaling_data, f, indent=2)

    return scaling_data


def save_all_results(all_results: Dict[str, List[Dict[str, Any]]], output_dir: str, config: ResidualConfig):
    """全結果を保存"""

    # 全データを統合
    all_data = []
    for algo_name, results in all_results.items():
        for r in results:
            all_data.append(r)

    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'context_dim': config.context_dim,
            'embed_dim': config.embed_dim,
            'num_layers': config.num_layers,
            'num_input_tokens': config.num_input_tokens,
            'dist_reg_weight': config.dist_reg_weight,
            'phase1_max_iterations': config.phase1_max_iterations,
            'phase2_epochs': config.phase2_epochs,
        },
        'algorithm_descriptions': ALGORITHM_DESCRIPTIONS,
        'results': all_data,
    }

    output_path = os.path.join(output_dir, 'all_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print_flush(f"\nAll results saved to: {output_path}")


# =============================================================================
# メイン
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Diversity Algorithm Full Experiment (Phase 1 + Phase 2 + α Analysis)'
    )
    parser.add_argument(
        '--algorithms', '-a',
        nargs='+',
        default=['MCDL', 'ODCM'],
        help='Algorithms to test (default: MCDL ODCM)'
    )
    parser.add_argument(
        '--samples', '-s',
        nargs='+',
        type=int,
        default=[50, 100, 200],
        help='Sample sizes to test (default: 50 100 200)'
    )
    parser.add_argument(
        '--context-dim', '-c',
        type=int,
        default=1000,
        help='Context dimension (default: 1000)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory (default: auto-generated)'
    )

    args = parser.parse_args()

    # 出力ディレクトリ
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"importants/logs/{timestamp}_diversity_full"
    os.makedirs(output_dir, exist_ok=True)

    # 設定
    config = ResidualConfig()
    config.context_dim = args.context_dim
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # 情報表示
    print_flush("=" * 70)
    print_flush("DIVERSITY ALGORITHM FULL EXPERIMENT")
    print_flush("(Phase 1 + Phase 2 + α Analysis)")
    print_flush("=" * 70)
    print_flush(f"Device: {device}")
    if device.type == "cuda":
        print_flush(f"GPU: {torch.cuda.get_device_name(0)}")
        print_flush(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    print_flush(f"\nAlgorithms: {args.algorithms}")
    print_flush(f"Sample sizes: {args.samples}")
    print_flush(f"Context dim: {args.context_dim}")
    print_flush(f"Output: {output_dir}")
    print_flush("\nConfig:")
    print_flush(f"  num_layers: {config.num_layers}")
    print_flush(f"  dist_reg_weight: {config.dist_reg_weight}")
    print_flush(f"  phase1_max_iterations: {config.phase1_max_iterations}")
    print_flush(f"  phase2_epochs: {config.phase2_epochs}")

    # 実験実行
    start_time = time.time()

    all_results = run_all_experiments(
        algorithms=args.algorithms,
        sample_sizes=args.samples,
        context_dim=args.context_dim,
        base_config=config,
        device=device,
        output_dir=output_dir
    )

    elapsed = time.time() - start_time

    # 結果表示
    print_results_table(all_results, args.context_dim)
    print_scaling_analysis(all_results, output_dir)

    # 結果保存
    save_all_results(all_results, output_dir, config)

    # メタデータ保存
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'algorithms': args.algorithms,
        'sample_sizes': args.samples,
        'context_dim': args.context_dim,
        'config': {
            'num_layers': config.num_layers,
            'embed_dim': config.embed_dim,
            'dist_reg_weight': config.dist_reg_weight,
            'phase1_max_iterations': config.phase1_max_iterations,
            'phase2_epochs': config.phase2_epochs,
        }
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    print_flush(f"\nTotal time: {elapsed/60:.1f} min")
    print_flush("\nExperiment completed!")


if __name__ == '__main__':
    main()
