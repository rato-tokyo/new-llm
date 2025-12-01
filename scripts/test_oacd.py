#!/usr/bin/env python3
"""
OACD (Origin-Anchored Centroid Dispersion) アルゴリズムのテストスクリプト

dwr=1.0（CVFP完全無効）でOACDの収束特性をテストする。
MCDLとの比較も実施。

使用方法:
  # ローカル（CPU）: 2サンプルで動作確認
  python3 scripts/test_oacd.py

  # Colab（GPU）: 100-200サンプルで本格実験
  python3 scripts/test_oacd.py -s 100 200

  # MCDLとの比較
  python3 scripts/test_oacd.py -a MCDL OACD -s 100 200
"""

import os
import sys
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
    context_dim: int = 500,
    dist_reg_weight: float = 1.0,  # デフォルトでCVFP無効
) -> Dict[str, Any]:
    """単一の実験を実行（Phase 1 + Phase 2）"""

    set_seed(seed)

    # データ読み込み用設定
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

    # Phase 1用設定
    phase1_config = Phase1Config.from_base(
        base_config, device,
        context_dim=context_dim,
        dist_reg_weight=dist_reg_weight,
    )

    # Phase 1 トレーナー作成
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
    convergence_rate = phase1_stats.get('convergence_rate', 0.0)

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
                f"ER={train_er_pct:.1f}%/{best_val_er_pct:.1f}%, Conv={convergence_rate*100:.1f}%")

    # Phase 2用設定
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
        'dist_reg_weight': dist_reg_weight,
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
        'convergence_rate': convergence_rate,
        'phase2_time': phase2_time,
        'best_epoch': best_epoch,
        'train_ppl': best_train_ppl,
        'val_ppl': best_ppl,
        'val_acc': best_acc,
        'total_time': total_time,
    }


# =============================================================================
# メイン
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Test OACD algorithm with dwr=1.0 (CVFP disabled)'
    )
    parser.add_argument(
        '--algorithms', '-a',
        nargs='+',
        default=['OACD'],
        help='Algorithms to test (default: OACD)'
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
        default=500,
        help='Context dimension (default: 500)'
    )
    parser.add_argument(
        '--dist-reg-weight', '-d',
        type=float,
        default=1.0,
        help='Diversity regularization weight (default: 1.0 = CVFP disabled)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory (default: auto-generated)'
    )

    args = parser.parse_args()

    # 設定
    config = ResidualConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # 出力ディレクトリ
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"importants/logs/{timestamp}_oacd_test"

    os.makedirs(output_dir, exist_ok=True)

    # 情報表示
    print_flush("=" * 70)
    print_flush("OACD ALGORITHM TEST")
    print_flush("=" * 70)
    print_flush(f"Device: {device}")
    if device.type == "cuda":
        print_flush(f"GPU: {torch.cuda.get_device_name(0)}")
        print_flush(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    print_flush(f"\nAlgorithms: {args.algorithms}")
    print_flush(f"Sample sizes: {args.samples}")
    print_flush(f"Context dim: {args.context_dim}")
    print_flush(f"dist_reg_weight: {args.dist_reg_weight}")
    print_flush(f"Output: {output_dir}")
    print_flush("=" * 70)

    # 実験実行
    all_results = {}

    for algo_name in args.algorithms:
        if algo_name not in DIVERSITY_ALGORITHMS:
            print_flush(f"Warning: Unknown algorithm '{algo_name}', skipping")
            continue

        diversity_fn = DIVERSITY_ALGORITHMS[algo_name]
        algo_desc = ALGORITHM_DESCRIPTIONS.get(algo_name, algo_name)

        print_flush(f"\n{'='*70}")
        print_flush(f"Algorithm: {algo_name} - {algo_desc}")
        print_flush(f"{'='*70}")

        algo_results = []

        for num_samples in args.samples:
            print_flush(f"\n  [{algo_name}] Samples: {num_samples}")

            try:
                result = run_single_experiment(
                    algorithm_name=algo_name,
                    diversity_fn=diversity_fn,
                    num_samples=num_samples,
                    base_config=config,
                    device=device,
                    seed=42,
                    context_dim=args.context_dim,
                    dist_reg_weight=args.dist_reg_weight,
                )
                algo_results.append(result)
            except Exception as e:
                print_flush(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        if algo_results:
            # スケーリング則計算
            scaling = calculate_scaling_law(algo_results)
            all_results[algo_name] = {
                'experiments': algo_results,
                'scaling': scaling,
            }

            print_flush(f"\n  [{algo_name}] Scaling Law:")
            if scaling['alpha'] is not None:
                print_flush(f"    α = {scaling['alpha']:.4f}")
                print_flush(f"    A = {scaling['A']:.2e}")
                print_flush(f"    R² = {scaling['r_squared']:.4f}")
            else:
                print_flush("    Could not calculate (insufficient data)")

    # サマリー表示
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    print_flush(f"\n{'Algorithm':<10} {'Samples':<10} {'Tokens':<12} {'Val PPL':<10} "
                f"{'Acc':<8} {'ER%':<8} {'Conv%':<8} {'Iter':<6}")
    print_flush("-" * 80)

    for algo_name, data in all_results.items():
        for r in data['experiments']:
            print_flush(f"{r['algorithm']:<10} {r['num_samples']:<10} {r['train_tokens']:<12,} "
                       f"{r['val_ppl']:<10.1f} {r['val_acc']*100:<8.1f} "
                       f"{r['val_er_pct']:<8.1f} {r['convergence_rate']*100:<8.1f} "
                       f"{r['phase1_iterations']:<6}")

    print_flush("\n" + "-" * 80)
    print_flush(f"\n{'Algorithm':<10} {'α':<10} {'A':<12} {'R²':<10}")
    print_flush("-" * 50)

    for algo_name, data in all_results.items():
        s = data['scaling']
        if s['alpha'] is not None:
            print_flush(f"{algo_name:<10} {s['alpha']:<10.4f} {s['A']:<12.2e} {s['r_squared']:<10.4f}")
        else:
            print_flush(f"{algo_name:<10} {'N/A':<10} {'N/A':<12} {'N/A':<10}")

    print_flush("\n" + "=" * 70)
    print_flush("DONE")
    print_flush("=" * 70)


if __name__ == '__main__':
    main()
