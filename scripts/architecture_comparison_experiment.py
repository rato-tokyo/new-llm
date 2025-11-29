#!/usr/bin/env python3
"""
アーキテクチャ比較実験スクリプト（スケーリング則計算付き）

4つの設定を複数サンプル数で比較し、α値を算出:
1. Baseline: 6層、768次元、1トークン入力
2. Exp1: 6層、768次元、2トークン入力 (num_input_tokens=2)
3. Exp2: 6層、1152次元、1トークン入力 (context_dim*1.5)
4. Exp3: 9層、768次元、1トークン入力 (num_layers=9)

スケーリング則: PPL = A × tokens^α
- α値が小さい（より負）ほど、データ効率が良い

Colab実行用:
    !cd /content/new-llm && python3 scripts/architecture_comparison_experiment.py

出力: results/architecture_comparison_YYYYMMDD_HHMMSS/
"""

import os
import sys
import json
import time
from datetime import datetime

import numpy as np
from scipy import stats

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments import ExperimentRunner, ExperimentConfig

# 実験設定
SAMPLE_SIZES = [50, 100, 200, 500]  # スケーリング則計算用
RANDOM_SEED = 42


def print_flush(msg):
    print(msg, flush=True)


def calculate_scaling_law(results: list):
    """スケーリング則を計算: PPL = A × tokens^α"""
    if len(results) < 2:
        return {'alpha': None, 'A': None, 'r_squared': None, 'p_value': None, 'std_err': None}

    tokens = np.array([r['train_tokens'] for r in results])
    ppl = np.array([r['val_ppl'] for r in results])

    # 対数変換
    log_tokens = np.log(tokens)
    log_ppl = np.log(ppl)

    # 線形回帰
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_tokens, log_ppl)

    alpha = slope  # 負の値が期待される
    A = np.exp(intercept)
    r_squared = r_value ** 2

    return {
        'alpha': alpha,
        'A': A,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_err': std_err,
    }


def run_architecture_experiment(runner, arch_config, output_dir):
    """単一アーキテクチャで複数サンプル数の実験を実行"""
    name = arch_config['name']
    print_flush(f"\n{'='*70}")
    print_flush(f"Architecture: {name}")
    print_flush(f"  num_layers={arch_config['num_layers']}, "
               f"context_dim={arch_config['context_dim']}, "
               f"num_input_tokens={arch_config['num_input_tokens']}")
    print_flush(f"{'='*70}")

    # パラメータ数計算
    sample_config = ExperimentConfig(
        num_layers=arch_config['num_layers'],
        context_dim=arch_config['context_dim'],
        embed_dim=arch_config['embed_dim'],
        num_input_tokens=arch_config['num_input_tokens'],
    )
    params = runner.calculate_params(sample_config)
    print_flush(f"  Phase1 params: {params['trainable_phase1']/1e6:.2f}M")
    print_flush(f"  Phase2 params: {params['trainable_phase2']/1e6:.2f}M")

    results = []
    start_time = time.time()

    for num_samples in SAMPLE_SIZES:
        print_flush(f"\n  --- {num_samples} samples ---")

        config = ExperimentConfig(
            num_layers=arch_config['num_layers'],
            context_dim=arch_config['context_dim'],
            embed_dim=arch_config['embed_dim'],
            num_input_tokens=arch_config['num_input_tokens'],
            num_samples=num_samples,
            random_seed=RANDOM_SEED,
            verbose=True,
        )

        result = runner.run(config)
        results.append(result)

        print_flush(f"    → PPL: {result['val_ppl']:.1f}, "
                   f"Acc: {result['val_acc']*100:.1f}%, "
                   f"ER: {result['val_effective_rank']*100:.1f}%")

    total_time = time.time() - start_time

    # スケーリング則計算
    scaling = calculate_scaling_law(results)

    print_flush(f"\n  Scaling Law: α = {scaling['alpha']:.4f} (R² = {scaling['r_squared']:.4f})")

    # 結果をまとめる
    arch_result = {
        'config_name': name,
        'num_layers': arch_config['num_layers'],
        'context_dim': arch_config['context_dim'],
        'embed_dim': arch_config['embed_dim'],
        'num_input_tokens': arch_config['num_input_tokens'],
        'params': params,
        'total_time': total_time,
        'sample_results': results,
        'scaling_law': scaling,
    }

    # 個別結果を保存
    result_file = os.path.join(output_dir, f'{name}.json')
    with open(result_file, 'w') as f:
        json.dump(arch_result, f, indent=2)

    return arch_result


def main():
    print_flush("="*70)
    print_flush("Architecture Comparison Experiment (Scaling Law)")
    print_flush("="*70)
    print_flush(f"Sample sizes: {SAMPLE_SIZES}")

    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/architecture_comparison_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print_flush(f"Output: {output_dir}")

    # Runner作成
    runner = ExperimentRunner()

    # 実験設定
    experiments = [
        {
            'name': 'baseline',
            'num_layers': 6,
            'context_dim': 768,
            'embed_dim': 768,
            'num_input_tokens': 1,
        },
        {
            'name': 'input_tokens_2',
            'num_layers': 6,
            'context_dim': 768,
            'embed_dim': 768,
            'num_input_tokens': 2,
        },
        {
            'name': 'context_dim_1152',
            'num_layers': 6,
            'context_dim': 1152,  # 768 * 1.5
            'embed_dim': 768,
            'num_input_tokens': 1,
        },
        {
            'name': 'layers_9',
            'num_layers': 9,
            'context_dim': 768,
            'embed_dim': 768,
            'num_input_tokens': 1,
        },
    ]

    # 実験実行
    all_results = []
    total_start = time.time()

    for exp in experiments:
        result = run_architecture_experiment(runner, exp, output_dir)
        all_results.append(result)

    total_time = time.time() - total_start

    # サマリー出力
    print_flush("\n" + "="*70)
    print_flush("SUMMARY: Scaling Law Comparison")
    print_flush("="*70)

    print_flush(f"\n{'Config':<20} {'Phase1 Params':>14} {'α (slope)':>12} {'R²':>8} {'Best PPL':>10}")
    print_flush("-"*70)

    for r in all_results:
        # 500サンプルでの最良PPL
        best_ppl = r['sample_results'][-1]['val_ppl'] if r['sample_results'] else 0
        alpha = r['scaling_law']['alpha']
        r2 = r['scaling_law']['r_squared']
        print_flush(
            f"{r['config_name']:<20} "
            f"{r['params']['trainable_phase1']/1e6:>12.2f}M "
            f"{alpha:>12.4f} "
            f"{r2:>8.4f} "
            f"{best_ppl:>10.1f}"
        )

    print_flush("-"*70)
    print_flush(f"\nTotal time: {total_time/60:.1f} min")

    # α値の解釈
    print_flush("\n" + "="*70)
    print_flush("INTERPRETATION")
    print_flush("="*70)
    print_flush("Scaling Law: PPL = A × tokens^α")
    print_flush("  - より負のα → データ効率が良い（少ないデータでPPL低下）")
    print_flush("  - R² > 0.9 → スケーリング則への適合度が高い")

    # ランキング
    sorted_results = sorted(all_results, key=lambda x: x['scaling_law']['alpha'] or 0)
    print_flush("\nRanking by α (better = more negative):")
    for i, r in enumerate(sorted_results, 1):
        alpha = r['scaling_law']['alpha']
        if alpha is not None:
            print_flush(f"  {i}. {r['config_name']}: α = {alpha:.4f}")

    # 全結果を保存
    summary = {
        'timestamp': timestamp,
        'sample_sizes': SAMPLE_SIZES,
        'total_time_sec': total_time,
        'results': all_results
    }

    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print_flush(f"\nSaved: {summary_file}")


if __name__ == '__main__':
    main()
