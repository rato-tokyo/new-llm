#!/usr/bin/env python3
"""
統一設定でのスケーリング実験スクリプト

目的:
- 統一した設定（truncation=False）で複数サンプル数での実験を実施
- 正確なα値を導出

使用方法:
  Colab: !python scripts/unified_scaling_experiment.py
  Local: python3 scripts/unified_scaling_experiment.py

設定:
- トークン化: truncation=False（全長使用）
- サンプル数: [50, 100, 200, 500]
- num_input_tokens: 1
- モデル: 6層/768dim
"""

import sys
import os
import json
import time
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy import stats

from src.experiments import ExperimentRunner, ExperimentConfig

# 設定（実験用）
SAMPLE_SIZES = [50, 100, 200, 500]  # 全実験
NUM_LAYERS = 6
CONTEXT_DIM = 768
EMBED_DIM = 768
NUM_INPUT_TOKENS = 1
DIST_REG_WEIGHT = 0.8  # 多様性正則化の重み
RANDOM_SEED = 42


def print_flush(msg):
    print(msg, flush=True)


def calculate_scaling_law(results: list):
    """スケーリング則を計算: PPL = A × tokens^α"""
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


def main():
    print_flush("=" * 60)
    print_flush("SCALING EXPERIMENT")
    print_flush("=" * 60)
    print_flush(f"Samples: {SAMPLE_SIZES} | Model: {NUM_LAYERS}L/{CONTEXT_DIM}D | Seed: {RANDOM_SEED}")

    # Runner作成
    runner = ExperimentRunner()

    results = []
    total_start = time.time()

    # 結果保存ディレクトリ
    output_dir = './results/unified_scaling'
    os.makedirs(output_dir, exist_ok=True)

    for num_samples in SAMPLE_SIZES:
        config = ExperimentConfig(
            num_layers=NUM_LAYERS,
            context_dim=CONTEXT_DIM,
            embed_dim=EMBED_DIM,
            num_input_tokens=NUM_INPUT_TOKENS,
            num_samples=num_samples,
            dist_reg_weight=DIST_REG_WEIGHT,
            random_seed=RANDOM_SEED,
            verbose=True,
        )

        result = runner.run(config)
        results.append(result)

        # 途中結果をJSON保存（クラッシュ対策）
        partial_file = os.path.join(output_dir, 'partial_results.json')
        with open(partial_file, 'w') as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - total_start

    scaling = calculate_scaling_law(results)

    # 結果サマリー
    print_flush("\n" + "=" * 60)
    print_flush("RESULTS")
    print_flush("=" * 60)
    print_flush(f"{'Samples':>8} {'Tokens':>10} {'Val PPL':>10} {'Val Acc':>8} {'Val ER':>8}")
    print_flush("-" * 50)
    for r in results:
        print_flush(
            f"{r['num_samples']:>8} {r['train_tokens']:>10,} "
            f"{r['val_ppl']:>10.1f} {r['val_acc']*100:>7.1f}% "
            f"{r['val_effective_rank']*100:>7.1f}%"
        )

    print_flush(f"\nScaling: α={scaling['alpha']:.3f} (R²={scaling['r_squared']:.3f})")
    print_flush(f"Total: {total_time/60:.1f} min")

    # 結果保存
    output = {
        'settings': {
            'sample_sizes': SAMPLE_SIZES,
            'num_layers': NUM_LAYERS,
            'context_dim': CONTEXT_DIM,
            'embed_dim': EMBED_DIM,
            'num_input_tokens': NUM_INPUT_TOKENS,
            'dist_reg_weight': DIST_REG_WEIGHT,
            'tokenization': 'truncation=False (full length)',
            'random_seed': RANDOM_SEED,
        },
        'results': results,
        'scaling_law': scaling,
        'total_time_minutes': total_time / 60,
        'timestamp': datetime.now().isoformat(),
    }

    output_file = os.path.join(output_dir, 'results.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print_flush(f"Saved: {output_file}")


if __name__ == '__main__':
    main()
