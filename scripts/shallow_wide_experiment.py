#!/usr/bin/env python3
"""
浅く広いアーキテクチャ実験スクリプト

設定:
- 3層、768×2=1536次元、2トークン入力
- レイヤー数を半分に減らし、context_dimとinput_tokensを増やす

仮説:
- CVFPは深さより入力の豊かさが重要
- 3層でも十分に収束可能
- パラメータ効率が向上する可能性

Colab実行用:
    !cd /content/new-llm && python3 scripts/shallow_wide_experiment.py

出力: results/shallow_wide_YYYYMMDD_HHMMSS/
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


def main():
    print_flush("=" * 70)
    print_flush("Shallow & Wide Architecture Experiment")
    print_flush("=" * 70)
    print_flush("Config: 3 layers, context_dim=1536, num_input_tokens=2")
    print_flush(f"Sample sizes: {SAMPLE_SIZES}")

    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/shallow_wide_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print_flush(f"Output: {output_dir}")

    # Runner作成
    runner = ExperimentRunner()

    # 実験設定: 3層、1536次元、2トークン入力
    arch_config = {
        'name': 'shallow_wide_3L_1536d_2tok',
        'num_layers': 3,
        'context_dim': 1536,  # 768 * 2
        'embed_dim': 768,
        'num_input_tokens': 2,
    }

    print_flush(f"\n{'=' * 70}")
    print_flush(f"Architecture: {arch_config['name']}")
    print_flush(f"  num_layers={arch_config['num_layers']}, "
                f"context_dim={arch_config['context_dim']}, "
                f"num_input_tokens={arch_config['num_input_tokens']}")
    print_flush(f"{'=' * 70}")

    # パラメータ数計算
    sample_config = ExperimentConfig(
        num_layers=arch_config['num_layers'],
        context_dim=arch_config['context_dim'],
        embed_dim=arch_config['embed_dim'],
        num_input_tokens=arch_config['num_input_tokens'],
    )
    params = runner.calculate_params(sample_config)
    print_flush(f"  Phase1 params: {params['trainable_phase1'] / 1e6:.2f}M")
    print_flush(f"  Phase2 params: {params['trainable_phase2'] / 1e6:.2f}M")

    results = []
    total_start = time.time()

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
                    f"Acc: {result['val_acc'] * 100:.1f}%, "
                    f"ER: {result['val_effective_rank'] * 100:.1f}%")

        # 途中結果を保存（クラッシュ対策）
        partial_file = os.path.join(output_dir, 'partial_results.json')
        with open(partial_file, 'w') as f:
            json.dump(results, f, indent=2)

    total_time = time.time() - total_start

    # スケーリング則計算
    scaling = calculate_scaling_law(results)

    # サマリー出力
    print_flush("\n" + "=" * 70)
    print_flush("RESULTS")
    print_flush("=" * 70)
    print_flush(f"{'Samples':>8} {'Tokens':>10} {'Val PPL':>10} {'Val Acc':>8} {'Val ER':>8}")
    print_flush("-" * 50)
    for r in results:
        print_flush(
            f"{r['num_samples']:>8} {r['train_tokens']:>10,} "
            f"{r['val_ppl']:>10.1f} {r['val_acc'] * 100:>7.1f}% "
            f"{r['val_effective_rank'] * 100:>7.1f}%"
        )

    print_flush(f"\nScaling Law: α = {scaling['alpha']:.4f} (R² = {scaling['r_squared']:.4f})")
    print_flush(f"Total time: {total_time / 60:.1f} min")

    # 比較用ベースライン（前回実験結果）
    print_flush("\n" + "=" * 70)
    print_flush("COMPARISON WITH PREVIOUS RESULTS")
    print_flush("=" * 70)
    print_flush(f"{'Config':<30} {'α':>10} {'Best PPL':>10} {'Best Acc':>10}")
    print_flush("-" * 70)
    print_flush(f"{'baseline (6L/768d/1tok)':<30} {-0.4860:>10.4f} {249.3:>10.1f} {21.3:>9.1f}%")
    print_flush(f"{'input_tokens_2 (6L/768d/2tok)':<30} {-0.4702:>10.4f} {198.1:>10.1f} {22.5:>9.1f}%")
    print_flush(f"{'context_dim_1152 (6L/1152d/1tok)':<30} {-0.4988:>10.4f} {246.9:>10.1f} {21.4:>9.1f}%")
    print_flush(f"{'layers_9 (9L/768d/1tok)':<30} {-0.4818:>10.4f} {256.8:>10.1f} {21.1:>9.1f}%")
    print_flush("-" * 70)
    best_ppl = results[-1]['val_ppl'] if results else 0
    best_acc = results[-1]['val_acc'] * 100 if results else 0
    print_flush(f"{'shallow_wide (3L/1536d/2tok)':<30} {scaling['alpha']:>10.4f} {best_ppl:>10.1f} {best_acc:>9.1f}%")

    # 結果保存
    output = {
        'config': arch_config,
        'params': params,
        'sample_sizes': SAMPLE_SIZES,
        'results': results,
        'scaling_law': scaling,
        'total_time_minutes': total_time / 60,
        'timestamp': timestamp,
    }

    output_file = os.path.join(output_dir, 'results.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print_flush(f"\nSaved: {output_file}")


if __name__ == '__main__':
    main()
