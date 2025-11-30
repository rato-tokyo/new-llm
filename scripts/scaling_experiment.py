#!/usr/bin/env python3
"""
スケーリング則実験スクリプト（統合版）

任意のnum_input_tokens, num_layers, context_dimでスケーリング則を計算。
複数設定をまとめて実行し、関係式の導出に使用可能。

使用方法:
  # 単一設定
  python3 scripts/scaling_experiment.py --input-tokens 2 --layers 3 --context-dim 1536

  # 9設定マトリックス（関係式導出用）
  # context_dim=768固定、input_tokens=[1,2,3], layers=[1,2,3]
  python3 scripts/scaling_experiment.py --matrix

  # カスタムマトリックス（context_dim固定）
  python3 scripts/scaling_experiment.py --input-tokens 1 2 3 --layers 1 2 3 --context-dim 768

  # context_dimも変えたい場合（等倍: 768, 1536, 2304 推奨）
  python3 scripts/scaling_experiment.py --input-tokens 1 2 --layers 1 2 --context-dim 768 1536

  # サンプルサイズ指定
  python3 scripts/scaling_experiment.py --input-tokens 2 --layers 3 --context-dim 1536 --samples 50 100 200 500

Colab実行用:
  !cd /content/new-llm && python3 scripts/scaling_experiment.py --matrix

出力: results/scaling_YYYYMMDD_HHMMSS/
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from itertools import product

import numpy as np
from scipy import stats

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiments import ExperimentRunner, ExperimentConfig
from src.utils.io import print_flush


# デフォルト設定
DEFAULT_SAMPLE_SIZES = [50, 100, 200, 500]
DEFAULT_EMBED_DIM = 768
RANDOM_SEED = 42


def calculate_scaling_law(results: list):
    """スケーリング則を計算: PPL = A × tokens^α"""
    if len(results) < 2:
        return {'alpha': None, 'A': None, 'r_squared': None}

    tokens = np.array([r['train_tokens'] for r in results])
    ppl = np.array([r['val_ppl'] for r in results])

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


def run_single_config(runner, num_input_tokens, num_layers, context_dim,
                      sample_sizes, output_dir):
    """単一設定で複数サンプル数の実験を実行"""
    config_name = f"{num_layers}L_{context_dim}d_{num_input_tokens}tok"

    print_flush(f"\n{'='*70}")
    print_flush(f"Config: {config_name}")
    print_flush(f"  num_layers={num_layers}, context_dim={context_dim}, "
                f"num_input_tokens={num_input_tokens}")
    print_flush(f"{'='*70}")

    results = []

    for num_samples in sample_sizes:
        print_flush(f"\n  --- {num_samples} samples ---")

        exp_config = ExperimentConfig(
            num_samples=num_samples,
            num_layers=num_layers,
            context_dim=context_dim,
            embed_dim=DEFAULT_EMBED_DIM,
            num_input_tokens=num_input_tokens,
            random_seed=RANDOM_SEED,
        )

        try:
            result = runner.run(exp_config)
            results.append({
                'num_samples': num_samples,
                'train_tokens': result['train_tokens'],
                'val_ppl': result['val_ppl'],
                'val_acc': result['val_acc'],
                'val_er': result.get('val_er', 0),
            })
            print_flush(f"    → PPL: {result['val_ppl']:.1f}, "
                       f"Acc: {result['val_acc']:.1%}, "
                       f"ER: {result.get('val_er', 0):.1%}")
        except Exception as e:
            print_flush(f"    ✗ Error: {e}")
            continue

    # スケーリング則計算
    scaling = calculate_scaling_law(results)

    # 結果保存
    config_result = {
        'config_name': config_name,
        'num_input_tokens': num_input_tokens,
        'num_layers': num_layers,
        'context_dim': context_dim,
        'results': results,
        'scaling': scaling,
    }

    # JSONファイル保存
    config_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    with open(os.path.join(config_dir, 'results.json'), 'w') as f:
        json.dump(config_result, f, indent=2)

    return config_result


def print_summary(all_results, output_dir):
    """結果サマリーを表示・保存"""
    print_flush("\n" + "=" * 70)
    print_flush("RESULTS SUMMARY")
    print_flush("=" * 70)

    # ヘッダー
    header = f"{'Config':<25} {'α':>8} {'A':>12} {'R²':>8} {'Best PPL':>10} {'Best Acc':>10}"
    print_flush(header)
    print_flush("-" * 70)

    summary_data = []

    for result in all_results:
        config_name = result['config_name']
        scaling = result['scaling']

        if result['results']:
            best_ppl = min(r['val_ppl'] for r in result['results'])
            best_acc = max(r['val_acc'] for r in result['results'])
        else:
            best_ppl = float('inf')
            best_acc = 0

        alpha = scaling.get('alpha')
        A = scaling.get('A')
        r_sq = scaling.get('r_squared')

        alpha_str = f"{alpha:.4f}" if alpha else "N/A"
        A_str = f"{A:.2e}" if A else "N/A"
        r_sq_str = f"{r_sq:.4f}" if r_sq else "N/A"

        print_flush(f"{config_name:<25} {alpha_str:>8} {A_str:>12} {r_sq_str:>8} "
                   f"{best_ppl:>10.1f} {best_acc:>9.1%}")

        summary_data.append({
            'config_name': config_name,
            'num_input_tokens': result['num_input_tokens'],
            'num_layers': result['num_layers'],
            'context_dim': result['context_dim'],
            'alpha': alpha,
            'A': A,
            'r_squared': r_sq,
            'best_ppl': best_ppl,
            'best_acc': best_acc,
        })

    # サマリーJSON保存
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)

    print_flush(f"\nResults saved to: {output_dir}")

    return summary_data


def main():
    parser = argparse.ArgumentParser(
        description='スケーリング則実験スクリプト',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 単一設定
  python3 scripts/scaling_experiment.py --input-tokens 2 --layers 3 --context-dim 1536

  # 9設定マトリックス（context_dim=768固定）
  python3 scripts/scaling_experiment.py --matrix

  # カスタムマトリックス（context_dim固定）
  python3 scripts/scaling_experiment.py --input-tokens 1 2 3 --layers 1 2 3 --context-dim 768
        """
    )

    parser.add_argument('--input-tokens', type=int, nargs='+', default=[2],
                        help='num_input_tokens の値（複数指定可）')
    parser.add_argument('--layers', type=int, nargs='+', default=[3],
                        help='num_layers の値（複数指定可）')
    parser.add_argument('--context-dim', type=int, nargs='+', default=[768],
                        help='context_dim の値（複数指定可、デフォルト: 768）')
    parser.add_argument('--samples', type=int, nargs='+', default=DEFAULT_SAMPLE_SIZES,
                        help=f'サンプルサイズ（デフォルト: {DEFAULT_SAMPLE_SIZES}）')
    parser.add_argument('--matrix', action='store_true',
                        help='9設定マトリックス実行（context_dim=768固定、input_tokens=[1,2,3], layers=[1,2,3]）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='出力ディレクトリ（デフォルト: results/scaling_YYYYMMDD_HHMMSS）')

    args = parser.parse_args()

    # マトリックスモード: context_dim=768固定、input_tokens×layers
    if args.matrix:
        input_tokens_list = [1, 2, 3]
        layers_list = [1, 2, 3]
        context_dim_list = [768]  # 固定
    else:
        input_tokens_list = args.input_tokens
        layers_list = args.layers
        context_dim_list = args.context_dim

    # 出力ディレクトリ
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/scaling_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 実験設定の生成
    configs = []
    for input_tokens, layers, context_dim in product(input_tokens_list, layers_list, context_dim_list):
        configs.append((input_tokens, layers, context_dim))

    # 実験情報表示
    print_flush("=" * 70)
    print_flush("SCALING LAW EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Configurations: {len(configs)}")
    print_flush(f"Sample sizes: {args.samples}")
    print_flush(f"Total experiments: {len(configs) * len(args.samples)}")
    print_flush(f"Output: {output_dir}")

    # GPU情報
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print_flush(f"Device: cuda ({gpu_name}, {gpu_mem:.1f}GB)")
        else:
            print_flush("Device: cpu")
    except:
        print_flush("Device: unknown")

    # 設定一覧
    print_flush("\nConfigurations:")
    for i, (input_tokens, layers, context_dim) in enumerate(configs):
        print_flush(f"  {i+1}. {layers}L_{context_dim}d_{input_tokens}tok")

    # 実験実行
    start_time = time.time()
    runner = ExperimentRunner()

    all_results = []
    for input_tokens, layers, context_dim in configs:
        result = run_single_config(
            runner=runner,
            num_input_tokens=input_tokens,
            num_layers=layers,
            context_dim=context_dim,
            sample_sizes=args.samples,
            output_dir=output_dir,
        )
        all_results.append(result)

    # サマリー表示
    summary = print_summary(all_results, output_dir)

    # 実行時間
    elapsed = time.time() - start_time
    print_flush(f"\nTotal time: {elapsed/60:.1f} min")

    # 実験メタデータ保存
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'sample_sizes': args.samples,
        'configs': [
            {'input_tokens': it, 'layers': l, 'context_dim': cd}
            for it, l, cd in configs
        ],
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()
