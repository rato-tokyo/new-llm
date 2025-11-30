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

  # α値のデータ量依存性測定（スライディングウィンドウ方式）
  # 初期サンプル50, 倍率2, ウィンドウ4点, ステップ1で5サンプル生成 [50,100,200,400,800]
  # 採取標本1: [50,100,200,400] → α₁
  # 採取標本2: [100,200,400,800] → α₂
  python3 scripts/scaling_experiment.py --alpha-scaling --init-samples 50 --multiplier 2 --window-size 4 --num-windows 2

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


def generate_sample_sizes(init_samples: int, multiplier: float,
                          window_size: int, num_windows: int) -> list[int]:
    """サンプルサイズのリストを生成

    Args:
        init_samples: 初期サンプル数（例: 50）
        multiplier: 倍率（例: 2 → 50, 100, 200, ...）
        window_size: 1つのウィンドウに含むサンプル数（例: 4）
        num_windows: ウィンドウ数（例: 2）

    Returns:
        サンプルサイズのリスト

    Example:
        generate_sample_sizes(50, 2, 4, 2)
        → [50, 100, 200, 400, 800]  (5点 = 4 + 2 - 1)
        Window 1: [50, 100, 200, 400]
        Window 2: [100, 200, 400, 800]
    """
    # 必要な点数: window_size + (num_windows - 1)
    total_points = window_size + (num_windows - 1)

    samples = []
    current: float = float(init_samples)
    for _ in range(total_points):
        samples.append(int(current))
        current *= multiplier

    return samples


def calculate_sliding_window_alphas(results: list, window_size: int) -> list[dict]:
    """スライディングウィンドウでα値を計算

    Args:
        results: 実験結果リスト（num_samplesでソート済み）
        window_size: ウィンドウサイズ

    Returns:
        各ウィンドウのスケーリング情報リスト
    """
    if len(results) < window_size:
        return []

    window_results = []
    num_windows = len(results) - window_size + 1

    for i in range(num_windows):
        window_data = results[i:i + window_size]
        scaling = calculate_scaling_law(window_data)

        # ウィンドウの範囲情報
        min_samples = window_data[0]['num_samples']
        max_samples = window_data[-1]['num_samples']
        min_tokens = window_data[0]['train_tokens']
        max_tokens = window_data[-1]['train_tokens']

        window_results.append({
            'window_index': i + 1,
            'sample_range': [min_samples, max_samples],
            'token_range': [min_tokens, max_tokens],
            'samples_in_window': [r['num_samples'] for r in window_data],
            **scaling,
        })

    return window_results


def print_alpha_progression(all_results: list, window_size: int, output_dir: str):
    """α値の推移を表示・保存"""
    print_flush("\n" + "=" * 100)
    print_flush("ALPHA PROGRESSION ANALYSIS (Sliding Window)")
    print_flush(f"Window size: {window_size} points")
    print_flush("=" * 100)

    all_progressions = []

    for result in all_results:
        config_name = result['config_name']
        results_sorted = sorted(result['results'], key=lambda r: r['num_samples'])

        window_results = calculate_sliding_window_alphas(results_sorted, window_size)

        if not window_results:
            continue

        print_flush(f"\n{config_name}:")
        print_flush("-" * 80)
        header = f"{'Window':>8} {'Samples':>20} {'Tokens':>25} {'α':>10} {'A':>12} {'R²':>8}"
        print_flush(header)
        print_flush("-" * 80)

        for w in window_results:
            samples_str = f"{w['sample_range'][0]}-{w['sample_range'][1]}"
            tokens_str = f"{w['token_range'][0]:,}-{w['token_range'][1]:,}"
            alpha_str = f"{w['alpha']:.5f}" if w['alpha'] else "N/A"
            A_str = f"{w['A']:.2e}" if w['A'] else "N/A"
            r_sq_str = f"{w['r_squared']:.4f}" if w['r_squared'] else "N/A"

            print_flush(f"{w['window_index']:>8} {samples_str:>20} {tokens_str:>25} "
                       f"{alpha_str:>10} {A_str:>12} {r_sq_str:>8}")

        # α値の変化を計算
        if len(window_results) >= 2:
            alphas = [w['alpha'] for w in window_results if w['alpha'] is not None]
            if len(alphas) >= 2:
                alpha_change = alphas[-1] - alphas[0]
                alpha_change_pct = (alphas[-1] - alphas[0]) / abs(alphas[0]) * 100
                print_flush(f"\n  α change: {alphas[0]:.5f} → {alphas[-1]:.5f} "
                           f"(Δ = {alpha_change:+.5f}, {alpha_change_pct:+.2f}%)")

                # 傾向判定
                if alpha_change < -0.01:
                    trend = "↓ IMPROVING (α becoming more negative = better scaling)"
                elif alpha_change > 0.01:
                    trend = "↑ DEGRADING (α becoming less negative = worse scaling)"
                else:
                    trend = "→ STABLE"
                print_flush(f"  Trend: {trend}")

        all_progressions.append({
            'config_name': config_name,
            'num_input_tokens': result['num_input_tokens'],
            'num_layers': result['num_layers'],
            'context_dim': result['context_dim'],
            'window_size': window_size,
            'windows': window_results,
        })

    # JSON保存
    with open(os.path.join(output_dir, 'alpha_progression.json'), 'w') as f:
        json.dump(all_progressions, f, indent=2)

    print_flush(f"\nAlpha progression saved to: {output_dir}/alpha_progression.json")

    return all_progressions


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
                'phase1_iter': result.get('phase1_iterations', 0),
                'train_er': result.get('train_effective_rank', 0),
                'val_er': result.get('val_effective_rank', 0),
                'train_ppl': result.get('train_ppl', 0),
                'val_ppl': result['val_ppl'],
                'val_acc': result['val_acc'],
            })
            print_flush(f"    → PPL: {result['val_ppl']:.1f}, "
                       f"Acc: {result['val_acc']:.1%}, "
                       f"ER: {result.get('val_effective_rank', 0):.1%}")
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
    print_flush("\n" + "=" * 100)
    print_flush("RESULTS SUMMARY")
    print_flush("=" * 100)

    # ヘッダー
    header = f"{'Config':<18} {'α':>7} {'A':>10} {'R²':>6} {'PPL':>7} {'Acc':>6} {'T.PPL':>7} {'ER':>5} {'Iter':>4}"
    print_flush(header)
    print_flush("-" * 100)

    summary_data = []

    for result in all_results:
        config_name = result['config_name']
        scaling = result['scaling']

        if result['results']:
            # 最良サンプル（500サンプル）のデータを取得
            best_result = max(result['results'], key=lambda r: r['num_samples'])
            best_ppl = best_result['val_ppl']
            best_acc = best_result['val_acc']
            best_train_ppl = best_result.get('train_ppl', 0)
            best_val_er = best_result.get('val_er', 0)
            best_iter = best_result.get('phase1_iter', 0)
        else:
            best_ppl = float('inf')
            best_acc = 0
            best_train_ppl = 0
            best_val_er = 0
            best_iter = 0

        alpha = scaling.get('alpha')
        A = scaling.get('A')
        r_sq = scaling.get('r_squared')

        alpha_str = f"{alpha:.4f}" if alpha else "N/A"
        A_str = f"{A:.2e}" if A else "N/A"
        r_sq_str = f"{r_sq:.3f}" if r_sq else "N/A"
        train_ppl_str = f"{best_train_ppl:.1f}" if best_train_ppl else "N/A"
        val_er_str = f"{best_val_er*100:.1f}%" if best_val_er else "N/A"
        iter_str = f"{best_iter}" if best_iter else "N/A"

        print_flush(f"{config_name:<18} {alpha_str:>7} {A_str:>10} {r_sq_str:>6} "
                   f"{best_ppl:>7.1f} {best_acc*100:>5.1f}% {train_ppl_str:>7} {val_er_str:>5} {iter_str:>4}")

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
            'best_train_ppl': best_train_ppl,
            'best_val_er': best_val_er,
            'best_iter': best_iter,
        })

    # 詳細テーブル（全サンプル数）
    print_flush("\n" + "=" * 100)
    print_flush("DETAILED RESULTS (All sample sizes)")
    print_flush("=" * 100)
    header2 = f"{'Config':<18} {'Samples':>7} {'Tokens':>10} {'P1 Iter':>7} {'Train ER':>8} {'Val ER':>7} {'T.PPL':>7} {'V.PPL':>7} {'Acc':>6}"
    print_flush(header2)
    print_flush("-" * 100)

    for result in all_results:
        config_name = result['config_name']
        for r in result['results']:
            train_er_str = f"{r.get('train_er', 0)*100:.1f}%" if r.get('train_er') else "N/A"
            val_er_str = f"{r.get('val_er', 0)*100:.1f}%" if r.get('val_er') else "N/A"
            train_ppl_str = f"{r.get('train_ppl', 0):.1f}" if r.get('train_ppl') else "N/A"
            iter_str = f"{r.get('phase1_iter', 0)}" if r.get('phase1_iter') else "N/A"
            print_flush(f"{config_name:<18} {r['num_samples']:>7} {r['train_tokens']:>10,} {iter_str:>7} "
                       f"{train_er_str:>8} {val_er_str:>7} {train_ppl_str:>7} {r['val_ppl']:>7.1f} {r['val_acc']*100:>5.1f}%")

    # サマリーJSON保存
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)

    # 詳細結果もJSON保存
    detailed_data = []
    for result in all_results:
        for r in result['results']:
            detailed_data.append({
                'config_name': result['config_name'],
                'num_layers': result['num_layers'],
                'context_dim': result['context_dim'],
                'num_input_tokens': result['num_input_tokens'],
                **r
            })
    with open(os.path.join(output_dir, 'detailed_results.json'), 'w') as f:
        json.dump(detailed_data, f, indent=2)

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

  # α値のデータ量依存性測定
  python3 scripts/scaling_experiment.py --alpha-scaling --init-samples 50 --multiplier 2 --window-size 4 --num-windows 2
        """
    )

    parser.add_argument('--input-tokens', type=int, nargs='+', default=[1],
                        help='num_input_tokens の値（複数指定可）')
    parser.add_argument('--layers', type=int, nargs='+', default=[1],
                        help='num_layers の値（複数指定可）')
    parser.add_argument('--context-dim', type=int, nargs='+', default=[768],
                        help='context_dim の値（複数指定可、デフォルト: 768）')
    parser.add_argument('--samples', type=int, nargs='+', default=DEFAULT_SAMPLE_SIZES,
                        help=f'サンプルサイズ（デフォルト: {DEFAULT_SAMPLE_SIZES}）')
    parser.add_argument('--matrix', action='store_true',
                        help='9設定マトリックス実行（context_dim=768固定、input_tokens=[1,2,3], layers=[1,2,3]）')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='出力ディレクトリ（デフォルト: results/scaling_YYYYMMDD_HHMMSS）')

    # α値スケーリング分析オプション
    parser.add_argument('--alpha-scaling', action='store_true',
                        help='α値のデータ量依存性を測定（スライディングウィンドウ方式）')
    parser.add_argument('--init-samples', type=int, default=50,
                        help='初期サンプル数（デフォルト: 50）')
    parser.add_argument('--multiplier', type=float, default=2.0,
                        help='サンプル数の倍率（デフォルト: 2.0）')
    parser.add_argument('--window-size', type=int, default=4,
                        help='α計算に使うウィンドウサイズ（デフォルト: 4点）')
    parser.add_argument('--num-windows', type=int, default=2,
                        help='ウィンドウ数（デフォルト: 2）')

    args = parser.parse_args()

    # α値スケーリングモード: サンプルサイズを自動生成
    if args.alpha_scaling:
        sample_sizes = generate_sample_sizes(
            init_samples=args.init_samples,
            multiplier=args.multiplier,
            window_size=args.window_size,
            num_windows=args.num_windows,
        )
        window_size = args.window_size
    else:
        sample_sizes = args.samples
        window_size = None

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
        if args.alpha_scaling:
            output_dir = f"results/alpha_scaling_{timestamp}"
        else:
            output_dir = f"results/scaling_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 実験設定の生成
    configs = []
    for input_tokens, layers, context_dim in product(input_tokens_list, layers_list, context_dim_list):
        configs.append((input_tokens, layers, context_dim))

    # 実験情報表示
    print_flush("=" * 70)
    if args.alpha_scaling:
        print_flush("ALPHA SCALING EXPERIMENT (Data Amount Dependency)")
    else:
        print_flush("SCALING LAW EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Configurations: {len(configs)}")
    print_flush(f"Sample sizes: {sample_sizes}")
    if args.alpha_scaling:
        print_flush(f"Window size: {args.window_size} points")
        print_flush(f"Number of windows: {args.num_windows}")
        print_flush(f"Multiplier: {args.multiplier}x")
    print_flush(f"Total experiments: {len(configs) * len(sample_sizes)}")
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
    except Exception:
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
            sample_sizes=sample_sizes,
            output_dir=output_dir,
        )
        all_results.append(result)

    # サマリー表示
    print_summary(all_results, output_dir)

    # α値スケーリング分析
    if args.alpha_scaling and window_size:
        print_alpha_progression(all_results, window_size, output_dir)

    # 実行時間
    elapsed = time.time() - start_time
    print_flush(f"\nTotal time: {elapsed/60:.1f} min")

    # 実験メタデータ保存
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'sample_sizes': sample_sizes,
        'configs': [
            {'input_tokens': it, 'layers': layers, 'context_dim': cd}
            for it, layers, cd in configs
        ],
    }
    # α値スケーリングモードの追加情報
    if args.alpha_scaling:
        metadata['alpha_scaling'] = {
            'init_samples': args.init_samples,
            'multiplier': args.multiplier,
            'window_size': args.window_size,
            'num_windows': args.num_windows,
        }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()
