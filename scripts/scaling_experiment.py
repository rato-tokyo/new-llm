#!/usr/bin/env python3
"""
スケーリング則実験スクリプト（統合版）

すべての設定はconfig.pyで一元管理。コマンドライン引数は不要。

使用方法:
  # 基本実行（config.pyの設定を使用）
  python3 scripts/scaling_experiment.py

  # 出力ディレクトリのみ指定可能
  python3 scripts/scaling_experiment.py --output-dir importants/logs/1130_v8

Colab実行用:
  !cd /content/new-llm && python3 scripts/scaling_experiment.py

設定変更:
  config.pyの以下のセクションを編集:
  - scaling_alpha_mode: α値スケーリングモード
  - scaling_matrix_mode: マトリックスモード
  - scaling_input_tokens, scaling_layers, scaling_context_dim: 実験対象
  - fnn_num_layers, fnn_expand_factor, fnn_activation: FFN設定
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from itertools import product

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.experiments import ExperimentRunner, ExperimentConfig
from src.evaluation.metrics import calculate_scaling_law
from src.utils.io import print_flush


# 定数
RANDOM_SEED = 42


def generate_sample_sizes(init_samples: int, multiplier: float,
                          window_size: int, num_windows: int) -> list[int]:
    """サンプルサイズのリストを生成"""
    total_points = window_size + (num_windows - 1)
    samples = []
    current: float = float(init_samples)
    for _ in range(total_points):
        samples.append(int(current))
        current *= multiplier
    return samples


def calculate_sliding_window_alphas(results: list, window_size: int) -> list[dict]:
    """スライディングウィンドウでα値を計算"""
    if len(results) < window_size:
        return []

    window_results = []
    num_windows = len(results) - window_size + 1

    for i in range(num_windows):
        window_data = results[i:i + window_size]
        scaling = calculate_scaling_law(window_data)

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

        if len(window_results) >= 2:
            alphas = [w['alpha'] for w in window_results if w['alpha'] is not None]
            if len(alphas) >= 2:
                alpha_change = alphas[-1] - alphas[0]
                alpha_change_pct = (alphas[-1] - alphas[0]) / abs(alphas[0]) * 100
                print_flush(f"\n  α change: {alphas[0]:.5f} → {alphas[-1]:.5f} "
                           f"(Δ = {alpha_change:+.5f}, {alpha_change_pct:+.2f}%)")

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

    with open(os.path.join(output_dir, 'alpha_progression.json'), 'w') as f:
        json.dump(all_progressions, f, indent=2)

    print_flush(f"\nAlpha progression saved to: {output_dir}/alpha_progression.json")

    return all_progressions


def run_single_config(runner, num_input_tokens, num_layers, context_dim,
                      sample_sizes, output_dir, config):
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
            embed_dim=config.embed_dim,
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

    scaling = calculate_scaling_law(results)

    config_result = {
        'config_name': config_name,
        'num_input_tokens': num_input_tokens,
        'num_layers': num_layers,
        'context_dim': context_dim,
        'results': results,
        'scaling': scaling,
    }

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

    header = f"{'Config':<18} {'α':>7} {'A':>10} {'R²':>6} {'PPL':>7} {'Acc':>6} {'T.PPL':>7} {'ER':>5} {'Iter':>4}"
    print_flush(header)
    print_flush("-" * 100)

    summary_data = []

    for result in all_results:
        config_name = result['config_name']
        scaling = result['scaling']

        if result['results']:
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

    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=2)

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
    # configから設定を読み込み
    config = Config()

    # コマンドライン引数は出力ディレクトリのみ
    parser = argparse.ArgumentParser(
        description='スケーリング則実験スクリプト（設定はconfig.pyで一元管理）',
    )
    parser.add_argument('--output-dir', type=str, default=None,
                        help='出力ディレクトリ（デフォルト: config.pyのscaling_output_dirまたは自動生成）')
    args = parser.parse_args()

    # 設定の表示
    print_flush("=" * 70)
    print_flush("CONFIGURATION (from config.py)")
    print_flush("=" * 70)
    print_flush(f"FFN: type={config.fnn_type}, layers={config.fnn_num_layers}, "
                f"expand={config.fnn_expand_factor}, activation={config.fnn_activation}")
    print_flush(f"Phase 1: max_iter={config.phase1_max_iterations}, "
                f"grad_clip={config.phase1_gradient_clip}")

    # Phase 1 Validation Early Stopping設定
    val_es_enabled = getattr(config, 'phase1_val_early_stopping', False)
    if val_es_enabled:
        val_freq = getattr(config, 'phase1_val_frequency', 5)
        val_patience = getattr(config, 'phase1_val_patience', 2)
        val_sample = getattr(config, 'phase1_val_sample_size', 10000)
        print_flush(f"Phase 1 Val ES: enabled, freq={val_freq}, patience={val_patience}, sample={val_sample}")
    else:
        print_flush("Phase 1 Val ES: disabled")

    # サンプルサイズの決定
    if config.scaling_alpha_mode:
        sample_sizes = generate_sample_sizes(
            init_samples=config.scaling_init_samples,
            multiplier=config.scaling_multiplier,
            window_size=config.scaling_window_size,
            num_windows=config.scaling_num_windows,
        )
        window_size = config.scaling_window_size
    else:
        sample_sizes = config.scaling_sample_sizes
        window_size = None

    # 実験設定の決定
    if config.scaling_matrix_mode:
        input_tokens_list = config.scaling_input_tokens_list
        layers_list = config.scaling_layers_list
        context_dim_list = config.scaling_context_dim_list
    else:
        input_tokens_list = [config.scaling_input_tokens]
        layers_list = [config.scaling_layers]
        context_dim_list = [config.scaling_context_dim]

    # 出力ディレクトリ
    if args.output_dir:
        output_dir = args.output_dir
    elif config.scaling_output_dir:
        output_dir = config.scaling_output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if config.scaling_alpha_mode:
            output_dir = f"results/alpha_scaling_{timestamp}"
        else:
            output_dir = f"results/scaling_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 実験設定の生成
    configs = []
    for input_tokens, layers, context_dim in product(input_tokens_list, layers_list, context_dim_list):
        configs.append((input_tokens, layers, context_dim))

    # 実験情報表示
    print_flush("\n" + "=" * 70)
    if config.scaling_alpha_mode:
        print_flush("ALPHA SCALING EXPERIMENT (Data Amount Dependency)")
    else:
        print_flush("SCALING LAW EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Configurations: {len(configs)}")
    print_flush(f"Sample sizes: {sample_sizes}")
    if config.scaling_alpha_mode:
        print_flush(f"Window size: {config.scaling_window_size} points")
        print_flush(f"Number of windows: {config.scaling_num_windows}")
        print_flush(f"Multiplier: {config.scaling_multiplier}x")
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
    runner = ExperimentRunner(base_config=config)

    all_results = []
    for input_tokens, layers, context_dim in configs:
        result = run_single_config(
            runner=runner,
            num_input_tokens=input_tokens,
            num_layers=layers,
            context_dim=context_dim,
            sample_sizes=sample_sizes,
            output_dir=output_dir,
            config=config,
        )
        all_results.append(result)

    # サマリー表示
    print_summary(all_results, output_dir)

    # α値スケーリング分析
    if config.scaling_alpha_mode and window_size:
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
        'ffn_settings': {
            'type': config.fnn_type,
            'num_layers': config.fnn_num_layers,
            'expand_factor': config.fnn_expand_factor,
            'activation': config.fnn_activation,
        },
        'phase1_settings': {
            'max_iterations': config.phase1_max_iterations,
            'gradient_clip': config.phase1_gradient_clip,
            'val_early_stopping': getattr(config, 'phase1_val_early_stopping', False),
            'val_frequency': getattr(config, 'phase1_val_frequency', 5),
            'val_patience': getattr(config, 'phase1_val_patience', 2),
            'val_sample_size': getattr(config, 'phase1_val_sample_size', 10000),
        },
    }
    if config.scaling_alpha_mode:
        metadata['alpha_scaling'] = {
            'init_samples': config.scaling_init_samples,
            'multiplier': config.scaling_multiplier,
            'window_size': config.scaling_window_size,
            'num_windows': config.scaling_num_windows,
        }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()
