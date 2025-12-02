#!/usr/bin/env python3
"""
統一実験スクリプト

Phase 1 (OACD) + Phase 2 (トークン予測) の実験を実行。
レイヤー構成を柔軟に指定可能。

使用方法:
  # デフォルト (C2T2, 2000サンプル)
  python3 scripts/experiment.py

  # レイヤー指定
  python3 scripts/experiment.py --context-layers 1 --token-layers 1  # C1T1
  python3 scripts/experiment.py --context-layers 2 --token-layers 2  # C2T2
  python3 scripts/experiment.py -cl 1 -tl 2                          # C1T2

  # サンプル数・context_dim指定
  python3 scripts/experiment.py -s 1000 -c 300

  # 複数構成を比較 (all)
  python3 scripts/experiment.py --compare c1t1 c2t2

  # プリセット
  python3 scripts/experiment.py --preset c1t1
  python3 scripts/experiment.py --preset c2t2
"""

import os
import sys
import argparse
from datetime import datetime
from typing import List, Tuple

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.experiments.runner import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentResult,
    print_results_summary,
    save_results,
)
from src.utils.io import print_flush


# プリセット定義
PRESETS = {
    'c1t1': (1, 1),
    'c2t1': (2, 1),
    'c1t2': (1, 2),
    'c2t2': (2, 2),
    'c3t3': (3, 3),
}


def parse_layer_config(config_str: str) -> Tuple[int, int]:
    """
    レイヤー構成文字列をパース

    例: 'c1t1' -> (1, 1), 'c2t2' -> (2, 2)
    """
    config_str = config_str.lower()
    if config_str in PRESETS:
        return PRESETS[config_str]

    # cXtY形式をパース
    import re
    match = re.match(r'c(\d+)t(\d+)', config_str)
    if match:
        return int(match.group(1)), int(match.group(2))

    raise ValueError(f"Invalid config format: {config_str}. Use 'cXtY' format (e.g., c1t1, c2t2)")


def main():
    parser = argparse.ArgumentParser(
        description='Run Phase 1 + Phase 2 experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/experiment.py                          # Default (config.num_layers)
  python3 scripts/experiment.py --preset c1t1            # C1T1 preset
  python3 scripts/experiment.py -cl 2 -tl 2              # Custom layers
  python3 scripts/experiment.py --compare c1t1 c2t2      # Compare multiple configs
        """
    )

    # レイヤー指定
    layer_group = parser.add_argument_group('Layer Configuration')
    layer_group.add_argument(
        '--preset', '-p',
        choices=list(PRESETS.keys()),
        help='Use preset layer configuration'
    )
    layer_group.add_argument(
        '--context-layers', '-cl',
        type=int,
        help='Number of ContextBlock layers'
    )
    layer_group.add_argument(
        '--token-layers', '-tl',
        type=int,
        help='Number of TokenBlock layers'
    )
    layer_group.add_argument(
        '--compare',
        nargs='+',
        metavar='CONFIG',
        help='Compare multiple configurations (e.g., c1t1 c2t2)'
    )

    # データ・アーキテクチャ
    data_group = parser.add_argument_group('Data & Architecture')
    data_group.add_argument(
        '--samples', '-s',
        type=int,
        default=2000,
        help='Number of samples (default: 2000)'
    )
    data_group.add_argument(
        '--context-dim', '-c',
        type=int,
        default=500,
        help='Context dimension (default: 500)'
    )

    # 出力
    output_group = parser.add_argument_group('Output')
    output_group.add_argument(
        '--output-dir', '-o',
        help='Output directory (default: auto-generated)'
    )
    output_group.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress verbose output'
    )

    args = parser.parse_args()

    # 実験構成を決定
    configs_to_run: List[Tuple[int, int]] = []

    if args.compare:
        # 複数構成を比較
        for config_str in args.compare:
            configs_to_run.append(parse_layer_config(config_str))
    elif args.preset:
        # プリセット使用
        configs_to_run.append(PRESETS[args.preset])
    elif args.context_layers is not None or args.token_layers is not None:
        # 明示的なレイヤー指定
        base_config = Config()
        cl = args.context_layers if args.context_layers is not None else base_config.num_layers
        tl = args.token_layers if args.token_layers is not None else base_config.num_layers
        configs_to_run.append((cl, tl))
    else:
        # デフォルト: base_configのnum_layers
        base_config = Config()
        configs_to_run.append((base_config.num_layers, base_config.num_layers))

    # 出力ディレクトリ
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_dir = args.output_dir
    else:
        config_name = '_'.join(f"c{cl}t{tl}" for cl, tl in configs_to_run)
        output_dir = f"importants/logs/{timestamp}_{config_name}"

    os.makedirs(output_dir, exist_ok=True)

    # ヘッダー表示
    print_flush("=" * 70)
    print_flush("EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Configurations: {[f'C{cl}T{tl}' for cl, tl in configs_to_run]}")
    print_flush(f"Samples: {args.samples}")
    print_flush(f"Context dim: {args.context_dim}")
    print_flush(f"Output: {output_dir}")
    print_flush("=" * 70)

    # 実験実行
    runner = ExperimentRunner(verbose=not args.quiet)
    results: List[ExperimentResult] = []

    for context_layers, token_layers in configs_to_run:
        try:
            exp_config = ExperimentConfig(
                num_samples=args.samples,
                context_dim=args.context_dim,
                context_layers=context_layers,
                token_layers=token_layers,
                verbose=not args.quiet,
            )
            result = runner.run(exp_config)
            results.append(result)
        except Exception as e:
            print_flush(f"ERROR in C{context_layers}T{token_layers}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 結果表示・保存
    if results:
        print_results_summary(results)

        output_file = os.path.join(output_dir, "results.txt")
        save_results(
            results,
            output_file,
            metadata={
                'Samples': args.samples,
                'Context dim': args.context_dim,
                'Configurations': [f'C{cl}T{tl}' for cl, tl in configs_to_run],
            }
        )
        print_flush(f"\nResults saved to: {output_file}")

    print_flush("\n" + "=" * 70)
    print_flush("DONE")
    print_flush("=" * 70)


if __name__ == '__main__':
    main()
