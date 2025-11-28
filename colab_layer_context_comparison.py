#!/usr/bin/env python3
"""
レイヤー数 vs コンテキスト次元 比較実験

比較対象:
  - Experiment A: num_layers=6, context_multiplier=1 (context_dim=768)
  - Experiment B: num_layers=3, context_multiplier=2 (context_dim=1536)

両者のパラメータ数を比較し、Val PPL, Val Accuracy, Effective Rank を評価する。

設定値:
  - embed_dim, num_input_tokens, vocab_size 等は config.py から取得
  - デフォルト値は使用しない（config.py が唯一の設定ソース）

GPUで実行: L4推奨（23.8GB）
実験時間: 約1-2時間

使い方:
    !python colab_layer_context_comparison.py

出力:
    - results/layer_context_comparison/all_results.json（全結果）
    - results/layer_context_comparison/summary.md（サマリー）
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiments import ExperimentConfig, ExperimentRunner


# ============================================================================
# 実験設定
# ============================================================================

# 比較実験
EXPERIMENTS = [
    ExperimentConfig(
        name="A",
        num_layers=6,
        context_multiplier=1,
        description="Deep & Narrow (6層, 768dim)",
    ),
    ExperimentConfig(
        name="B",
        num_layers=3,
        context_multiplier=2,
        description="Shallow & Wide (3層, 1536dim)",
    ),
]

# サンプル数
SAMPLE_SIZES = [500, 1000]

# 出力ディレクトリ
OUTPUT_DIR = "./results/layer_context_comparison"


# ============================================================================
# メイン
# ============================================================================


def main() -> None:
    """実験を実行"""
    runner = ExperimentRunner(
        experiments=EXPERIMENTS,
        sample_sizes=SAMPLE_SIZES,
        output_dir=OUTPUT_DIR,
        val_samples=50,
        skip_existing=True,
    )
    runner.run()


if __name__ == "__main__":
    main()
