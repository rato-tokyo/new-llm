"""
実験フレームワーク

柔軟な実験を行うための共通コンポーネントを提供する。

使用例:
    from src.experiments import ExperimentRunner, ExperimentConfig

    # 実験設定を定義
    experiments = [
        ExperimentConfig(name="A", num_layers=6, context_multiplier=1),
        ExperimentConfig(name="B", num_layers=3, context_multiplier=2),
    ]

    # 実験を実行
    runner = ExperimentRunner(
        experiments=experiments,
        sample_sizes=[500, 1000],
        output_dir="./results/my_experiment"
    )
    results = runner.run()
"""

from src.experiments.runner import ExperimentRunner
from src.experiments.config import ExperimentConfig
from src.experiments.data_loader import UltraChatDataLoader
from src.experiments.utils import format_time, format_number, set_seed

__all__ = [
    "ExperimentRunner",
    "ExperimentConfig",
    "UltraChatDataLoader",
    "format_time",
    "format_number",
    "set_seed",
]
