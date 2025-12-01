#!/usr/bin/env python3
"""
多様性損失アルゴリズム比較実験スクリプト（5アルゴリズム版）

採用アルゴリズム:
  - MCDL: 現行ベースライン（最速）
  - ODCM: VICReg風（推奨、低コスト・高ER）
  - SDL: ER直接最大化（最高ER、高コスト）
  - NUC: 核ノルム最大化（高ER、高コスト）
  - WMSE: 白色化ベース（中コスト）

使用方法:
  # デフォルト: context_dim=768,1000 で全5アルゴリズム実行
  python3 scripts/diversity_algorithm_experiment.py

  # 特定のアルゴリズムのみ実行
  python3 scripts/diversity_algorithm_experiment.py -a MCDL ODCM

  # サンプルサイズを指定
  python3 scripts/diversity_algorithm_experiment.py -s 100

  # context_dimを指定（複数可）
  python3 scripts/diversity_algorithm_experiment.py -c 768 1000

  # 高コスト（SDL, NUC）を含める
  python3 scripts/diversity_algorithm_experiment.py --include-high-cost

Colab実行用:
  !cd /content/new-llm && python3 scripts/diversity_algorithm_experiment.py -s 100 -c 768 1000 --include-high-cost
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Callable, Dict, Any, List, Optional

import torch

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ResidualConfig
from src.models import LLM
from src.trainers.phase1 import FlexibleDiversityTrainer
from src.evaluation.metrics import analyze_fixed_points
from src.providers.data import MemoryDataProvider
from src.utils.io import print_flush
from src.utils.device import clear_gpu_cache
from src.utils.seed import set_seed
from src.evaluation.convergence import forward_sequential
from src.losses.diversity import (
    DIVERSITY_ALGORITHMS,
    ALGORITHM_DESCRIPTIONS,
    HIGH_COST_ALGORITHMS,
)


# =============================================================================
# タイミング計測付きトレーナー（実験用拡張）
# =============================================================================

class TimedDiversityTrainer(FlexibleDiversityTrainer):
    """タイミング計測機能付きの多様性トレーナー（実験用）"""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,
        device: torch.device,
        diversity_fn: Callable[[torch.Tensor], torch.Tensor],
        algorithm_name: str = "Custom"
    ):
        super().__init__(model, config, device, diversity_fn, algorithm_name)
        self._diversity_loss_times: List[float] = []

    def _compute_diversity_loss(self, contexts: torch.Tensor) -> torch.Tensor:
        """オーバーライド: タイミング計測付きの多様性損失計算"""
        start = time.perf_counter()
        result = super()._compute_diversity_loss(contexts)
        elapsed = time.perf_counter() - start
        self._diversity_loss_times.append(elapsed * 1000)  # ms
        return result

    def get_avg_diversity_loss_time_ms(self) -> float:
        """多様性損失計算の平均時間(ms)を取得"""
        if not self._diversity_loss_times:
            return 0.0
        return sum(self._diversity_loss_times) / len(self._diversity_loss_times)

    def reset_timing_stats(self):
        """タイミング統計をリセット"""
        self._diversity_loss_times = []


# =============================================================================
# 実験実行
# =============================================================================

def run_single_experiment(
    algorithm_name: str,
    diversity_fn: Callable[[torch.Tensor], torch.Tensor],
    num_samples: int,
    base_config: ResidualConfig,
    device: torch.device,
    seed: int = 42,
    max_iterations: Optional[int] = None,
    context_dim: Optional[int] = None
) -> Dict[str, Any]:
    """単一の実験を実行"""

    set_seed(seed)

    # データ読み込み用設定
    data_config = ResidualConfig()
    data_config.num_samples = num_samples
    data_config.val_text_file = "./data/example_val.txt"

    # イテレーション数を上書き
    if max_iterations is not None:
        base_config.phase1_max_iterations = max_iterations

    # context_dimを上書き
    if context_dim is not None:
        base_config.context_dim = context_dim

    # データプロバイダー
    data_provider = MemoryDataProvider(data_config)
    data_provider.load_data()

    train_token_ids = data_provider.get_all_train_tokens(device)
    val_token_ids = data_provider.get_all_val_tokens(device)

    num_train_tokens = len(train_token_ids)
    num_val_tokens = len(val_token_ids)

    # モデル作成
    set_seed(seed)  # モデル初期化前に再度シード固定
    model = LLM(
        vocab_size=base_config.vocab_size,
        embed_dim=base_config.embed_dim,
        context_dim=base_config.context_dim,
        num_layers=base_config.num_layers,
        num_input_tokens=base_config.num_input_tokens,
        use_pretrained_embeddings=base_config.use_pretrained_embeddings,
        use_weight_tying=base_config.use_weight_tying,
        config=base_config
    )
    model.to(device)

    # トレーナー作成
    trainer = TimedDiversityTrainer(
        model, base_config, device,
        diversity_fn=diversity_fn,
        algorithm_name=algorithm_name
    )

    # Phase 1実行（val_token_idsを渡してearly stoppingを有効化）
    train_start = time.time()
    train_result = trainer.train(train_token_ids, label=f"{algorithm_name}", val_token_ids=val_token_ids)
    # train()はTensor or Tuple[Tensor, Tensor, Tensor]を返す
    if isinstance(train_result, tuple):
        train_contexts = train_result[0]
    else:
        train_contexts = train_result
    train_time = time.time() - train_start

    # 評価
    model.eval()
    with torch.no_grad():
        # 訓練データER
        train_metrics = analyze_fixed_points(train_contexts, label="Train", verbose=False)
        train_er = train_metrics['effective_rank']
        train_er_pct = train_er / base_config.context_dim * 100

        # 検証データER（シーケンシャル処理で評価、サンプリング）
        # GPUでは10000トークンで十分高速
        val_sample_size = min(len(val_token_ids), 10000)
        val_sample_ids = val_token_ids[:val_sample_size]
        val_token_embeds = model.token_embedding(val_sample_ids.unsqueeze(0).to(device))
        val_token_embeds = model.embed_norm(val_token_embeds).squeeze(0)
        val_contexts = forward_sequential(
            model, val_token_embeds, None, device,
            base_config.num_input_tokens
        )
        val_metrics = analyze_fixed_points(val_contexts, label="Val", verbose=False)
        val_er = val_metrics['effective_rank']
        val_er_pct = val_er / base_config.context_dim * 100

    avg_loss_time_ms = trainer.get_avg_diversity_loss_time_ms()

    # メモリ解放
    del model, trainer, train_contexts, val_contexts
    del train_token_ids, val_token_ids
    clear_gpu_cache(device)

    return {
        'algorithm': algorithm_name,
        'context_dim': base_config.context_dim,
        'num_samples': num_samples,
        'num_train_tokens': num_train_tokens,
        'num_val_tokens': num_val_tokens,
        'train_er': train_er,
        'train_er_pct': train_er_pct,
        'val_er': val_er,
        'val_er_pct': val_er_pct,
        'train_time_sec': train_time,
        'avg_loss_time_ms': avg_loss_time_ms,
    }


def run_all_experiments(
    algorithms: List[str],
    sample_sizes: List[int],
    context_dims: List[int],
    base_config: ResidualConfig,
    device: torch.device,
    output_dir: str
) -> List[Dict[str, Any]]:
    """全実験を実行"""

    all_results = []
    total_experiments = len(algorithms) * len(sample_sizes) * len(context_dims)
    current = 0

    for ctx_dim in context_dims:
        print_flush(f"\n{'#'*70}")
        print_flush(f"# context_dim = {ctx_dim}")
        print_flush(f"{'#'*70}")

        for algorithm_name in algorithms:
            if algorithm_name not in DIVERSITY_ALGORITHMS:
                print_flush(f"⚠️ Unknown algorithm: {algorithm_name}, skipping")
                continue

            diversity_fn = DIVERSITY_ALGORITHMS[algorithm_name]
            print_flush(f"\n{'='*70}")
            print_flush(f"Algorithm: {algorithm_name} - {ALGORITHM_DESCRIPTIONS.get(algorithm_name, '')}")
            print_flush(f"{'='*70}")

            for num_samples in sample_sizes:
                current += 1
                print_flush(f"\n[{current}/{total_experiments}] {algorithm_name} | ctx_dim={ctx_dim} | {num_samples} samples")

                try:
                    result = run_single_experiment(
                        algorithm_name=algorithm_name,
                        diversity_fn=diversity_fn,
                        num_samples=num_samples,
                        base_config=base_config,
                        device=device,
                        context_dim=ctx_dim
                    )
                    all_results.append(result)

                    print_flush(f"  Train ER: {result['train_er_pct']:.1f}%")
                    print_flush(f"  Val ER: {result['val_er_pct']:.1f}%")
                    print_flush(f"  Time: {result['train_time_sec']:.1f}s")
                    print_flush(f"  Loss calc: {result['avg_loss_time_ms']:.2f}ms/iter")

                except Exception as e:
                    print_flush(f"  ❌ Error: {e}")
                    all_results.append({
                        'algorithm': algorithm_name,
                        'context_dim': ctx_dim,
                        'num_samples': num_samples,
                        'error': str(e),
                    })

    return all_results


def print_results_table(results: List[Dict[str, Any]], context_dims: List[int]):
    """結果をテーブル形式で表示"""

    print_flush("\n" + "=" * 115)
    print_flush("DIVERSITY ALGORITHM COMPARISON RESULTS")
    print_flush("=" * 115)

    # ヘッダー
    header = f"{'Algorithm':<10} {'ctx_dim':>8} {'Samples':>8} {'Tokens':>10} {'Train ER%':>10} {'Val ER%':>10} {'Time(s)':>8} {'Loss(ms)':>10}"
    print_flush(header)
    print_flush("-" * 115)

    # 結果を表示
    for r in results:
        if 'error' in r:
            print_flush(f"{r['algorithm']:<10} {r.get('context_dim', '-'):>8} {r['num_samples']:>8} {'ERROR':>10} {'-':>10} {'-':>10} {'-':>8} {'-':>10}")
            continue

        print_flush(
            f"{r['algorithm']:<10} "
            f"{r['context_dim']:>8} "
            f"{r['num_samples']:>8} "
            f"{r['num_train_tokens']:>10,} "
            f"{r['train_er_pct']:>10.1f} "
            f"{r['val_er_pct']:>10.1f} "
            f"{r['train_time_sec']:>8.1f} "
            f"{r['avg_loss_time_ms']:>10.2f}"
        )

    print_flush("=" * 115)

    # context_dim別サマリー
    for ctx_dim in context_dims:
        print_flush(f"\n" + "=" * 100)
        print_flush(f"SUMMARY BY ALGORITHM (context_dim={ctx_dim})")
        print_flush("=" * 100)

        # アルゴリズムごとに集計（該当context_dimのみ）
        algo_stats: Dict[str, Dict[str, List[float]]] = {}
        for r in results:
            if 'error' in r:
                continue
            if r.get('context_dim') != ctx_dim:
                continue
            algo = r['algorithm']
            if algo not in algo_stats:
                algo_stats[algo] = {
                    'train_er_pct': [],
                    'val_er_pct': [],
                    'time': [],
                    'loss_time': []
                }
            algo_stats[algo]['train_er_pct'].append(r['train_er_pct'])
            algo_stats[algo]['val_er_pct'].append(r['val_er_pct'])
            algo_stats[algo]['time'].append(r['train_time_sec'])
            algo_stats[algo]['loss_time'].append(r['avg_loss_time_ms'])

        header = f"{'Algorithm':<10} {'Avg Train ER%':>14} {'Avg Val ER%':>12} {'Avg Time(s)':>12} {'Avg Loss(ms)':>14}"
        print_flush(header)
        print_flush("-" * 100)

        # Val ER%でソート（降順）
        sorted_algos = sorted(
            algo_stats.items(),
            key=lambda x: sum(x[1]['val_er_pct']) / len(x[1]['val_er_pct']) if x[1]['val_er_pct'] else 0,
            reverse=True
        )

        for algo, stats in sorted_algos:
            avg_train = sum(stats['train_er_pct']) / len(stats['train_er_pct'])
            avg_val = sum(stats['val_er_pct']) / len(stats['val_er_pct'])
            avg_time = sum(stats['time']) / len(stats['time'])
            avg_loss = sum(stats['loss_time']) / len(stats['loss_time'])

            print_flush(
                f"{algo:<10} "
                f"{avg_train:>14.1f} "
                f"{avg_val:>12.1f} "
                f"{avg_time:>12.1f} "
                f"{avg_loss:>14.2f}"
            )

    print_flush("(Sorted by Val ER% descending)")


def save_results(results: List[Dict[str, Any]], output_dir: str, config: ResidualConfig):
    """結果をJSONファイルに保存"""

    os.makedirs(output_dir, exist_ok=True)

    output = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'context_dim': config.context_dim,
            'embed_dim': config.embed_dim,
            'num_layers': config.num_layers,
            'num_input_tokens': config.num_input_tokens,
            'dist_reg_weight': config.dist_reg_weight,
            'phase1_max_iterations': config.phase1_max_iterations,
            'phase1_learning_rate': config.phase1_learning_rate,
        },
        'algorithm_descriptions': ALGORITHM_DESCRIPTIONS,
        'results': results,
    }

    output_path = os.path.join(output_dir, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print_flush(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Diversity Algorithm Comparison Experiment')
    parser.add_argument(
        '--algorithms', '-a',
        nargs='+',
        default=list(DIVERSITY_ALGORITHMS.keys()),
        help='Algorithms to test (default: all)'
    )
    parser.add_argument(
        '--samples', '-s',
        nargs='+',
        type=int,
        default=[100],
        help='Sample sizes to test (default: 100)'
    )
    parser.add_argument(
        '--context-dims', '-c',
        nargs='+',
        type=int,
        default=[768, 1000],
        help='Context dimensions to test (default: 768 1000)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory (default: auto-generated)'
    )
    parser.add_argument(
        '--include-high-cost',
        action='store_true',
        help='Include high-cost algorithms (SDL, NUC)'
    )

    args = parser.parse_args()

    # 高コストアルゴリズムをフィルタリング
    algorithms = args.algorithms
    if not args.include_high_cost:
        skipped = [a for a in algorithms if a in HIGH_COST_ALGORITHMS]
        algorithms = [a for a in algorithms if a not in HIGH_COST_ALGORITHMS]
        if skipped:
            print_flush(f"Note: Skipping high-cost algorithms: {skipped} (use --include-high-cost to include)")

    # 出力ディレクトリ
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"importants/logs/{timestamp}_diversity_comparison"

    # 設定
    config = ResidualConfig()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # 情報表示
    print_flush("=" * 70)
    print_flush("DIVERSITY ALGORITHM COMPARISON EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Device: {device}")
    if device.type == "cuda":
        print_flush(f"GPU: {torch.cuda.get_device_name(0)}")
        print_flush(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    print_flush(f"\nAlgorithms: {algorithms}")
    print_flush(f"Sample sizes: {args.samples}")
    print_flush(f"Context dims: {args.context_dims}")
    print_flush(f"Output: {output_dir}")
    print_flush("\nConfig:")
    print_flush(f"  num_layers: {config.num_layers}")
    print_flush(f"  dist_reg_weight: {config.dist_reg_weight}")
    print_flush(f"  phase1_max_iterations: {config.phase1_max_iterations}")

    # 実験実行
    results = run_all_experiments(
        algorithms=algorithms,
        sample_sizes=args.samples,
        context_dims=args.context_dims,
        base_config=config,
        device=device,
        output_dir=output_dir
    )

    # 結果表示
    print_results_table(results, args.context_dims)

    # 結果保存
    save_results(results, output_dir, config)

    print_flush("\n✅ Experiment completed!")


if __name__ == '__main__':
    main()
