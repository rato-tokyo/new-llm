#!/usr/bin/env python3
"""
多様性損失アルゴリズム比較実験スクリプト

各アルゴリズムの純粋な性能を比較するため、Phase 1のみ実行し、
Val ER、Train ER、処理時間を測定する。

使用方法:
  python3 scripts/diversity_algorithm_experiment.py

  # 特定のアルゴリズムのみ実行
  python3 scripts/diversity_algorithm_experiment.py --algorithms MCDL ODCM

  # サンプルサイズを指定
  python3 scripts/diversity_algorithm_experiment.py --samples 50 100

Colab実行用:
  !cd /content/new-llm && python3 scripts/diversity_algorithm_experiment.py
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
from src.trainers.phase1.memory import MemoryPhase1Trainer
from src.evaluation.metrics import analyze_fixed_points
from src.providers.data import MemoryDataProvider
from src.utils.io import print_flush
from src.utils.device import clear_gpu_cache
from src.utils.seed import set_seed
from src.evaluation.convergence import forward_sequential


# =============================================================================
# 多様性損失アルゴリズム定義
# =============================================================================

def compute_diversity_loss_mcdl(contexts: torch.Tensor) -> torch.Tensor:
    """
    MCDL (Mean-Centered Dispersion Loss) - 現行アルゴリズム

    平均中心からの分散を最大化（L2ノルム）
    計算コスト: O(n×d)
    """
    context_mean = contexts.mean(dim=0)
    deviation = contexts - context_mean
    return -torch.norm(deviation, p=2) / len(contexts)


def compute_diversity_loss_odcm(contexts: torch.Tensor) -> torch.Tensor:
    """
    ODCM (Off-Diagonal Covariance Minimization) - VICReg風

    VICRegの完全な実装:
    1. Variance Loss: 各次元の分散を1以上に維持
    2. Covariance Loss: 共分散行列の非対角成分を最小化

    計算コスト: O(n×d + d²)
    """
    centered = contexts - contexts.mean(dim=0)

    # Variance Loss: 各次元の標準偏差を1以上に
    std = torch.sqrt(centered.var(dim=0) + 1e-4)
    var_loss = torch.relu(1.0 - std).mean()  # std < 1の次元にペナルティ

    # Covariance Loss: 非対角成分を最小化
    cov = torch.mm(centered.t(), centered) / len(contexts)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / contexts.shape[1]

    # VICRegでは variance:covariance = 25:1 だが、CVFPでは調整
    return var_loss + 0.04 * cov_loss


def compute_diversity_loss_due(contexts: torch.Tensor) -> torch.Tensor:
    """
    DUE (Dimension Usage Entropy) - 次元活性度均一化

    各次元の使用頻度のエントロピーを最大化（死んだ次元を防ぐ）
    計算コスト: O(n×d)
    """
    dim_usage = contexts.abs().mean(dim=0)
    dim_usage_normalized = dim_usage / (dim_usage.sum() + 1e-10)
    entropy = -(dim_usage_normalized * torch.log(dim_usage_normalized + 1e-10)).sum()
    max_entropy = torch.log(torch.tensor(contexts.shape[1], dtype=torch.float, device=contexts.device))
    return (max_entropy - entropy) / max_entropy


def compute_diversity_loss_ctm(contexts: torch.Tensor) -> torch.Tensor:
    """
    CTM (Covariance Trace Maximization) - 統計的分散最大化

    共分散行列のトレースを最大化（総分散の最大化）
    計算コスト: O(n×d + d²)
    """
    centered = contexts - contexts.mean(dim=0)
    cov = torch.mm(centered.t(), centered) / len(contexts)
    return -torch.trace(cov)


def compute_diversity_loss_udel(contexts: torch.Tensor) -> torch.Tensor:
    """
    UDEL (Uniform Distribution Entropy Loss) - Barlow Twins風

    各次元の値分布を均一化（分散が1に近づくように）
    計算コスト: O(n×d)
    """
    std = contexts.std(dim=0)
    # 分散が0の次元を避けるためのクランプ
    std = torch.clamp(std, min=1e-10)
    normalized = (contexts - contexts.mean(dim=0)) / std
    var_per_dim = normalized.var(dim=0)
    return ((var_per_dim - 1) ** 2).mean()


def compute_diversity_loss_sdl(contexts: torch.Tensor) -> torch.Tensor:
    """
    SDL (Spectral Diversity Loss) - Effective Rank直接最大化

    SVD特異値のエントロピーを直接最大化
    計算コスト: O(n×d²) - 高コスト、参考用
    """
    _, S, _ = torch.svd(contexts)
    S_normalized = S / (S.sum() + 1e-10)
    entropy = -(S_normalized * torch.log(S_normalized + 1e-10)).sum()
    return -entropy


# =============================================================================
# 追加アルゴリズム（自己教師あり学習文献より）
# =============================================================================

def compute_diversity_loss_uniformity(contexts: torch.Tensor) -> torch.Tensor:
    """
    UNIF (Uniformity Loss) - 球面上の一様分布への誘導

    L2正規化後のペア間距離の指数和を最小化
    計算コスト: O(n²×d) - ペア計算
    参考: Understanding Contrastive Learning
    """
    import torch.nn.functional as F
    normalized = F.normalize(contexts, dim=1)
    # ペア間の二乗距離を計算
    dists_sq = torch.cdist(normalized, normalized, p=2).pow(2)
    # 対角成分（自己距離）を除外
    n = contexts.size(0)
    mask = ~torch.eye(n, dtype=torch.bool, device=contexts.device)
    dists_sq = dists_sq[mask]
    # 一様性損失（小さいほど一様）
    return torch.log(torch.exp(-2 * dists_sq).mean() + 1e-10)


def compute_diversity_loss_decorr(contexts: torch.Tensor) -> torch.Tensor:
    """
    DECORR (Decorrelation Loss) - 相関行列の対角化

    相関行列を単位行列に近づける
    計算コスト: O(n×d + d²)
    参考: Decorrelated Batch Normalization
    """
    # 標準化
    std = contexts.std(dim=0)
    std = torch.clamp(std, min=1e-4)
    normalized = (contexts - contexts.mean(dim=0)) / std
    # 相関行列
    corr = torch.mm(normalized.t(), normalized) / len(contexts)
    # 単位行列との差
    identity = torch.eye(contexts.size(1), device=contexts.device)
    return ((corr - identity) ** 2).mean()


def compute_diversity_loss_nuclear(contexts: torch.Tensor) -> torch.Tensor:
    """
    NUC (Nuclear Norm Maximization) - 核ノルム最大化

    特異値の和（核ノルム）を最大化
    計算コスト: O(n×d²) - SVD計算
    """
    # 中心化
    centered = contexts - contexts.mean(dim=0)
    # 核ノルム = 特異値の和
    nuclear_norm = torch.linalg.matrix_norm(centered, ord='nuc')
    return -nuclear_norm / len(contexts)


def compute_diversity_loss_hsic(contexts: torch.Tensor) -> torch.Tensor:
    """
    HSIC (Hilbert-Schmidt Independence Criterion) - 次元間独立性

    各次元ペア間のHSICを最小化（独立性促進）
    計算コスト: O(n²×d) - カーネル計算
    参考: HSIC Bottleneck
    """
    n = contexts.size(0)
    d = contexts.size(1)

    # サンプル数が多すぎる場合はサブサンプリング
    if n > 1000:
        indices = torch.randperm(n)[:1000]
        contexts = contexts[indices]
        n = 1000

    # 中心化行列
    H = torch.eye(n, device=contexts.device) - torch.ones(n, n, device=contexts.device) / n

    # 各次元のRBFカーネル
    total_hsic = torch.tensor(0.0, device=contexts.device)
    # 計算量削減のため、ランダムに次元ペアをサンプリング
    num_pairs = min(50, d * (d - 1) // 2)
    for _ in range(num_pairs):
        i, j = torch.randint(0, d, (2,))
        if i == j:
            continue
        xi = contexts[:, i:i+1]
        xj = contexts[:, j:j+1]

        # RBFカーネル（簡易版）
        Ki = torch.exp(-torch.cdist(xi, xi, p=2).pow(2))
        Kj = torch.exp(-torch.cdist(xj, xj, p=2).pow(2))

        # HSIC = trace(KHLH) / (n-1)^2
        hsic = torch.trace(Ki @ H @ Kj @ H) / ((n - 1) ** 2)
        total_hsic = total_hsic + hsic

    return total_hsic / num_pairs


def compute_diversity_loss_infonce(contexts: torch.Tensor) -> torch.Tensor:
    """
    InfoNCE - 情報量最大化コントラスティブ損失

    自己とのみ正例、他は負例として扱うInfoNCE
    計算コスト: O(n²×d)
    参考: CPC, MoCo
    """
    import torch.nn.functional as F
    normalized = F.normalize(contexts, dim=1)
    # 類似度行列
    sim = torch.mm(normalized, normalized.t())
    # 温度パラメータ
    temperature = 0.1
    sim = sim / temperature
    # 対角成分が正例（自己）、他は負例
    # ただしバッチ内自己教師なので、シフトした形で計算
    n = contexts.size(0)
    # ラベルは対角（自分自身）
    labels = torch.arange(n, device=contexts.device)
    # InfoNCE損失（CrossEntropy形式）
    loss = F.cross_entropy(sim, labels)
    return loss


def compute_diversity_loss_wmse(contexts: torch.Tensor) -> torch.Tensor:
    """
    W-MSE (Whitening MSE) - 白色化変換後のMSE

    ZCA白色化後の表現のMSEを最小化
    計算コスト: O(n×d + d³) - 固有値分解
    参考: W-MSE, Shuffled-DBN
    """
    # 中心化
    centered = contexts - contexts.mean(dim=0)
    # 共分散行列
    cov = torch.mm(centered.t(), centered) / (len(contexts) - 1)
    # 正則化
    cov = cov + 1e-4 * torch.eye(contexts.size(1), device=contexts.device)

    # 固有値分解
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    eigenvalues = torch.clamp(eigenvalues, min=1e-4)

    # ZCA白色化行列: W = V @ diag(1/sqrt(λ)) @ V^T
    whitening = eigenvectors @ torch.diag(1.0 / torch.sqrt(eigenvalues)) @ eigenvectors.t()

    # 白色化後の表現
    whitened = centered @ whitening

    # 目標: 白色化後の共分散が単位行列になること
    whitened_cov = torch.mm(whitened.t(), whitened) / (len(contexts) - 1)
    identity = torch.eye(contexts.size(1), device=contexts.device)

    return ((whitened_cov - identity) ** 2).mean()


# アルゴリズム辞書
DIVERSITY_ALGORITHMS: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    # 既存アルゴリズム
    'MCDL': compute_diversity_loss_mcdl,
    'ODCM': compute_diversity_loss_odcm,
    'DUE': compute_diversity_loss_due,
    'CTM': compute_diversity_loss_ctm,
    'UDEL': compute_diversity_loss_udel,
    'SDL': compute_diversity_loss_sdl,
    # 追加アルゴリズム
    'UNIF': compute_diversity_loss_uniformity,
    'DECORR': compute_diversity_loss_decorr,
    'NUC': compute_diversity_loss_nuclear,
    'HSIC': compute_diversity_loss_hsic,
    'InfoNCE': compute_diversity_loss_infonce,
    'WMSE': compute_diversity_loss_wmse,
}

ALGORITHM_DESCRIPTIONS = {
    # 既存アルゴリズム
    'MCDL': 'Mean-Centered Dispersion Loss (現行)',
    'ODCM': 'Off-Diagonal Covariance Minimization (VICReg風)',
    'DUE': 'Dimension Usage Entropy (次元活性度均一化)',
    'CTM': 'Covariance Trace Maximization (統計的分散)',
    'UDEL': 'Uniform Distribution Entropy Loss (Barlow Twins風)',
    'SDL': 'Spectral Diversity Loss (ER直接最大化, 高コスト)',
    # 追加アルゴリズム
    'UNIF': 'Uniformity Loss (球面一様分布)',
    'DECORR': 'Decorrelation Loss (相関行列対角化)',
    'NUC': 'Nuclear Norm Maximization (核ノルム最大化)',
    'HSIC': 'HSIC (次元間独立性)',
    'InfoNCE': 'InfoNCE (コントラスティブ)',
    'WMSE': 'Whitening MSE (白色化)',
}

# 高コストアルゴリズム（デフォルトでスキップ）
HIGH_COST_ALGORITHMS = {'SDL', 'UNIF', 'HSIC', 'InfoNCE'}


# =============================================================================
# カスタムPhase1Trainer（多様性損失関数を注入可能）
# =============================================================================

class CustomDiversityPhase1Trainer(MemoryPhase1Trainer):
    """多様性損失関数をカスタマイズ可能なPhase1Trainer"""

    def __init__(
        self,
        model: torch.nn.Module,
        config: Any,
        device: torch.device,
        diversity_fn: Callable[[torch.Tensor], torch.Tensor],
        algorithm_name: str = "Custom"
    ):
        super().__init__(model, config, device)
        self._diversity_fn = diversity_fn
        self._algorithm_name = algorithm_name
        self._diversity_loss_times: List[float] = []

    def _compute_diversity_loss(self, contexts: torch.Tensor) -> torch.Tensor:
        """オーバーライド: カスタム多様性損失関数を使用"""
        start = time.perf_counter()
        result = self._diversity_fn(contexts)
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
    max_iterations: Optional[int] = None
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
    trainer = CustomDiversityPhase1Trainer(
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
    base_config: ResidualConfig,
    device: torch.device,
    output_dir: str
) -> List[Dict[str, Any]]:
    """全実験を実行"""

    all_results = []
    total_experiments = len(algorithms) * len(sample_sizes)
    current = 0

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
            print_flush(f"\n[{current}/{total_experiments}] {algorithm_name} with {num_samples} samples")

            try:
                result = run_single_experiment(
                    algorithm_name=algorithm_name,
                    diversity_fn=diversity_fn,
                    num_samples=num_samples,
                    base_config=base_config,
                    device=device
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
                    'num_samples': num_samples,
                    'error': str(e),
                })

    return all_results


def print_results_table(results: List[Dict[str, Any]], context_dim: int):
    """結果をテーブル形式で表示"""

    print_flush("\n" + "=" * 100)
    print_flush("DIVERSITY ALGORITHM COMPARISON RESULTS")
    print_flush("=" * 100)

    # ヘッダー
    header = f"{'Algorithm':<10} {'Samples':>8} {'Tokens':>10} {'Train ER%':>10} {'Val ER%':>10} {'Time(s)':>8} {'Loss(ms)':>10}"
    print_flush(header)
    print_flush("-" * 100)

    # 結果を表示
    for r in results:
        if 'error' in r:
            print_flush(f"{r['algorithm']:<10} {r['num_samples']:>8} {'ERROR':>10} {'-':>10} {'-':>10} {'-':>8} {'-':>10}")
            continue

        print_flush(
            f"{r['algorithm']:<10} "
            f"{r['num_samples']:>8} "
            f"{r['num_train_tokens']:>10,} "
            f"{r['train_er_pct']:>10.1f} "
            f"{r['val_er_pct']:>10.1f} "
            f"{r['train_time_sec']:>8.1f} "
            f"{r['avg_loss_time_ms']:>10.2f}"
        )

    print_flush("=" * 100)

    # アルゴリズム別サマリー
    print_flush("\n" + "=" * 100)
    print_flush("SUMMARY BY ALGORITHM (Average across sample sizes)")
    print_flush("=" * 100)

    # アルゴリズムごとに集計
    algo_stats: Dict[str, Dict[str, List[float]]] = {}
    for r in results:
        if 'error' in r:
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

    print_flush("=" * 100)
    print_flush("(Sorted by Avg Val ER% descending)")


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
        help='Sample sizes to test (default: 50 100 200 400)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory (default: auto-generated)'
    )
    parser.add_argument(
        '--skip-sdl',
        action='store_true',
        help='Skip SDL algorithm (high computational cost)'
    )

    args = parser.parse_args()

    # SDLをスキップ
    algorithms = args.algorithms
    if args.skip_sdl and 'SDL' in algorithms:
        algorithms = [a for a in algorithms if a != 'SDL']
        print_flush("Note: Skipping SDL algorithm (use --no-skip-sdl to include)")

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
    print_flush(f"Output: {output_dir}")
    print_flush("\nConfig:")
    print_flush(f"  context_dim: {config.context_dim}")
    print_flush(f"  num_layers: {config.num_layers}")
    print_flush(f"  dist_reg_weight: {config.dist_reg_weight}")
    print_flush(f"  phase1_max_iterations: {config.phase1_max_iterations}")

    # 実験実行
    results = run_all_experiments(
        algorithms=algorithms,
        sample_sizes=args.samples,
        base_config=config,
        device=device,
        output_dir=output_dir
    )

    # 結果表示
    print_results_table(results, config.context_dim)

    # 結果保存
    save_results(results, output_dir, config)

    print_flush("\n✅ Experiment completed!")


if __name__ == '__main__':
    main()
