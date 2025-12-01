#!/usr/bin/env python3
"""
CVFP仮説検証スクリプト

仮説: 多様性損失のみ（dist_reg_weight=1.0、CVFPなし）で訓練しても、
      ある程度固定点に収束した状態になっているのではないか？

検証方法:
1. 各多様性アルゴリズムをdist_reg_weight=1.0で訓練
2. Val ERでearly stopping
3. 訓練後、追加で5回contextを伝搬
4. 各イテレーションでcontextの変化量（MSE、コサイン類似度）を測定
5. 変化量が小さければ「固定点に近い状態」と判断

Usage:
    # ローカル（CPU）: 最小サンプル
    python3 scripts/cvfp_hypothesis_test.py -s 2

    # Colab（GPU）: 本格実験
    python3 scripts/cvfp_hypothesis_test.py -s 100
"""

import os
import sys
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Callable

import torch
import torch.nn.functional as F

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ResidualConfig
from src.models import LLM
from src.trainers.phase1.memory import MemoryPhase1Trainer
from src.providers.data.memory import MemoryDataProvider
from src.evaluation.metrics import analyze_fixed_points
from src.utils.io import print_flush


def set_seed(seed: int = 42) -> None:
    """乱数シード固定"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 多様性アルゴリズム定義（diversity_algorithm_experiment.pyと同じ）
# =============================================================================

def mcdl_loss(contexts: torch.Tensor) -> torch.Tensor:
    """MCDL: Mean-Centered Dispersion Loss（現行）"""
    context_mean = contexts.mean(dim=0)
    deviation = contexts - context_mean
    return -torch.norm(deviation, p=2) / len(contexts)


def odcm_loss(contexts: torch.Tensor) -> torch.Tensor:
    """ODCM: Off-Diagonal Covariance Minimization（VICReg風）

    VICRegの完全な実装:
    1. Variance Loss: 各次元の分散を1以上に維持
    2. Covariance Loss: 共分散行列の非対角成分を最小化
    """
    centered = contexts - contexts.mean(dim=0)

    # Variance Loss: 各次元の標準偏差を1以上に
    std = torch.sqrt(centered.var(dim=0) + 1e-4)
    var_loss = torch.relu(1.0 - std).mean()

    # Covariance Loss: 非対角成分を最小化
    cov = (centered.T @ centered) / (len(contexts) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = off_diag.pow(2).sum() / contexts.size(1)

    return var_loss + 0.04 * cov_loss


def due_loss(contexts: torch.Tensor) -> torch.Tensor:
    """DUE: Dimension Usage Entropy（次元活性度均一化）"""
    dim_activation = contexts.abs().mean(dim=0)
    dim_probs = dim_activation / (dim_activation.sum() + 1e-10)
    entropy = -(dim_probs * torch.log(dim_probs + 1e-10)).sum()
    max_entropy = torch.log(torch.tensor(contexts.size(1), dtype=torch.float32, device=contexts.device))
    return -(entropy / max_entropy)


def ctm_loss(contexts: torch.Tensor) -> torch.Tensor:
    """CTM: Covariance Trace Maximization（統計的分散）"""
    contexts_centered = contexts - contexts.mean(dim=0)
    cov = (contexts_centered.T @ contexts_centered) / (len(contexts) - 1)
    return -torch.trace(cov) / contexts.size(1)


def udel_loss(contexts: torch.Tensor) -> torch.Tensor:
    """UDEL: Uniform Distribution Entropy Loss（Barlow Twins風）"""
    contexts_std = contexts.std(dim=0)
    std_mean = contexts_std.mean()
    std_var = ((contexts_std - std_mean) ** 2).mean()
    return std_var - std_mean


DIVERSITY_ALGORITHMS = {
    'MCDL': mcdl_loss,
    'ODCM': odcm_loss,
    'DUE': due_loss,
    'CTM': ctm_loss,
    'UDEL': udel_loss,
}


# =============================================================================
# カスタムトレーナー（dist_reg_weight=1.0専用）
# =============================================================================

class DiversityOnlyTrainer(MemoryPhase1Trainer):
    """多様性損失のみで訓練するトレーナー（CVFP損失なし）"""

    def __init__(
        self,
        model: LLM,
        config: ResidualConfig,
        device: torch.device,
        diversity_fn: Callable[[torch.Tensor], torch.Tensor],
        algorithm_name: str
    ):
        super().__init__(model, config, device)
        self.diversity_fn = diversity_fn
        self.algorithm_name = algorithm_name

    def _compute_diversity_loss(self, contexts: torch.Tensor) -> torch.Tensor:
        """カスタム多様性損失関数を使用"""
        return self.diversity_fn(contexts)


# =============================================================================
# 固定点収束度測定
# =============================================================================

def measure_convergence_after_training(
    model: LLM,
    token_embeds: torch.Tensor,
    initial_contexts: torch.Tensor,
    device: torch.device,
    num_input_tokens: int,
    num_iterations: int = 5
) -> List[Dict[str, float]]:
    """
    訓練後の追加伝播で固定点収束度を測定

    Args:
        model: 訓練済みモデル
        token_embeds: トークン埋め込み [num_tokens, embed_dim]
        initial_contexts: 訓練終了時のコンテキスト [num_tokens, context_dim]
        device: デバイス
        num_input_tokens: 入力トークン数
        num_iterations: 追加伝播回数

    Returns:
        各イテレーションの測定結果リスト
    """
    model.eval()
    results = []

    previous_contexts = initial_contexts.clone()
    num_tokens = len(token_embeds)

    for iteration in range(num_iterations):
        # シーケンシャル処理で1回伝播
        current_contexts = torch.zeros(num_tokens, model.context_dim, device=device)

        # 最終contextから開始（前回の最終状態を引き継ぐ）
        context = previous_contexts[-1].unsqueeze(0)

        # トークン履歴を初期化
        token_history = [torch.zeros(model.embed_dim, device=device)
                         for _ in range(num_input_tokens - 1)]

        with torch.no_grad():
            for i, token_embed in enumerate(token_embeds):
                token_history.append(token_embed)
                combined_tokens = torch.cat(token_history[-num_input_tokens:], dim=-1)

                context = model.context_block(context, combined_tokens.unsqueeze(0))
                current_contexts[i] = context.squeeze(0)

                if len(token_history) > num_input_tokens:
                    token_history = token_history[-num_input_tokens:]

        # 変化量を測定
        mse = F.mse_loss(current_contexts, previous_contexts).item()

        # コサイン類似度（トークンごとに計算して平均）
        cos_sim = F.cosine_similarity(current_contexts, previous_contexts, dim=1).mean().item()

        # L2ノルム変化
        l2_change = (current_contexts - previous_contexts).norm(p=2).item() / num_tokens

        # Effective Rank
        metrics = analyze_fixed_points(current_contexts, label=f"Iter{iteration+1}", verbose=False)
        er = metrics['effective_rank']
        er_pct = er / model.context_dim * 100

        results.append({
            'iteration': iteration + 1,
            'mse': mse,
            'cosine_similarity': cos_sim,
            'l2_change_per_token': l2_change,
            'effective_rank': er,
            'effective_rank_pct': er_pct,
        })

        # 次のイテレーションのために更新
        previous_contexts = current_contexts.clone()

    return results


# =============================================================================
# 単一実験実行
# =============================================================================

def run_single_experiment(
    algorithm_name: str,
    diversity_fn: Callable[[torch.Tensor], torch.Tensor],
    num_samples: int,
    device: torch.device,
    seed: int = 42,
    num_post_iterations: int = 5
) -> Dict[str, Any]:
    """単一アルゴリズムの実験を実行"""

    set_seed(seed)

    # 設定作成（dist_reg_weight=1.0 = CVFPなし）
    config = ResidualConfig()
    config.num_samples = num_samples
    config.dist_reg_weight = 1.0  # 多様性損失のみ
    config.val_text_file = "./data/example_val.txt"

    # データ読み込み
    data_provider = MemoryDataProvider(config)
    data_provider.load_data()

    train_token_ids = data_provider.get_all_train_tokens(device)
    val_token_ids = data_provider.get_all_val_tokens(device)

    num_train_tokens = len(train_token_ids)
    print_flush(f"\n{'='*60}")
    print_flush(f"Algorithm: {algorithm_name} (dist_reg_weight=1.0, NO CVFP)")
    print_flush(f"Samples: {num_samples}, Tokens: {num_train_tokens:,}")
    print_flush(f"{'='*60}")

    # モデル作成
    set_seed(seed)
    model = LLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        num_layers=config.num_layers,
        num_input_tokens=config.num_input_tokens,
        use_pretrained_embeddings=config.use_pretrained_embeddings,
        use_weight_tying=config.use_weight_tying,
        config=config
    )
    model.to(device)

    # トレーナー作成
    trainer = DiversityOnlyTrainer(
        model, config, device,
        diversity_fn=diversity_fn,
        algorithm_name=algorithm_name
    )

    # Phase 1実行（val early stopping付き）
    train_start = time.time()
    train_result = trainer.train(train_token_ids, label=algorithm_name, val_token_ids=val_token_ids)
    if isinstance(train_result, tuple):
        train_contexts = train_result[0]
    else:
        train_contexts = train_result
    train_time = time.time() - train_start

    # 訓練終了時のER
    train_metrics = analyze_fixed_points(train_contexts, label="Train", verbose=False)
    train_er = train_metrics['effective_rank']
    train_er_pct = train_er / config.context_dim * 100

    print_flush(f"\nTraining completed in {train_time:.1f}s")
    print_flush(f"Train ER: {train_er_pct:.1f}%")

    # トークン埋め込み取得（追加伝播用）
    with torch.no_grad():
        token_embeds = model.token_embedding(train_token_ids.unsqueeze(0).to(device))
        token_embeds = model.embed_norm(token_embeds).squeeze(0)

    # 追加伝播で固定点収束度を測定
    print_flush(f"\nMeasuring convergence with {num_post_iterations} additional iterations...")
    convergence_results = measure_convergence_after_training(
        model, token_embeds, train_contexts, device,
        config.num_input_tokens, num_post_iterations
    )

    # 結果表示
    print_flush("\n  Iter | MSE      | Cos Sim | L2/tok  | ER%")
    print_flush("  " + "-"*50)
    for r in convergence_results:
        print_flush(
            f"  {r['iteration']:4d} | {r['mse']:.6f} | {r['cosine_similarity']:.4f}  | "
            f"{r['l2_change_per_token']:.4f}  | {r['effective_rank_pct']:.1f}%"
        )

    # 収束判定
    final_mse = convergence_results[-1]['mse']
    final_cos_sim = convergence_results[-1]['cosine_similarity']
    mse_trend = convergence_results[-1]['mse'] - convergence_results[0]['mse']

    if final_cos_sim > 0.99 and final_mse < 0.01:
        verdict = "CONVERGED (fixed point)"
    elif final_cos_sim > 0.95:
        verdict = "NEAR-CONVERGED"
    elif mse_trend < 0:
        verdict = "CONVERGING (improving)"
    else:
        verdict = "NOT CONVERGED"

    print_flush(f"\n  Verdict: {verdict}")

    return {
        'algorithm': algorithm_name,
        'num_samples': num_samples,
        'num_tokens': num_train_tokens,
        'train_time': train_time,
        'train_er_pct': train_er_pct,
        'convergence_iterations': convergence_results,
        'final_mse': final_mse,
        'final_cosine_similarity': final_cos_sim,
        'mse_trend': mse_trend,
        'verdict': verdict,
    }


# =============================================================================
# メイン
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='CVFP仮説検証: 多様性損失のみで固定点に収束するか？')
    parser.add_argument(
        '--algorithms', '-a',
        nargs='+',
        default=list(DIVERSITY_ALGORITHMS.keys()),
        help='テストするアルゴリズム'
    )
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=2,
        help='サンプル数（ローカル: 2-5, Colab: 100+）'
    )
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=5,
        help='追加伝播回数'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='出力ディレクトリ'
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_flush("="*70)
    print_flush("CVFP HYPOTHESIS TEST")
    print_flush("Hypothesis: Diversity-only training (no CVFP) leads to near-fixed-point")
    print_flush("="*70)
    print_flush(f"Device: {device}")
    print_flush(f"Algorithms: {args.algorithms}")
    print_flush(f"Samples: {args.samples}")
    print_flush(f"Post-training iterations: {args.iterations}")

    # 実験実行
    all_results = []
    for algo_name in args.algorithms:
        if algo_name not in DIVERSITY_ALGORITHMS:
            print_flush(f"Unknown algorithm: {algo_name}, skipping")
            continue

        result = run_single_experiment(
            algorithm_name=algo_name,
            diversity_fn=DIVERSITY_ALGORITHMS[algo_name],
            num_samples=args.samples,
            device=device,
            num_post_iterations=args.iterations
        )
        all_results.append(result)

    # サマリー
    print_flush("\n" + "="*70)
    print_flush("SUMMARY")
    print_flush("="*70)
    print_flush(f"{'Algorithm':<10} | {'Train ER%':>10} | {'Final MSE':>10} | {'Final Cos':>10} | Verdict")
    print_flush("-"*70)
    for r in all_results:
        print_flush(
            f"{r['algorithm']:<10} | {r['train_er_pct']:>10.1f} | {r['final_mse']:>10.6f} | "
            f"{r['final_cosine_similarity']:>10.4f} | {r['verdict']}"
        )

    # 結果保存
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"./importants/logs/cvfp_hypothesis_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.json")

    with open(output_file, 'w') as f:
        json.dump({
            'hypothesis': 'Diversity-only training leads to near-fixed-point convergence',
            'config': {
                'dist_reg_weight': 1.0,
                'samples': args.samples,
                'post_iterations': args.iterations,
            },
            'results': all_results,
        }, f, indent=2)

    print_flush(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
