"""
サンプル数スケーリング実験

サンプル数と検証データの収束・Effective Rankの関係を調査。
精度予測のためのデータ収集。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import torch
import random
import numpy as np
from datetime import datetime

from config import ResidualConfig
from src.models import LLM
from src.providers.data import MemoryDataProvider
from src.trainers.phase1 import MemoryPhase1Trainer
from src.evaluation import analyze_fixed_points


def set_seed(seed=42):
    """再現性のためのシード固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(num_samples: int, config: ResidualConfig, val_tokens: torch.Tensor, device: torch.device) -> dict:
    """単一サンプル数での実験を実行

    Args:
        num_samples: 訓練サンプル数
        config: 設定
        val_tokens: 検証データ（全実験で共通）
        device: デバイス
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: num_samples = {num_samples}")
    print(f"{'='*70}")

    set_seed(42)

    # Config更新
    config.num_samples = num_samples

    # モデル初期化
    model = LLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        num_layers=config.num_layers,
        use_pretrained_embeddings=config.use_pretrained_embeddings
    )
    model.to(device)

    # データプロバイダー（訓練データのみ）
    data_provider = MemoryDataProvider(config)
    data_provider.load_data()

    train_tokens = data_provider.get_all_train_tokens(device)

    num_train_tokens = len(train_tokens)
    num_val_tokens = len(val_tokens)

    print(f"  Train tokens: {num_train_tokens:,}")
    print(f"  Val tokens: {num_val_tokens:,}")

    # トレーナー
    trainer = MemoryPhase1Trainer(model, config, device)

    # 訓練
    start_time = time.time()
    train_contexts = trainer.train(train_tokens, label=f"Train (samples={num_samples})")
    train_time = time.time() - start_time

    # 訓練統計
    train_stats = trainer.get_training_stats()

    # 検証データ評価
    val_result = trainer.evaluate(val_tokens, label="Val")

    # Effective Rank分析
    train_analysis = analyze_fixed_points(train_contexts, label="Train", verbose=False)
    val_analysis = analyze_fixed_points(val_result.contexts, label="Val", verbose=False)

    # 結果をまとめる
    result = {
        "num_samples": num_samples,
        "num_train_tokens": num_train_tokens,
        "num_val_tokens": num_val_tokens,
        "train_time_sec": train_time,

        # 訓練結果
        "train_iterations": train_stats.get("iterations", 0),
        "train_convergence_rate": train_stats.get("convergence_rate", 0.0),
        "train_effective_rank": train_analysis["effective_rank"],
        "train_effective_rank_percent": train_analysis["effective_rank"] / config.context_dim * 100,
        "train_avg_distance": train_analysis["avg_distance"],
        "train_avg_cosine": train_analysis["avg_cosine"],
        "train_is_global_attractor": train_analysis["is_global_attractor"],

        # 検証結果
        "val_status": val_result.status,
        "val_is_converging": val_result.is_converging,
        "val_initial_loss": val_result.initial_loss,
        "val_final_loss": val_result.final_loss,
        "val_reduction_percent": val_result.reduction_percent,
        "val_slope": val_result.slope,
        "val_effective_rank": val_analysis["effective_rank"],
        "val_effective_rank_percent": val_analysis["effective_rank"] / config.context_dim * 100,
        "val_avg_distance": val_analysis["avg_distance"],
        "val_avg_cosine": val_analysis["avg_cosine"],
        "val_is_global_attractor": val_analysis["is_global_attractor"],
    }

    # クリーンアップ
    data_provider.close()
    del model, trainer, train_contexts
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return result


def print_summary_table(results: list):
    """結果のサマリーテーブルを表示"""
    print(f"\n{'='*100}")
    print("SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"{'Samples':>8} | {'Tokens':>10} | {'Time(s)':>8} | {'Train ER%':>10} | {'Val ER%':>10} | {'Val Status':>12} | {'Val Final':>10}")
    print(f"{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")

    for r in results:
        print(f"{r['num_samples']:>8} | {r['num_train_tokens']:>10,} | {r['train_time_sec']:>8.1f} | {r['train_effective_rank_percent']:>9.1f}% | {r['val_effective_rank_percent']:>9.1f}% | {r['val_status']:>12} | {r['val_final_loss']:>10.6f}")


def main():
    print(f"\n{'='*70}")
    print("SAMPLE SCALING EXPERIMENT")
    print(f"{'='*70}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 設定
    config = ResidualConfig()

    # テストするサンプル数（対数スケールで増加）
    sample_sizes = [10, 25, 50, 100, 200, 500]

    print(f"\nSample sizes to test: {sample_sizes}")
    print(f"Config: num_layers={config.num_layers}, context_dim={config.context_dim}")
    print(f"Config: phase1_max_iterations={config.phase1_max_iterations}")
    print(f"Config: val_convergence_trials={getattr(config, 'val_convergence_trials', 10)}")

    results = []

    for num_samples in sample_sizes:
        try:
            result = run_experiment(num_samples, config)
            results.append(result)

            # 中間結果を表示
            print(f"\n--- Result for {num_samples} samples ---")
            print(f"  Train ER: {result['train_effective_rank']:.1f} ({result['train_effective_rank_percent']:.1f}%)")
            print(f"  Val ER: {result['val_effective_rank']:.1f} ({result['val_effective_rank_percent']:.1f}%)")
            print(f"  Val Status: {result['val_status']}")
            print(f"  Val Loss: {result['val_initial_loss']:.6f} -> {result['val_final_loss']:.6f} ({result['val_reduction_percent']:+.1f}%)")

        except Exception as e:
            print(f"ERROR for {num_samples} samples: {e}")
            import traceback
            traceback.print_exc()

    # サマリー表示
    print_summary_table(results)

    # JSON保存
    output_path = "./results/sample_scaling_experiment.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_layers": config.num_layers,
            "context_dim": config.context_dim,
            "phase1_max_iterations": config.phase1_max_iterations,
            "phase1_learning_rate": config.phase1_learning_rate,
            "dist_reg_weight": config.dist_reg_weight,
            "val_convergence_trials": getattr(config, 'val_convergence_trials', 10),
        },
        "sample_sizes": sample_sizes,
        "results": results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
