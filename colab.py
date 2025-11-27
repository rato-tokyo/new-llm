"""
Colab用 サンプル数スケーリング実験 (Phase1 + Phase2)

GPUで実行: L4/T4/A100対応
実験時間: 約20-40分（サンプル数による）

使い方:
    !python colab.py

出力:
    - 進捗表示（各サンプル数の結果）
    - results/colab_scaling_experiment.json（詳細データ）
    - 最終サマリーテーブル
"""

import sys
import os
import json
import time
import torch
import random
import numpy as np
from datetime import datetime, timedelta

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ResidualConfig
from src.models import LLM
from src.trainers.phase1 import MemoryPhase1Trainer
from src.trainers.phase2 import Phase2Trainer
from src.evaluation import analyze_fixed_points


def print_flush(msg: str):
    print(msg, flush=True)
    sys.stdout.flush()


def set_seed(seed=42):
    """再現性のためのシード固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def format_time(seconds: float) -> str:
    """秒を読みやすい形式に変換"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def estimate_remaining_time(elapsed_times: list, remaining_count: int) -> str:
    """残り時間を推定"""
    if not elapsed_times:
        return "不明"
    avg_time = sum(elapsed_times) / len(elapsed_times)
    remaining = avg_time * remaining_count
    return format_time(remaining)


class DataLoader:
    """データローダー（1回のロードでtrain/valを分離）"""

    def __init__(self, config: ResidualConfig, device: torch.device):
        self.config = config
        self.device = device
        self.sample_tokens = []  # 各サンプルのトークン列
        self.sample_boundaries = []  # 各サンプルの開始インデックス

    def load_all(self, total_samples: int):
        """全サンプルをロード（train + val用）"""
        print_flush(f"\n  Loading {total_samples} samples from UltraChat...")

        from transformers import GPT2Tokenizer
        from datasets import load_dataset

        tokenizer = GPT2Tokenizer.from_pretrained(self.config.tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset(self.config.dataset_name, split=self.config.dataset_split)

        self.sample_tokens = []
        self.sample_boundaries = [0]

        for i in range(total_samples):
            text = dataset[i]["messages"][0]["content"]
            tokens = tokenizer(
                text,
                max_length=self.config.max_seq_length,
                truncation=True,
                return_tensors="pt"
            )
            self.sample_tokens.append(tokens["input_ids"].squeeze(0))
            self.sample_boundaries.append(self.sample_boundaries[-1] + len(self.sample_tokens[-1]))

        all_tokens = torch.cat(self.sample_tokens).to(self.device)
        print_flush(f"  Total tokens: {len(all_tokens):,}")
        print_flush(f"  Samples loaded: {total_samples}")
        self.all_tokens = all_tokens

    def get_train(self, num_samples: int) -> torch.Tensor:
        """訓練データを取得（サンプル0〜num_samples-1）"""
        end_idx = self.sample_boundaries[num_samples]
        return self.all_tokens[:end_idx]

    def get_val(self, train_samples: int, val_samples: int = 10) -> torch.Tensor:
        """検証データを取得（訓練データの直後）"""
        start_idx = self.sample_boundaries[train_samples]
        end_idx = self.sample_boundaries[train_samples + val_samples]
        return self.all_tokens[start_idx:end_idx]


def run_single_experiment(
    num_samples: int,
    config: ResidualConfig,
    data_loader: DataLoader,
    device: torch.device,
    experiment_idx: int,
    total_experiments: int,
    max_train_samples: int
) -> dict:
    """単一サンプル数での実験（Phase1 + Phase2）"""

    experiment_start = time.time()

    print_flush(f"\n{'='*70}")
    print_flush(f"EXPERIMENT {experiment_idx}/{total_experiments}: num_samples = {num_samples}")
    print_flush(f"{'='*70}")

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

    # データ取得（DataLoaderから）
    train_tokens = data_loader.get_train(num_samples)
    val_tokens = data_loader.get_val(max_train_samples)  # valは常に訓練最大サンプルの直後

    num_train_tokens = len(train_tokens)
    num_val_tokens = len(val_tokens)

    print_flush(f"  Train tokens: {num_train_tokens:,}")
    print_flush(f"  Val tokens: {num_val_tokens:,} (固定)")

    result = {
        "num_samples": num_samples,
        "num_train_tokens": num_train_tokens,
        "num_val_tokens": num_val_tokens,
    }

    # ========== Phase 1 ==========
    print_flush(f"\n--- Phase 1: CVFP Learning ---")
    phase1_start = time.time()

    trainer1 = MemoryPhase1Trainer(model, config, device)
    train_contexts = trainer1.train(train_tokens, label=f"Train (samples={num_samples})")

    phase1_time = time.time() - phase1_start
    train_stats = trainer1.get_training_stats()

    # Phase 1評価
    val_result = trainer1.evaluate(val_tokens, label="Val")
    train_analysis = analyze_fixed_points(train_contexts, label="Train", verbose=False)
    val_analysis = analyze_fixed_points(val_result.contexts, label="Val", verbose=False)

    result.update({
        "phase1_time_sec": phase1_time,
        "phase1_iterations": train_stats.get("iterations", 0),
        "phase1_convergence_rate": train_stats.get("convergence_rate", 0.0),
        "train_effective_rank": train_analysis["effective_rank"],
        "train_effective_rank_percent": train_analysis["effective_rank"] / config.context_dim * 100,
        "val_convergence_status": val_result.status,
        "val_convergence_is_converging": val_result.is_converging,
        "val_effective_rank": val_analysis["effective_rank"],
        "val_effective_rank_percent": val_analysis["effective_rank"] / config.context_dim * 100,
    })

    print_flush(f"  Phase 1 完了: {format_time(phase1_time)}")
    print_flush(f"  Train ER: {result['train_effective_rank']:.1f} ({result['train_effective_rank_percent']:.1f}%)")
    print_flush(f"  Val ER: {result['val_effective_rank']:.1f} ({result['val_effective_rank_percent']:.1f}%)")

    # ========== Phase 2 ==========
    print_flush(f"\n--- Phase 2: Next-Token Prediction ---")
    phase2_start = time.time()

    trainer2 = Phase2Trainer(model, config)
    history = trainer2.train_full(
        train_tokens, val_tokens, device,
        epochs=config.phase2_epochs,
        patience=config.phase2_patience,
        batch_size=config.phase2_batch_size
    )

    phase2_time = time.time() - phase2_start

    # Phase 2結果
    best_epoch = history['best_epoch']
    result.update({
        "phase2_time_sec": phase2_time,
        "phase2_epochs_run": len(history['train_loss']),
        "phase2_early_stopped": history['early_stopped'],
        "phase2_best_epoch": best_epoch,
        "train_loss": history['train_loss'][-1],
        "train_ppl": history['train_ppl'][-1],
        "val_loss": history['val_loss'][best_epoch - 1],
        "val_ppl": history['val_ppl'][best_epoch - 1],
        "val_accuracy": history['val_acc'][best_epoch - 1],
        "best_val_loss": min(history['val_loss']),
        "best_val_ppl": min(history['val_ppl']),
        "best_val_accuracy": max(history['val_acc']),
    })

    experiment_time = time.time() - experiment_start
    result["total_time_sec"] = experiment_time

    # サマリー表示
    print_flush(f"\n{'='*50}")
    print_flush(f"RESULT: {num_samples} samples ({format_time(experiment_time)})")
    print_flush(f"{'='*50}")
    print_flush(f"  Phase 1: Train ER={result['train_effective_rank_percent']:.1f}%, Val ER={result['val_effective_rank_percent']:.1f}%")
    print_flush(f"  Phase 2: Val PPL={result['val_ppl']:.2f}, Val Acc={result['val_accuracy']*100:.2f}%")

    # クリーンアップ
    del model, trainer1, trainer2, train_contexts, train_tokens
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return result


def print_summary_table(results: list):
    """結果のサマリーテーブルを表示"""
    print_flush(f"\n{'='*120}")
    print_flush("SUMMARY TABLE")
    print_flush(f"{'='*120}")
    print_flush(f"{'Samples':>8} | {'Tokens':>10} | {'Time':>8} | {'Train ER%':>10} | {'Val ER%':>10} | {'Val PPL':>10} | {'Val Acc%':>10} | {'P2 Epochs':>10}")
    print_flush(f"{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for r in results:
        time_str = format_time(r['total_time_sec'])
        print_flush(
            f"{r['num_samples']:>8} | "
            f"{r['num_train_tokens']:>10,} | "
            f"{time_str:>8} | "
            f"{r['train_effective_rank_percent']:>9.1f}% | "
            f"{r['val_effective_rank_percent']:>9.1f}% | "
            f"{r['val_ppl']:>10.2f} | "
            f"{r['val_accuracy']*100:>9.2f}% | "
            f"{r['phase2_epochs_run']:>10}"
        )


def main():
    total_start = time.time()

    print_flush(f"\n{'='*70}")
    print_flush("COLAB SCALING EXPERIMENT (Phase1 + Phase2)")
    print_flush(f"{'='*70}")
    print_flush(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # GPU確認
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_flush(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = torch.device('cpu')
        print_flush("WARNING: Running on CPU (slow)")

    # 設定
    config = ResidualConfig()

    # テストするサンプル数（50から開始、対数スケール）
    sample_sizes = [50, 100, 200, 500]
    max_train_samples = max(sample_sizes)  # 訓練で使用する最大サンプル数
    val_samples = 10  # 検証サンプル数

    print_flush(f"\nSample sizes: {sample_sizes}")
    print_flush(f"Config:")
    print_flush(f"  - num_layers: {config.num_layers}")
    print_flush(f"  - context_dim: {config.context_dim}")
    print_flush(f"  - phase1_max_iterations: {config.phase1_max_iterations}")
    print_flush(f"  - phase2_epochs: {config.phase2_epochs}")
    print_flush(f"  - phase2_batch_size: {config.phase2_batch_size}")

    # データローダー初期化（1回で全データをロード）
    print_flush("\n" + "="*70)
    print_flush("Loading all data...")
    print_flush("="*70)
    data_loader = DataLoader(config, device)
    total_samples = max_train_samples + val_samples  # train最大 + val
    data_loader.load_all(total_samples)
    print_flush(f"  Train samples: 0 ~ {max_train_samples - 1}")
    print_flush(f"  Val samples: {max_train_samples} ~ {max_train_samples + val_samples - 1}")

    results = []
    elapsed_times = []

    for idx, num_samples in enumerate(sample_sizes, 1):
        remaining = len(sample_sizes) - idx
        est_remaining = estimate_remaining_time(elapsed_times, remaining + 1)

        print_flush(f"\n[Progress: {idx}/{len(sample_sizes)}] 残り推定時間: {est_remaining}")

        try:
            result = run_single_experiment(
                num_samples=num_samples,
                config=config,
                data_loader=data_loader,
                device=device,
                experiment_idx=idx,
                total_experiments=len(sample_sizes),
                max_train_samples=max_train_samples
            )
            results.append(result)
            elapsed_times.append(result['total_time_sec'])

        except Exception as e:
            print_flush(f"ERROR for {num_samples} samples: {e}")
            import traceback
            traceback.print_exc()

    # サマリー表示
    print_summary_table(results)

    # 合計時間
    total_time = time.time() - total_start
    print_flush(f"\n{'='*70}")
    print_flush(f"EXPERIMENT COMPLETE")
    print_flush(f"{'='*70}")
    print_flush(f"Total time: {format_time(total_time)}")
    print_flush(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # JSON保存
    output_path = "./results/colab_scaling_experiment.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "total_time_sec": total_time,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "config": {
            "num_layers": config.num_layers,
            "context_dim": config.context_dim,
            "embed_dim": config.embed_dim,
            "phase1_max_iterations": config.phase1_max_iterations,
            "phase1_learning_rate": config.phase1_learning_rate,
            "dist_reg_weight": config.dist_reg_weight,
            "phase2_epochs": config.phase2_epochs,
            "phase2_batch_size": config.phase2_batch_size,
            "phase2_learning_rate": config.phase2_learning_rate,
        },
        "val_samples": val_samples,
        "val_tokens_count": val_samples * config.max_seq_length,
        "sample_sizes": sample_sizes,
        "results": results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print_flush(f"\nResults saved to: {output_path}")

    # 精度予測用の要約
    print_flush(f"\n{'='*70}")
    print_flush("SCALING TRENDS (for prediction)")
    print_flush(f"{'='*70}")
    if len(results) >= 2:
        # PPLの変化
        ppls = [r['val_ppl'] for r in results]
        samples = [r['num_samples'] for r in results]
        print_flush(f"Val PPL trend: {ppls[0]:.2f} ({samples[0]} samples) -> {ppls[-1]:.2f} ({samples[-1]} samples)")

        # ERの変化
        ers = [r['val_effective_rank_percent'] for r in results]
        print_flush(f"Val ER% trend: {ers[0]:.1f}% ({samples[0]} samples) -> {ers[-1]:.1f}% ({samples[-1]} samples)")

        # 精度の変化
        accs = [r['val_accuracy']*100 for r in results]
        print_flush(f"Val Acc trend: {accs[0]:.2f}% ({samples[0]} samples) -> {accs[-1]:.2f}% ({samples[-1]} samples)")


if __name__ == "__main__":
    main()
