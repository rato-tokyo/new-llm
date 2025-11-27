"""
Colab用 拡張スケーリング実験 (案1 + 案2)

案1: サンプル数スケールアップ [500, 1000, 2000, 5000] × Layer 6
案2: レイヤー数比較 [6, 9, 12] × 2000 samples

GPUで実行: L4推奨（23.8GB）
実験時間: 約2-3時間

使い方:
    !python colab_extended.py

出力:
    - results/extended_scaling_samples.json（案1）
    - results/extended_scaling_layers.json（案2）
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


class DataLoader:
    """データローダー（1回のロードでtrain/valを分離）"""

    def __init__(self, config: ResidualConfig, device: torch.device):
        self.config = config
        self.device = device
        self.sample_tokens = []
        self.sample_boundaries = []

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
    num_layers: int,
    config: ResidualConfig,
    data_loader: DataLoader,
    device: torch.device,
    max_train_samples: int
) -> dict:
    """単一実験（Phase1 + Phase2）"""

    experiment_start = time.time()

    print_flush(f"\n  num_samples={num_samples}, num_layers={num_layers}")

    set_seed(42)

    # モデル初期化
    model = LLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        num_layers=num_layers,
        num_input_tokens=getattr(config, 'num_input_tokens', 1),
        use_pretrained_embeddings=config.use_pretrained_embeddings
    )
    model.to(device)

    # データ取得
    train_tokens = data_loader.get_train(num_samples)
    val_tokens = data_loader.get_val(max_train_samples)

    num_train_tokens = len(train_tokens)
    num_val_tokens = len(val_tokens)

    print_flush(f"  Train tokens: {num_train_tokens:,}")
    print_flush(f"  Val tokens: {num_val_tokens:,}")

    result = {
        "num_samples": num_samples,
        "num_layers": num_layers,
        "num_train_tokens": num_train_tokens,
        "num_val_tokens": num_val_tokens,
    }

    # ========== Phase 1 ==========
    print_flush(f"  Phase 1 starting...")
    phase1_start = time.time()

    trainer1 = MemoryPhase1Trainer(model, config, device)
    train_contexts = trainer1.train(train_tokens, label=f"Train")

    phase1_time = time.time() - phase1_start
    train_stats = trainer1.get_training_stats()

    # Phase 1評価
    val_result = trainer1.evaluate(val_tokens, label="Val")
    train_analysis = analyze_fixed_points(train_contexts, label="Train", verbose=False)
    val_analysis = analyze_fixed_points(val_result.contexts, label="Val", verbose=False)

    result.update({
        "phase1_time_sec": phase1_time,
        "phase1_iterations": train_stats.get("iterations", 0),
        "train_effective_rank": train_analysis["effective_rank"],
        "train_effective_rank_percent": train_analysis["effective_rank"] / config.context_dim * 100,
        "val_effective_rank": val_analysis["effective_rank"],
        "val_effective_rank_percent": val_analysis["effective_rank"] / config.context_dim * 100,
    })

    print_flush(f"  Phase 1 done: {format_time(phase1_time)}, Train ER={result['train_effective_rank_percent']:.1f}%")

    # ========== Phase 2 ==========
    print_flush(f"  Phase 2 starting...")
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
        "phase2_best_epoch": best_epoch,
        "val_loss": history['val_loss'][best_epoch - 1],
        "val_ppl": history['val_ppl'][best_epoch - 1],
        "val_accuracy": history['val_acc'][best_epoch - 1],
    })

    experiment_time = time.time() - experiment_start
    result["total_time_sec"] = experiment_time

    print_flush(f"  RESULT: Val PPL={result['val_ppl']:.2f}, Val Acc={result['val_accuracy']*100:.2f}% ({format_time(experiment_time)})")

    # クリーンアップ（確実にGPUメモリを解放）
    del model, trainer1, trainer2, train_contexts, train_tokens, val_tokens, val_result
    import gc
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return result


def run_experiment_set(
    experiment_name: str,
    sample_layer_pairs: list,
    config: ResidualConfig,
    data_loader: DataLoader,
    device: torch.device,
    max_train_samples: int,
    val_samples: int
) -> list:
    """実験セットを実行"""

    print_flush(f"\n{'='*70}")
    print_flush(f"EXPERIMENT SET: {experiment_name}")
    print_flush(f"{'='*70}")
    print_flush(f"Experiments: {len(sample_layer_pairs)}")

    results = []
    total_start = time.time()

    for idx, (num_samples, num_layers) in enumerate(sample_layer_pairs, 1):
        print_flush(f"\n[{idx}/{len(sample_layer_pairs)}] Running experiment...")

        try:
            result = run_single_experiment(
                num_samples=num_samples,
                num_layers=num_layers,
                config=config,
                data_loader=data_loader,
                device=device,
                max_train_samples=max_train_samples
            )
            results.append(result)

            # 進捗表示
            elapsed = time.time() - total_start
            avg_time = elapsed / idx
            remaining = avg_time * (len(sample_layer_pairs) - idx)
            print_flush(f"  Progress: {idx}/{len(sample_layer_pairs)}, Remaining: {format_time(remaining)}")

        except Exception as e:
            print_flush(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

    total_time = time.time() - total_start
    print_flush(f"\n{experiment_name} complete: {format_time(total_time)}")

    return results, total_time


def save_results(output_path: str, experiment_name: str, results: list, total_time: float, config: ResidualConfig, device: torch.device):
    """結果を保存"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_data = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "total_time_sec": total_time,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "config": {
            "context_dim": config.context_dim,
            "embed_dim": config.embed_dim,
            "phase1_max_iterations": config.phase1_max_iterations,
            "phase1_learning_rate": config.phase1_learning_rate,
            "dist_reg_weight": config.dist_reg_weight,
            "phase2_epochs": config.phase2_epochs,
            "phase2_batch_size": config.phase2_batch_size,
            "phase2_learning_rate": config.phase2_learning_rate,
        },
        "results": results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print_flush(f"Results saved to: {output_path}")


def print_summary_table(results: list, title: str):
    """結果サマリーを表示"""
    print_flush(f"\n{'='*100}")
    print_flush(f"{title}")
    print_flush(f"{'='*100}")
    print_flush(f"{'Samples':>8} | {'Layers':>6} | {'Tokens':>10} | {'Time':>8} | {'Train ER%':>10} | {'Val PPL':>10} | {'Val Acc%':>10}")
    print_flush(f"{'-'*8}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for r in results:
        time_str = format_time(r['total_time_sec'])
        print_flush(
            f"{r['num_samples']:>8} | "
            f"{r['num_layers']:>6} | "
            f"{r['num_train_tokens']:>10,} | "
            f"{time_str:>8} | "
            f"{r['train_effective_rank_percent']:>9.1f}% | "
            f"{r['val_ppl']:>10.2f} | "
            f"{r['val_accuracy']*100:>9.2f}%"
        )


def main():
    total_start = time.time()

    print_flush(f"\n{'='*70}")
    print_flush("EXTENDED SCALING EXPERIMENT")
    print_flush("案1: Sample scaling [500, 1000, 2000, 5000] × Layer 6")
    print_flush("案2: Layer comparison [6, 9, 12] × 2000 samples")
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
        print_flush("WARNING: Running on CPU (very slow)")

    # 設定
    config = ResidualConfig()

    # 実験パラメータ
    # 案1: サンプル数スケールアップ（Layer 6固定）
    sample_sizes_exp1 = [500, 1000, 2000, 5000]
    layer_exp1 = 6

    # 案2: レイヤー数比較（2000 samples固定）
    layers_exp2 = [6, 9, 12]
    samples_exp2 = 2000

    # 検証データ: 50サンプル（前回10→50に増加、約4000トークン）
    val_samples = 50
    max_train_samples = max(max(sample_sizes_exp1), samples_exp2)
    total_samples = max_train_samples + val_samples

    print_flush(f"\n案1: samples={sample_sizes_exp1}, layer={layer_exp1}")
    print_flush(f"案2: layers={layers_exp2}, samples={samples_exp2}")
    print_flush(f"Val samples: {val_samples} (固定)")
    print_flush(f"Total samples to load: {total_samples}")

    # データローダー初期化
    print_flush(f"\n{'='*70}")
    print_flush("Loading all data...")
    print_flush(f"{'='*70}")
    data_loader = DataLoader(config, device)
    data_loader.load_all(total_samples)

    all_results = []

    # ========== 案1: サンプル数スケールアップ ==========
    print_flush(f"\n{'#'*70}")
    print_flush("# 案1: Sample Scaling Experiment")
    print_flush(f"{'#'*70}")

    exp1_pairs = [(s, layer_exp1) for s in sample_sizes_exp1]
    results_exp1, time_exp1 = run_experiment_set(
        experiment_name="Sample Scaling (Layer 6)",
        sample_layer_pairs=exp1_pairs,
        config=config,
        data_loader=data_loader,
        device=device,
        max_train_samples=max_train_samples,
        val_samples=val_samples
    )

    save_results(
        "./results/extended_scaling_samples.json",
        "Sample Scaling (Layer 6)",
        results_exp1,
        time_exp1,
        config,
        device
    )

    print_summary_table(results_exp1, "案1: Sample Scaling Results")
    all_results.extend(results_exp1)

    # ========== 案2: レイヤー数比較 ==========
    print_flush(f"\n{'#'*70}")
    print_flush("# 案2: Layer Comparison Experiment")
    print_flush(f"{'#'*70}")

    # Layer 6は案1で既に2000 samplesを実行済みなのでスキップ可能だが、
    # 一貫性のため再実行
    exp2_pairs = [(samples_exp2, l) for l in layers_exp2]
    results_exp2, time_exp2 = run_experiment_set(
        experiment_name="Layer Comparison (2000 samples)",
        sample_layer_pairs=exp2_pairs,
        config=config,
        data_loader=data_loader,
        device=device,
        max_train_samples=max_train_samples,
        val_samples=val_samples
    )

    save_results(
        "./results/extended_scaling_layers.json",
        "Layer Comparison (2000 samples)",
        results_exp2,
        time_exp2,
        config,
        device
    )

    print_summary_table(results_exp2, "案2: Layer Comparison Results")

    # ========== 最終サマリー ==========
    total_time = time.time() - total_start

    print_flush(f"\n{'='*70}")
    print_flush("ALL EXPERIMENTS COMPLETE")
    print_flush(f"{'='*70}")
    print_flush(f"Total time: {format_time(total_time)}")
    print_flush(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 全結果サマリー
    print_summary_table(all_results + results_exp2, "ALL RESULTS SUMMARY")

    # スケーリング傾向
    print_flush(f"\n{'='*70}")
    print_flush("SCALING TRENDS")
    print_flush(f"{'='*70}")

    if len(results_exp1) >= 2:
        ppls = [r['val_ppl'] for r in results_exp1]
        samples = [r['num_samples'] for r in results_exp1]
        print_flush(f"案1 PPL trend: {ppls[0]:.2f} ({samples[0]} samples) -> {ppls[-1]:.2f} ({samples[-1]} samples)")

    if len(results_exp2) >= 2:
        ppls = [r['val_ppl'] for r in results_exp2]
        layers = [r['num_layers'] for r in results_exp2]
        print_flush(f"案2 PPL trend: {ppls[0]:.2f} ({layers[0]} layers) -> {ppls[-1]:.2f} ({layers[-1]} layers)")


if __name__ == "__main__":
    main()
