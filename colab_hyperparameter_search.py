"""
Colab用 ハイパーパラメータ探索実験

num_input_tokens, context_multiplier, sample_sizes の最適な組み合わせを探索し、
「トークン数 vs Val PPL」のスケーリング則を導出する。

GPUで実行: L4推奨（23.8GB）
実験時間: 約3-6時間（24実験）

使い方:
    !python colab_hyperparameter_search.py

出力:
    - results/hyperparameter_search/all_results.json（全結果）
    - results/hyperparameter_search/summary.csv（サマリー）
    - results/hyperparameter_search/exp_*.json（個別結果、クラッシュ対策）
"""

import sys
import os
import json
import csv
import time
import torch
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ResidualConfig
from src.models import LLM
from src.trainers.phase1 import MemoryPhase1Trainer
from src.trainers.phase2 import Phase2Trainer
from src.evaluation import analyze_fixed_points


# ============================================================================
# 実験パラメータ（ここを変更して実験をカスタマイズ）
# ============================================================================

# 探索するハイパーパラメータ
NUM_INPUT_TOKENS_LIST = [1, 2, 3]       # 入力トークン数
CONTEXT_MULTIPLIER_LIST = [1, 2]        # context_dim = embed_dim × この値
SAMPLE_SIZES = [500, 1000, 2000, 5000]  # 訓練サンプル数

# 固定パラメータ
NUM_LAYERS = 6                          # レイヤー数（先行実験の結果で決定）
VAL_SAMPLES = 50                        # 検証サンプル数

# 訓練パラメータ
PHASE1_MAX_ITERATIONS = 10
PHASE1_LEARNING_RATE = 0.002
DIST_REG_WEIGHT = 0.5
PHASE2_EPOCHS = 10
PHASE2_BATCH_SIZE = 512
PHASE2_PATIENCE = 2

# 実験制御
SKIP_EXISTING = True                    # 既存結果がある場合スキップ
OUTPUT_DIR = "./results/hyperparameter_search"

# ============================================================================


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


def get_experiment_id(exp_config: Dict) -> str:
    """実験IDを生成"""
    return f"nit{exp_config['num_input_tokens']}_cm{exp_config['context_multiplier']}_s{exp_config['num_samples']}_l{exp_config['num_layers']}"


def get_result_path(exp_id: str) -> str:
    """個別結果ファイルのパスを取得"""
    return os.path.join(OUTPUT_DIR, f"exp_{exp_id}.json")


def result_exists(exp_id: str) -> bool:
    """既存結果があるかチェック"""
    return os.path.exists(get_result_path(exp_id))


def save_result(result: Dict, exp_id: str):
    """個別結果を保存"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(get_result_path(exp_id), 'w') as f:
        json.dump(result, f, indent=2)


def load_existing_results() -> List[Dict]:
    """既存の結果をロード"""
    results = []
    if os.path.exists(OUTPUT_DIR):
        for filename in os.listdir(OUTPUT_DIR):
            if filename.startswith("exp_") and filename.endswith(".json"):
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, 'r') as f:
                    results.append(json.load(f))
    return results


class DataLoader:
    """データローダー（1回のロードでtrain/valを分離）"""

    def __init__(self, device: torch.device):
        self.device = device
        self.sample_tokens = []
        self.sample_boundaries = []
        self.all_tokens = None

    def load_all(self, total_samples: int):
        """全サンプルをロード（train + val用）"""
        print_flush(f"\n  Loading {total_samples} samples from UltraChat...")

        from transformers import GPT2Tokenizer
        from datasets import load_dataset

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

        self.sample_tokens = []
        self.sample_boundaries = [0]

        for i in range(total_samples):
            text = dataset[i]["messages"][0]["content"]
            tokens = tokenizer(text, return_tensors="pt")
            self.sample_tokens.append(tokens["input_ids"].squeeze(0))
            self.sample_boundaries.append(
                self.sample_boundaries[-1] + len(self.sample_tokens[-1])
            )

        self.all_tokens = torch.cat(self.sample_tokens).to(self.device)
        print_flush(f"  Total tokens: {len(self.all_tokens):,}")
        print_flush(f"  Samples loaded: {total_samples}")

    def get_train(self, num_samples: int) -> torch.Tensor:
        """訓練データを取得（サンプル0〜num_samples-1）"""
        end_idx = self.sample_boundaries[num_samples]
        return self.all_tokens[:end_idx]

    def get_val(self, train_samples: int, val_samples: int = VAL_SAMPLES) -> torch.Tensor:
        """検証データを取得（訓練データの直後）"""
        start_idx = self.sample_boundaries[train_samples]
        end_idx = self.sample_boundaries[train_samples + val_samples]
        return self.all_tokens[start_idx:end_idx]


def run_single_experiment(
    exp_config: Dict,
    data_loader: DataLoader,
    device: torch.device,
    max_train_samples: int
) -> Dict:
    """単一実験（Phase1 + Phase2）を実行"""

    exp_id = get_experiment_id(exp_config)
    experiment_start = time.time()

    print_flush(f"\n  Experiment: {exp_id}")
    print_flush(f"    num_input_tokens={exp_config['num_input_tokens']}")
    print_flush(f"    context_multiplier={exp_config['context_multiplier']}")
    print_flush(f"    num_samples={exp_config['num_samples']}")
    print_flush(f"    num_layers={exp_config['num_layers']}")

    set_seed(42)

    # config から設定を取得
    base_config = ResidualConfig()
    embed_dim = base_config.embed_dim
    vocab_size = base_config.vocab_size

    # context_dim を計算
    context_dim = embed_dim * exp_config['context_multiplier']

    # モデル初期化
    model = LLM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        context_dim=context_dim,
        num_layers=exp_config['num_layers'],
        num_input_tokens=exp_config['num_input_tokens'],
        use_pretrained_embeddings=True
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print_flush(f"    Model params: {total_params:,}")

    # データ取得
    train_tokens = data_loader.get_train(exp_config['num_samples'])
    val_tokens = data_loader.get_val(max_train_samples)

    num_train_tokens = len(train_tokens)
    num_val_tokens = len(val_tokens)

    print_flush(f"    Train tokens: {num_train_tokens:,}")
    print_flush(f"    Val tokens: {num_val_tokens:,}")

    result = {
        "experiment_id": exp_id,
        "num_input_tokens": exp_config['num_input_tokens'],
        "context_multiplier": exp_config['context_multiplier'],
        "context_dim": context_dim,
        "num_samples": exp_config['num_samples'],
        "num_layers": exp_config['num_layers'],
        "num_train_tokens": num_train_tokens,
        "num_val_tokens": num_val_tokens,
        "total_params": total_params,
    }

    # ========== Phase 1 ==========
    print_flush(f"    Phase 1 starting...")
    phase1_start = time.time()

    # Config を一時的に変更
    config = ResidualConfig()
    config.context_dim = context_dim
    config.num_input_tokens = exp_config['num_input_tokens']
    config.phase1_max_iterations = PHASE1_MAX_ITERATIONS
    config.phase1_learning_rate = PHASE1_LEARNING_RATE
    config.dist_reg_weight = DIST_REG_WEIGHT

    trainer1 = MemoryPhase1Trainer(model, config, device)
    train_contexts = trainer1.train(train_tokens, label="Train")

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
        "train_effective_rank_percent": train_analysis["effective_rank"] / context_dim * 100,
        "val_effective_rank": val_analysis["effective_rank"],
        "val_effective_rank_percent": val_analysis["effective_rank"] / context_dim * 100,
    })

    print_flush(f"    Phase 1 done: {format_time(phase1_time)}, Train ER={result['train_effective_rank_percent']:.1f}%")

    # ========== Phase 2 ==========
    print_flush(f"    Phase 2 starting...")
    phase2_start = time.time()

    config.phase2_epochs = PHASE2_EPOCHS
    config.phase2_batch_size = PHASE2_BATCH_SIZE
    config.phase2_patience = PHASE2_PATIENCE

    trainer2 = Phase2Trainer(model, config)
    history = trainer2.train_full(
        train_tokens, val_tokens, device,
        epochs=PHASE2_EPOCHS,
        patience=PHASE2_PATIENCE,
        batch_size=PHASE2_BATCH_SIZE
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
    result["timestamp"] = datetime.now().isoformat()

    print_flush(f"    RESULT: Val PPL={result['val_ppl']:.2f}, Val Acc={result['val_accuracy']*100:.2f}% ({format_time(experiment_time)})")

    # クリーンアップ
    del model, trainer1, trainer2, train_contexts, train_tokens, val_tokens, val_result
    import gc
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return result


def analyze_scaling_law(results: List[Dict]):
    """スケーリング則を分析"""
    print_flush(f"\n{'='*70}")
    print_flush("SCALING LAW ANALYSIS")
    print_flush(f"{'='*70}\n")

    # 各 (num_input_tokens, context_multiplier) 組み合わせごとに分析
    for nit in NUM_INPUT_TOKENS_LIST:
        for cm in CONTEXT_MULTIPLIER_LIST:
            subset = [r for r in results
                      if r["num_input_tokens"] == nit
                      and r["context_multiplier"] == cm]

            if len(subset) < 2:
                continue

            # データ抽出
            tokens = np.array([r["num_train_tokens"] for r in subset])
            ppls = np.array([r["val_ppl"] for r in subset])

            # log-log 線形回帰: log(ppl) = slope * log(tokens) + intercept
            log_tokens = np.log(tokens)
            log_ppls = np.log(ppls)

            # 最小二乗法
            n = len(log_tokens)
            sum_x = np.sum(log_tokens)
            sum_y = np.sum(log_ppls)
            sum_xy = np.sum(log_tokens * log_ppls)
            sum_xx = np.sum(log_tokens * log_tokens)

            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n

            # 係数
            A = np.exp(intercept)

            print_flush(f"nit={nit}, cm={cm}:")
            print_flush(f"  PPL = {A:.2f} × tokens^({slope:.4f})")
            print_flush(f"  Data points: {len(subset)}")

            # 各データポイント
            for r in sorted(subset, key=lambda x: x["num_train_tokens"]):
                predicted_ppl = A * (r["num_train_tokens"] ** slope)
                error = abs(r["val_ppl"] - predicted_ppl) / r["val_ppl"] * 100
                print_flush(f"    tokens={r['num_train_tokens']:,}: actual={r['val_ppl']:.2f}, pred={predicted_ppl:.2f}, err={error:.1f}%")
            print_flush("")


def save_all_results(results: List[Dict]):
    """全結果を保存"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # JSON保存
    json_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(json_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": {
                "num_input_tokens_list": NUM_INPUT_TOKENS_LIST,
                "context_multiplier_list": CONTEXT_MULTIPLIER_LIST,
                "sample_sizes": SAMPLE_SIZES,
                "num_layers": NUM_LAYERS,
                "val_samples": VAL_SAMPLES,
            },
            "results": results
        }, f, indent=2)
    print_flush(f"\nAll results saved to: {json_path}")

    # CSV保存
    csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "num_input_tokens", "context_multiplier", "num_samples",
            "num_train_tokens", "val_ppl", "val_accuracy",
            "train_er_percent", "val_er_percent", "total_time_sec"
        ])
        for r in sorted(results, key=lambda x: (x["num_input_tokens"], x["context_multiplier"], x["num_samples"])):
            writer.writerow([
                r["num_input_tokens"],
                r["context_multiplier"],
                r["num_samples"],
                r["num_train_tokens"],
                f"{r['val_ppl']:.2f}",
                f"{r['val_accuracy']:.4f}",
                f"{r['train_effective_rank_percent']:.1f}",
                f"{r['val_effective_rank_percent']:.1f}",
                f"{r['total_time_sec']:.1f}"
            ])
    print_flush(f"Summary CSV saved to: {csv_path}")


def print_summary_table(results: List[Dict]):
    """結果サマリーを表示"""
    print_flush(f"\n{'='*120}")
    print_flush("RESULTS SUMMARY")
    print_flush(f"{'='*120}")
    print_flush(f"{'NIT':>4} | {'CM':>3} | {'Samples':>7} | {'Tokens':>10} | {'Val PPL':>10} | {'Val Acc%':>10} | {'Train ER%':>10} | {'Time':>8}")
    print_flush(f"{'-'*4}-+-{'-'*3}-+-{'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    for r in sorted(results, key=lambda x: (x["num_input_tokens"], x["context_multiplier"], x["num_samples"])):
        time_str = format_time(r['total_time_sec'])
        print_flush(
            f"{r['num_input_tokens']:>4} | "
            f"{r['context_multiplier']:>3} | "
            f"{r['num_samples']:>7} | "
            f"{r['num_train_tokens']:>10,} | "
            f"{r['val_ppl']:>10.2f} | "
            f"{r['val_accuracy']*100:>9.2f}% | "
            f"{r['train_effective_rank_percent']:>9.1f}% | "
            f"{time_str:>8}"
        )


def main():
    total_start = time.time()

    print_flush(f"\n{'='*70}")
    print_flush("HYPERPARAMETER SEARCH EXPERIMENT")
    print_flush(f"{'='*70}")
    print_flush(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # パラメータ表示
    print_flush(f"\nSearch space:")
    print_flush(f"  num_input_tokens: {NUM_INPUT_TOKENS_LIST}")
    print_flush(f"  context_multiplier: {CONTEXT_MULTIPLIER_LIST}")
    print_flush(f"  sample_sizes: {SAMPLE_SIZES}")
    print_flush(f"  num_layers: {NUM_LAYERS} (fixed)")

    total_experiments = len(NUM_INPUT_TOKENS_LIST) * len(CONTEXT_MULTIPLIER_LIST) * len(SAMPLE_SIZES)
    print_flush(f"\nTotal experiments: {total_experiments}")

    # GPU確認
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_flush(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = torch.device('cpu')
        print_flush("WARNING: Running on CPU (very slow)")

    # パラメータグリッド生成
    experiments = []
    for num_input_tokens in NUM_INPUT_TOKENS_LIST:
        for context_multiplier in CONTEXT_MULTIPLIER_LIST:
            for num_samples in SAMPLE_SIZES:
                experiments.append({
                    "num_input_tokens": num_input_tokens,
                    "context_multiplier": context_multiplier,
                    "num_samples": num_samples,
                    "num_layers": NUM_LAYERS
                })

    # 既存結果をロード
    existing_results = load_existing_results() if SKIP_EXISTING else []
    existing_ids = {r["experiment_id"] for r in existing_results}

    if existing_results:
        print_flush(f"\nFound {len(existing_results)} existing results")

    # データローダー初期化
    max_samples = max(SAMPLE_SIZES)
    total_samples = max_samples + VAL_SAMPLES

    print_flush(f"\n{'='*70}")
    print_flush("Loading all data...")
    print_flush(f"{'='*70}")
    data_loader = DataLoader(device)
    data_loader.load_all(total_samples)

    # 実験実行
    results = list(existing_results)  # 既存結果をコピー
    experiments_to_run = [e for e in experiments if get_experiment_id(e) not in existing_ids]

    print_flush(f"\n{'='*70}")
    print_flush(f"Running {len(experiments_to_run)} experiments (skipping {len(existing_ids)} existing)")
    print_flush(f"{'='*70}")

    for idx, exp_config in enumerate(experiments_to_run, 1):
        exp_id = get_experiment_id(exp_config)
        print_flush(f"\n[{idx}/{len(experiments_to_run)}] Running {exp_id}...")

        try:
            result = run_single_experiment(exp_config, data_loader, device, max_samples)
            results.append(result)

            # 中間保存
            save_result(result, exp_id)

            # 進捗表示
            elapsed = time.time() - total_start
            avg_time = elapsed / idx
            remaining = avg_time * (len(experiments_to_run) - idx)
            print_flush(f"  Progress: {idx}/{len(experiments_to_run)}, Remaining: {format_time(remaining)}")

        except Exception as e:
            print_flush(f"ERROR in {exp_id}: {e}")
            import traceback
            traceback.print_exc()

            # エラー時もメモリクリーンアップ
            import gc
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    # 最終結果
    total_time = time.time() - total_start

    print_flush(f"\n{'='*70}")
    print_flush("ALL EXPERIMENTS COMPLETE")
    print_flush(f"{'='*70}")
    print_flush(f"Total time: {format_time(total_time)}")
    print_flush(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 結果保存
    save_all_results(results)

    # サマリー表示
    print_summary_table(results)

    # スケーリング則分析
    analyze_scaling_law(results)


if __name__ == "__main__":
    main()
