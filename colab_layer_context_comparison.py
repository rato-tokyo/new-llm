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
import json
import time
import torch
import random
import numpy as np
from datetime import datetime
from typing import List, Dict

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ResidualConfig
from src.models import LLM
from src.trainers.phase1 import MemoryPhase1Trainer
from src.trainers.phase2 import Phase2Trainer
from src.evaluation import analyze_fixed_points


# ============================================================================
# 実験パラメータ
# ============================================================================

# 比較実験設定
EXPERIMENTS = [
    {"name": "A", "num_layers": 6, "context_multiplier": 1, "description": "Deep & Narrow (6層, 768dim)"},
    {"name": "B", "num_layers": 3, "context_multiplier": 2, "description": "Shallow & Wide (3層, 1536dim)"},
]

# サンプル数（複数でスケーリング確認）
SAMPLE_SIZES = [500, 1000]

# 固定パラメータ（config.pyから取得）
_base_config = ResidualConfig()
VAL_SAMPLES = 50
EMBED_DIM = _base_config.embed_dim
NUM_INPUT_TOKENS = _base_config.num_input_tokens
VOCAB_SIZE = _base_config.vocab_size

# 訓練パラメータ（config.pyから取得）
PHASE1_MAX_ITERATIONS = _base_config.phase1_max_iterations
PHASE1_LEARNING_RATE = _base_config.phase1_learning_rate
DIST_REG_WEIGHT = _base_config.dist_reg_weight
PHASE2_EPOCHS = _base_config.phase2_epochs
PHASE2_BATCH_SIZE = _base_config.phase2_batch_size
PHASE2_PATIENCE = _base_config.phase2_patience
PHASE2_FREEZE_EMBEDDING = _base_config.phase2_freeze_embedding

# 出力設定
OUTPUT_DIR = "./results/layer_context_comparison"
SKIP_EXISTING = True

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


def format_number(n: int) -> str:
    """数値を読みやすい形式に変換"""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    elif n >= 1_000:
        return f"{n / 1_000:.2f}K"
    else:
        return str(n)


def get_experiment_id(exp_name: str, num_samples: int) -> str:
    """実験IDを生成"""
    return f"exp{exp_name}_s{num_samples}"


def get_result_path(exp_id: str) -> str:
    """個別結果ファイルのパスを取得"""
    return os.path.join(OUTPUT_DIR, f"{exp_id}.json")


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
            if filename.startswith("exp") and filename.endswith(".json"):
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, 'r') as f:
                    results.append(json.load(f))
    return results


class DataLoader:
    """データローダー"""

    def __init__(self, device: torch.device):
        self.device = device
        self.sample_tokens = []
        self.sample_boundaries = []
        self.all_tokens = None

    def load_all(self, total_samples: int):
        """全サンプルをロード"""
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

    def get_train(self, num_samples: int) -> torch.Tensor:
        """訓練データを取得"""
        end_idx = self.sample_boundaries[num_samples]
        return self.all_tokens[:end_idx]

    def get_val(self, train_samples: int, val_samples: int = VAL_SAMPLES) -> torch.Tensor:
        """検証データを取得"""
        start_idx = self.sample_boundaries[train_samples]
        end_idx = self.sample_boundaries[train_samples + val_samples]
        return self.all_tokens[start_idx:end_idx]


def calculate_params(num_layers: int, context_dim: int, embed_dim: int, num_input_tokens: int) -> Dict:
    """パラメータ数を計算"""
    input_dim = context_dim + embed_dim * num_input_tokens

    # ContextBlock
    context_linear = input_dim * context_dim + context_dim
    context_layernorm = context_dim * 2
    context_layer_params = context_linear + context_layernorm
    context_block_total = context_layer_params * num_layers

    # TokenBlock
    token_linear = input_dim * embed_dim + embed_dim
    token_layernorm = embed_dim * 2
    token_layer_params = token_linear + token_layernorm
    token_block_total = token_layer_params * num_layers

    # Embedding (凍結)
    embedding = VOCAB_SIZE * embed_dim

    # 合計
    total = embedding + context_block_total + token_block_total
    trainable_phase1 = context_block_total
    trainable_phase2 = token_block_total  # Embedding凍結時

    return {
        "total": total,
        "context_block": context_block_total,
        "token_block": token_block_total,
        "embedding": embedding,
        "trainable_phase1": trainable_phase1,
        "trainable_phase2": trainable_phase2,
    }


def run_single_experiment(
    exp_config: Dict,
    num_samples: int,
    data_loader: DataLoader,
    device: torch.device,
    max_train_samples: int
) -> Dict:
    """単一実験を実行"""

    exp_id = get_experiment_id(exp_config["name"], num_samples)
    experiment_start = time.time()

    context_dim = EMBED_DIM * exp_config["context_multiplier"]
    num_layers = exp_config["num_layers"]

    print_flush(f"\n  Experiment {exp_config['name']}: {exp_config['description']}")
    print_flush(f"    num_layers={num_layers}, context_dim={context_dim}")
    print_flush(f"    num_samples={num_samples}")

    set_seed(42)

    # パラメータ数計算
    params_info = calculate_params(num_layers, context_dim, EMBED_DIM, NUM_INPUT_TOKENS)
    print_flush(f"    ContextBlock params: {format_number(params_info['context_block'])}")
    print_flush(f"    TokenBlock params: {format_number(params_info['token_block'])}")
    print_flush(f"    Phase 1 学習対象: {format_number(params_info['trainable_phase1'])}")
    print_flush(f"    Phase 2 学習対象: {format_number(params_info['trainable_phase2'])}")

    # モデル初期化
    model = LLM(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        context_dim=context_dim,
        num_layers=num_layers,
        num_input_tokens=NUM_INPUT_TOKENS,
        use_pretrained_embeddings=True,
        use_weight_tying=True
    )
    model.to(device)

    # データ取得
    train_tokens = data_loader.get_train(num_samples)
    val_tokens = data_loader.get_val(max_train_samples)

    num_train_tokens = len(train_tokens)
    num_val_tokens = len(val_tokens)

    print_flush(f"    Train tokens: {num_train_tokens:,}")
    print_flush(f"    Val tokens: {num_val_tokens:,}")

    result = {
        "experiment_id": exp_id,
        "experiment_name": exp_config["name"],
        "description": exp_config["description"],
        "num_layers": num_layers,
        "context_multiplier": exp_config["context_multiplier"],
        "context_dim": context_dim,
        "num_samples": num_samples,
        "num_train_tokens": num_train_tokens,
        "num_val_tokens": num_val_tokens,
        "context_block_params": params_info["context_block"],
        "token_block_params": params_info["token_block"],
        "trainable_phase1": params_info["trainable_phase1"],
        "trainable_phase2": params_info["trainable_phase2"],
    }

    # ========== Phase 1 ==========
    print_flush(f"    Phase 1 starting...")
    phase1_start = time.time()

    config = ResidualConfig()
    config.context_dim = context_dim
    config.num_layers = num_layers
    config.num_input_tokens = NUM_INPUT_TOKENS
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

    print_flush(f"    Phase 1 done: {format_time(phase1_time)}")
    print_flush(f"      Train ER: {result['train_effective_rank']:.1f}/{context_dim} ({result['train_effective_rank_percent']:.1f}%)")
    print_flush(f"      Val ER: {result['val_effective_rank']:.1f}/{context_dim} ({result['val_effective_rank_percent']:.1f}%)")

    # ========== Phase 2 ==========
    print_flush(f"    Phase 2 starting (freeze_embedding={PHASE2_FREEZE_EMBEDDING})...")
    phase2_start = time.time()

    config.phase2_epochs = PHASE2_EPOCHS
    config.phase2_batch_size = PHASE2_BATCH_SIZE
    config.phase2_patience = PHASE2_PATIENCE
    config.phase2_freeze_embedding = PHASE2_FREEZE_EMBEDDING

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

    print_flush(f"    Phase 2 done: {format_time(phase2_time)}")
    print_flush(f"    RESULT: Val PPL={result['val_ppl']:.2f}, Val Acc={result['val_accuracy']*100:.2f}%")

    # クリーンアップ
    del model, trainer1, trainer2, train_contexts, train_tokens, val_tokens, val_result
    import gc
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return result


def save_summary_markdown(results: List[Dict]):
    """サマリーをMarkdownで保存"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    md_path = os.path.join(OUTPUT_DIR, "summary.md")

    with open(md_path, 'w') as f:
        f.write("# レイヤー数 vs コンテキスト次元 比較実験結果\n\n")
        f.write(f"実験日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 実験設定\n\n")
        f.write("| 実験 | レイヤー数 | context_dim | 説明 |\n")
        f.write("|------|-----------|-------------|------|\n")
        for exp in EXPERIMENTS:
            context_dim = EMBED_DIM * exp["context_multiplier"]
            f.write(f"| {exp['name']} | {exp['num_layers']} | {context_dim} | {exp['description']} |\n")

        f.write("\n## パラメータ数比較\n\n")
        f.write("| 実験 | ContextBlock | TokenBlock | Phase 1 学習 | Phase 2 学習 |\n")
        f.write("|------|-------------|------------|-------------|-------------|\n")

        # 各実験の最初の結果からパラメータ数を取得
        for exp in EXPERIMENTS:
            exp_results = [r for r in results if r["experiment_name"] == exp["name"]]
            if exp_results:
                r = exp_results[0]
                f.write(f"| {exp['name']} | {format_number(r['context_block_params'])} | {format_number(r['token_block_params'])} | {format_number(r['trainable_phase1'])} | {format_number(r['trainable_phase2'])} |\n")

        f.write("\n## 結果比較\n\n")
        f.write("| 実験 | Samples | Val PPL | Val Acc | Train ER% | Val ER% | Time |\n")
        f.write("|------|---------|---------|---------|-----------|---------|------|\n")

        for r in sorted(results, key=lambda x: (x["experiment_name"], x["num_samples"])):
            f.write(
                f"| {r['experiment_name']} | "
                f"{r['num_samples']} | "
                f"{r['val_ppl']:.2f} | "
                f"{r['val_accuracy']*100:.2f}% | "
                f"{r['train_effective_rank_percent']:.1f}% | "
                f"{r['val_effective_rank_percent']:.1f}% | "
                f"{format_time(r['total_time_sec'])} |\n"
            )

        # サンプル数ごとの比較
        f.write("\n## サンプル数別 A vs B 比較\n\n")
        for num_samples in SAMPLE_SIZES:
            f.write(f"### {num_samples} samples\n\n")
            exp_a = next((r for r in results if r["experiment_name"] == "A" and r["num_samples"] == num_samples), None)
            exp_b = next((r for r in results if r["experiment_name"] == "B" and r["num_samples"] == num_samples), None)

            if exp_a and exp_b:
                f.write("| 指標 | A (6層, 768dim) | B (3層, 1536dim) | 差分 |\n")
                f.write("|------|-----------------|------------------|------|\n")
                f.write(f"| Val PPL | {exp_a['val_ppl']:.2f} | {exp_b['val_ppl']:.2f} | {exp_b['val_ppl'] - exp_a['val_ppl']:+.2f} |\n")
                f.write(f"| Val Acc | {exp_a['val_accuracy']*100:.2f}% | {exp_b['val_accuracy']*100:.2f}% | {(exp_b['val_accuracy'] - exp_a['val_accuracy'])*100:+.2f}% |\n")
                f.write(f"| Train ER | {exp_a['train_effective_rank']:.1f} | {exp_b['train_effective_rank']:.1f} | - |\n")
                f.write(f"| Val ER | {exp_a['val_effective_rank']:.1f} | {exp_b['val_effective_rank']:.1f} | - |\n")
                f.write(f"| Phase 1学習パラメータ | {format_number(exp_a['trainable_phase1'])} | {format_number(exp_b['trainable_phase1'])} | - |\n")
                f.write(f"| Phase 2学習パラメータ | {format_number(exp_a['trainable_phase2'])} | {format_number(exp_b['trainable_phase2'])} | - |\n")
                f.write("\n")

    print_flush(f"\nSummary saved to: {md_path}")


def print_comparison_table(results: List[Dict]):
    """比較結果を表示"""
    print_flush(f"\n{'='*100}")
    print_flush("COMPARISON RESULTS")
    print_flush(f"{'='*100}")

    print_flush(f"\n{'Exp':>4} | {'Layers':>6} | {'CtxDim':>7} | {'Samples':>7} | {'Val PPL':>10} | {'Val Acc%':>10} | {'Train ER%':>10} | {'Val ER%':>10}")
    print_flush(f"{'-'*4}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for r in sorted(results, key=lambda x: (x["num_samples"], x["experiment_name"])):
        print_flush(
            f"{r['experiment_name']:>4} | "
            f"{r['num_layers']:>6} | "
            f"{r['context_dim']:>7} | "
            f"{r['num_samples']:>7} | "
            f"{r['val_ppl']:>10.2f} | "
            f"{r['val_accuracy']*100:>9.2f}% | "
            f"{r['train_effective_rank_percent']:>9.1f}% | "
            f"{r['val_effective_rank_percent']:>9.1f}%"
        )


def main():
    total_start = time.time()

    print_flush(f"\n{'='*70}")
    print_flush("LAYER vs CONTEXT DIMENSION COMPARISON")
    print_flush(f"{'='*70}")
    print_flush(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 実験設定表示
    print_flush(f"\nExperiments:")
    for exp in EXPERIMENTS:
        context_dim = EMBED_DIM * exp["context_multiplier"]
        print_flush(f"  {exp['name']}: {exp['description']} (context_dim={context_dim})")

    print_flush(f"\nSample sizes: {SAMPLE_SIZES}")
    print_flush(f"Embedding freeze: {PHASE2_FREEZE_EMBEDDING}")

    total_experiments = len(EXPERIMENTS) * len(SAMPLE_SIZES)
    print_flush(f"Total experiments: {total_experiments}")

    # GPU確認
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print_flush(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = torch.device('cpu')
        print_flush("WARNING: Running on CPU")

    # 既存結果をロード
    existing_results = load_existing_results() if SKIP_EXISTING else []
    existing_ids = {r["experiment_id"] for r in existing_results}

    if existing_results:
        print_flush(f"\nFound {len(existing_results)} existing results")

    # データローダー初期化
    max_samples = max(SAMPLE_SIZES)
    total_samples = max_samples + VAL_SAMPLES

    print_flush(f"\n{'='*70}")
    print_flush("Loading data...")
    print_flush(f"{'='*70}")
    data_loader = DataLoader(device)
    data_loader.load_all(total_samples)

    # 実験実行
    results = list(existing_results)

    print_flush(f"\n{'='*70}")
    print_flush("Running experiments...")
    print_flush(f"{'='*70}")

    exp_count = 0
    for exp_config in EXPERIMENTS:
        for num_samples in SAMPLE_SIZES:
            exp_id = get_experiment_id(exp_config["name"], num_samples)

            if SKIP_EXISTING and exp_id in existing_ids:
                print_flush(f"\n  Skipping {exp_id} (already exists)")
                continue

            exp_count += 1
            print_flush(f"\n[{exp_count}/{total_experiments}] Running {exp_id}...")

            try:
                result = run_single_experiment(
                    exp_config, num_samples, data_loader, device, max_samples
                )
                results.append(result)
                save_result(result, exp_id)

            except Exception as e:
                print_flush(f"ERROR in {exp_id}: {e}")
                import traceback
                traceback.print_exc()

                import gc
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()

    # 結果保存
    total_time = time.time() - total_start

    print_flush(f"\n{'='*70}")
    print_flush("ALL EXPERIMENTS COMPLETE")
    print_flush(f"{'='*70}")
    print_flush(f"Total time: {format_time(total_time)}")

    # JSON保存
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(json_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "experiments": EXPERIMENTS,
            "sample_sizes": SAMPLE_SIZES,
            "results": results
        }, f, indent=2)
    print_flush(f"\nResults saved to: {json_path}")

    # Markdown サマリー保存
    save_summary_markdown(results)

    # 比較表表示
    print_comparison_table(results)


if __name__ == "__main__":
    main()
