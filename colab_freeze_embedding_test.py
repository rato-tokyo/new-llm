#!/usr/bin/env python3
"""
Embedding凍結実験スクリプト

過去の実験（Embedding学習）との比較用
- 500サンプル、1000サンプルで比較
- Phase 2: Embedding凍結（TokenBlockのみ学習）

使用方法:
    python colab_freeze_embedding_test.py
"""

import os
import sys
import time
import json
import torch
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ResidualConfig
from src.models.llm import LLM
from src.trainers.phase1.memory import MemoryPhase1Trainer
from src.trainers.phase2 import Phase2Trainer
from src.evaluation.metrics import analyze_fixed_points


# ========== 実験設定 ==========
SAMPLE_SIZES = [500, 1000]  # 過去実験と同じ
VAL_SAMPLES = 50
NUM_LAYERS = 6
FREEZE_EMBEDDING = True  # Embedding凍結を有効化


def load_ultrachat_data(num_samples: int, device: str):
    """UltraChatデータをロード"""
    from transformers import GPT2Tokenizer
    from datasets import load_dataset

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading {num_samples} samples from UltraChat...")
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    all_tokens = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        messages = sample.get('messages', [])
        text = ' '.join([m.get('content', '') for m in messages])
        tokens = tokenizer.encode(text, add_special_tokens=True, max_length=512, truncation=True)
        all_tokens.extend(tokens)

    token_ids = torch.tensor(all_tokens, dtype=torch.long, device=device)
    print(f"  Total tokens: {len(token_ids):,}")
    return token_ids


def run_single_experiment(num_samples: int, config, device: str) -> dict:
    """単一実験を実行"""
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {num_samples} samples, freeze_embedding={FREEZE_EMBEDDING}")
    print(f"{'='*70}")

    start_time = time.time()

    # データロード
    total_samples = num_samples + VAL_SAMPLES
    all_tokens = load_ultrachat_data(total_samples, device)

    # 訓練/検証分割（サンプル単位ではなくトークン単位で分割）
    train_ratio = num_samples / total_samples
    split_idx = int(len(all_tokens) * train_ratio)
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]

    print(f"  Train tokens: {len(train_tokens):,}")
    print(f"  Val tokens: {len(val_tokens):,}")

    # モデル初期化
    model = LLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=config.context_dim,
        num_layers=NUM_LAYERS,
        num_input_tokens=config.num_input_tokens,
        use_pretrained_embeddings=True,
        use_weight_tying=config.use_weight_tying
    ).to(device)

    # Phase 1
    print(f"\n  Phase 1 starting...")
    phase1_start = time.time()

    trainer1 = MemoryPhase1Trainer(model, config, torch.device(device))
    train_contexts = trainer1.train(train_tokens, label="Train")

    phase1_time = time.time() - phase1_start

    # Effective Rank計算
    metrics = analyze_fixed_points(train_contexts, label="Train", verbose=False)
    train_er = metrics['effective_rank'] / config.context_dim * 100
    print(f"  Phase 1 done: {phase1_time/60:.1f}min, Train ER={train_er:.1f}%")

    # Phase 2（Embedding凍結オプション）
    print(f"  Phase 2 starting (freeze_embedding={FREEZE_EMBEDDING})...")
    phase2_start = time.time()

    # config に freeze_embedding を設定
    config.phase2_freeze_embedding = FREEZE_EMBEDDING

    trainer2 = Phase2Trainer(model, config)
    history = trainer2.train_full(
        train_token_ids=train_tokens,
        val_token_ids=val_tokens,
        device=device,
        epochs=config.phase2_epochs
    )

    phase2_time = time.time() - phase2_start
    total_time = time.time() - start_time

    # historyから結果を取得
    best_epoch = history['best_epoch']
    result = {
        'num_samples': num_samples,
        'num_layers': NUM_LAYERS,
        'freeze_embedding': FREEZE_EMBEDDING,
        'num_train_tokens': len(train_tokens),
        'num_val_tokens': len(val_tokens),
        'val_ppl': history['val_ppl'][best_epoch - 1],
        'val_accuracy': history['val_acc'][best_epoch - 1],
        'val_loss': history['val_loss'][best_epoch - 1],
        'best_epoch': best_epoch,
        'train_effective_rank_percent': train_er,
        'phase1_time_sec': phase1_time,
        'phase2_time_sec': phase2_time,
        'total_time_sec': total_time,
        'timestamp': datetime.now().isoformat()
    }

    print(f"\n  RESULT: Val PPL={result['val_ppl']:.2f}, Val Acc={result['val_accuracy']:.2%}")
    print(f"  Total time: {total_time/60:.1f}min")

    return result


def main():
    print("=" * 70)
    print("EMBEDDING FREEZE EXPERIMENT")
    print(f"Compare with previous results (Embedding trained)")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # デバイス
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")
    else:
        print("WARNING: Running on CPU")

    # 設定
    config = ResidualConfig()

    # 結果保存
    results = []
    os.makedirs("./results", exist_ok=True)

    # 実験実行
    for num_samples in SAMPLE_SIZES:
        try:
            result = run_single_experiment(num_samples, config, device)
            results.append(result)

            # 中間保存
            with open(f"./results/freeze_embed_{num_samples}.json", 'w') as f:
                json.dump(result, f, indent=2)

            # メモリクリア
            torch.cuda.empty_cache() if device == "cuda" else None

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 比較表を出力
    print("\n" + "=" * 70)
    print("COMPARISON: Embedding Frozen vs Embedding Trained")
    print("=" * 70)

    # 過去の結果（Embedding学習）
    previous_results = {
        500: {'val_ppl': 1189.15, 'val_acc': 0.1158},
        1000: {'val_ppl': 840.46, 'val_acc': 0.1303}
    }

    print(f"\n{'Samples':>8} | {'Embed Trained':>15} | {'Embed Frozen':>15} | {'Improvement':>12}")
    print("-" * 60)

    for r in results:
        samples = r['num_samples']
        if samples in previous_results:
            prev_ppl = previous_results[samples]['val_ppl']
            new_ppl = r['val_ppl']
            improvement = (prev_ppl - new_ppl) / prev_ppl * 100
            print(f"{samples:>8} | {prev_ppl:>12.2f} PPL | {new_ppl:>12.2f} PPL | {improvement:>+10.1f}%")

    # 全結果保存
    with open("./results/freeze_embedding_experiment.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to ./results/freeze_embedding_experiment.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
