#!/usr/bin/env python3
"""
統一設定でのスケーリング実験スクリプト

目的:
- 11/27と11/28の実験条件の違い（トークン化設定）を解消
- 統一した設定（truncation=False）で複数サンプル数での実験を実施
- 正確なα値を導出

使用方法:
  Colab: !python scripts/unified_scaling_experiment.py
  Local: python3 scripts/unified_scaling_experiment.py

設定:
- トークン化: truncation=False（全長使用）
- サンプル数: [50, 100, 200, 500, 1000]
- num_input_tokens: 1
- モデル: 6層/768dim
"""

import sys
import os
import json
import time
from datetime import datetime

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy import stats

# 設定
SAMPLE_SIZES = [50, 100, 200, 500, 1000]
NUM_LAYERS = 6
CONTEXT_DIM = 768
NUM_INPUT_TOKENS = 1
EMBEDDING_FREEZE = True


def print_flush(msg):
    print(msg, flush=True)


def run_experiment(num_samples: int, train_token_ids: torch.Tensor, val_token_ids: torch.Tensor, device: torch.device):
    """単一サンプル数での実験を実行"""
    from config import ResidualConfig
    from src.models.llm import NewLLM
    from src.trainers.phase1.memory import MemoryPhase1Trainer
    from src.trainers.phase2 import Phase2Trainer

    print_flush(f"\n{'='*70}")
    print_flush(f"Experiment: {num_samples} samples")
    print_flush(f"{'='*70}")

    # 設定を作成
    config = ResidualConfig()
    config.num_layers = NUM_LAYERS
    config.context_dim = CONTEXT_DIM
    config.num_input_tokens = NUM_INPUT_TOKENS
    config.phase2_freeze_embedding = EMBEDDING_FREEZE
    config.num_samples = num_samples

    # トークン数を計算（サンプル数に応じて調整）
    # 全データからnum_samples分を抽出
    # 簡易的に: トークン数 = 全トークン数 * (num_samples / max_samples)
    max_samples = max(SAMPLE_SIZES)
    ratio = num_samples / max_samples
    train_tokens = int(len(train_token_ids) * ratio)

    current_train_ids = train_token_ids[:train_tokens].to(device)
    current_val_ids = val_token_ids.to(device)

    print_flush(f"  Train tokens: {len(current_train_ids):,}")
    print_flush(f"  Val tokens: {len(current_val_ids):,}")

    # モデル作成
    model = NewLLM(config).to(device)

    # Phase 1
    print_flush(f"\n  Phase 1 starting...")
    phase1_start = time.time()

    trainer1 = MemoryPhase1Trainer(model, config)
    trainer1.train(current_train_ids, device)

    phase1_time = time.time() - phase1_start
    print_flush(f"  Phase 1 done: {phase1_time/60:.1f}min")

    # Phase 2
    print_flush(f"\n  Phase 2 starting...")
    phase2_start = time.time()

    trainer2 = Phase2Trainer(model, config)
    trainer2.freeze_for_phase2()

    best_val_loss = float('inf')
    best_val_ppl = float('inf')
    best_val_acc = 0.0

    for epoch in range(config.phase2_epochs):
        train_loss = trainer2.train_epoch(current_train_ids, device)
        val_loss, val_ppl, val_acc = trainer2.evaluate(current_val_ids, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_ppl = val_ppl
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.phase2_patience:
                print_flush(f"  Early stopping at epoch {epoch+1}")
                break

    phase2_time = time.time() - phase2_start
    print_flush(f"  Phase 2 done: {phase2_time/60:.1f}min")
    print_flush(f"  Val PPL: {best_val_ppl:.2f}, Val Acc: {best_val_acc*100:.2f}%")

    return {
        'num_samples': num_samples,
        'train_tokens': len(current_train_ids),
        'val_tokens': len(current_val_ids),
        'val_ppl': best_val_ppl,
        'val_acc': best_val_acc,
        'phase1_time': phase1_time,
        'phase2_time': phase2_time,
    }


def calculate_scaling_law(results: list):
    """スケーリング則を計算"""
    tokens = np.array([r['train_tokens'] for r in results])
    ppl = np.array([r['val_ppl'] for r in results])

    # 対数変換
    log_tokens = np.log(tokens)
    log_ppl = np.log(ppl)

    # 線形回帰
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_tokens, log_ppl)

    alpha = slope  # 負の値
    A = np.exp(intercept)
    r_squared = r_value ** 2

    return {
        'alpha': alpha,
        'A': A,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_err': std_err,
    }


def main():
    print_flush("="*70)
    print_flush("UNIFIED SCALING EXPERIMENT")
    print_flush("="*70)
    print_flush(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_flush(f"\nSettings:")
    print_flush(f"  Sample sizes: {SAMPLE_SIZES}")
    print_flush(f"  Model: {NUM_LAYERS} layers, {CONTEXT_DIM} dim")
    print_flush(f"  num_input_tokens: {NUM_INPUT_TOKENS}")
    print_flush(f"  Embedding freeze: {EMBEDDING_FREEZE}")
    print_flush(f"  Tokenization: truncation=False (full length)")

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print_flush(f"\nGPU: {gpu_name} ({gpu_mem:.1f}GB)")
    else:
        print_flush("\nUsing CPU")

    # データロード
    print_flush("\n" + "="*70)
    print_flush("Loading data...")
    print_flush("="*70)

    from config import ResidualConfig
    from src.providers.data.memory import MemoryDataProvider

    config = ResidualConfig()
    config.num_samples = max(SAMPLE_SIZES)

    provider = MemoryDataProvider(config)
    train_token_ids, val_token_ids = provider.load_data()

    print_flush(f"  Total train tokens: {len(train_token_ids):,}")
    print_flush(f"  Total val tokens: {len(val_token_ids):,}")
    print_flush(f"  Tokens per sample: {len(train_token_ids) / max(SAMPLE_SIZES):.1f}")

    # 実験実行
    print_flush("\n" + "="*70)
    print_flush("Running experiments...")
    print_flush("="*70)

    results = []
    total_start = time.time()

    for i, num_samples in enumerate(SAMPLE_SIZES):
        print_flush(f"\n[{i+1}/{len(SAMPLE_SIZES)}] {num_samples} samples")
        result = run_experiment(num_samples, train_token_ids, val_token_ids, device)
        results.append(result)

    total_time = time.time() - total_start

    # スケーリング則計算
    print_flush("\n" + "="*70)
    print_flush("SCALING LAW ANALYSIS")
    print_flush("="*70)

    scaling = calculate_scaling_law(results)

    print_flush(f"\nPPL = {scaling['A']:.2f} × tokens^({scaling['alpha']:.4f})")
    print_flush(f"α = {scaling['alpha']:.4f}")
    print_flush(f"R² = {scaling['r_squared']:.4f}")
    print_flush(f"p-value = {scaling['p_value']:.6f}")

    # 結果表示
    print_flush("\n" + "="*70)
    print_flush("RESULTS SUMMARY")
    print_flush("="*70)

    print_flush(f"\n{'Samples':>8} | {'Tokens':>10} | {'Val PPL':>10} | {'Val Acc':>10}")
    print_flush("-" * 50)
    for r in results:
        print_flush(f"{r['num_samples']:>8} | {r['train_tokens']:>10,} | {r['val_ppl']:>10.2f} | {r['val_acc']*100:>9.2f}%")

    print_flush(f"\nTotal time: {total_time/60:.1f} minutes")

    # 結果保存
    output_dir = './results/unified_scaling'
    os.makedirs(output_dir, exist_ok=True)

    output = {
        'settings': {
            'sample_sizes': SAMPLE_SIZES,
            'num_layers': NUM_LAYERS,
            'context_dim': CONTEXT_DIM,
            'num_input_tokens': NUM_INPUT_TOKENS,
            'embedding_freeze': EMBEDDING_FREEZE,
            'tokenization': 'truncation=False (full length)',
        },
        'results': results,
        'scaling_law': scaling,
        'total_time_minutes': total_time / 60,
        'timestamp': datetime.now().isoformat(),
    }

    output_file = os.path.join(output_dir, 'results.json')
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print_flush(f"\nResults saved to: {output_file}")

    # 11/27実験との比較予測
    print_flush("\n" + "="*70)
    print_flush("COMPARISON WITH 11/27 EXPERIMENT")
    print_flush("="*70)

    print_flush("\n11/27 experiment (max_length=128): α = -0.7463")
    print_flush(f"This experiment (truncation=False): α = {scaling['alpha']:.4f}")
    print_flush(f"Difference: {(scaling['alpha'] - (-0.7463)) / (-0.7463) * 100:.1f}%")


if __name__ == '__main__':
    main()
