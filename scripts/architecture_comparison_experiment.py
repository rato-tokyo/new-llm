#!/usr/bin/env python3
"""
アーキテクチャ比較実験スクリプト

3つの設定を比較:
1. Baseline: 6層、768次元、1トークン入力
2. Exp1: 6層、768次元、2トークン入力 (num_input_tokens=2)
3. Exp2: 6層、1152次元、1トークン入力 (context_dim*1.5)
4. Exp3: 9層、768次元、1トークン入力 (num_layers=9)

Colab実行用:
    !cd /content/new-llm && python3 scripts/architecture_comparison_experiment.py

出力: results/architecture_comparison_YYYYMMDD_HHMMSS/
"""

import os
import sys
import json
import time
import random
from datetime import datetime

import numpy as np
import torch

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ResidualConfig
from src.models import LLM
from src.trainers.phase1.memory import MemoryPhase1Trainer
from src.trainers.phase2 import Phase2Trainer
from src.evaluation.metrics import analyze_fixed_points
from src.providers.data import load_data


def print_flush(msg):
    print(msg, flush=True)


def set_seed(seed):
    """再現性のためのシード固定"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_params(num_layers, context_dim, embed_dim, num_input_tokens, vocab_size):
    """パラメータ数計算"""
    token_input_dim = embed_dim * num_input_tokens

    # ContextBlock
    context_block_total = 0
    for i in range(num_layers):
        layer_input_dim = context_dim + token_input_dim
        layer_output_dim = context_dim
        fnn_params = layer_input_dim * layer_output_dim + layer_output_dim
        layernorm_params = layer_output_dim * 2
        context_block_total += fnn_params + layernorm_params

    # TokenBlock
    token_block_total = 0
    for i in range(num_layers):
        token_in = embed_dim
        token_out = embed_dim
        ctx_dim = context_dim
        fnn_params = (ctx_dim + token_in) * token_out + token_out
        layernorm_params = token_out * 2
        token_block_total += fnn_params + layernorm_params

    embedding = vocab_size * embed_dim
    embed_norm = embed_dim * 2
    total = embedding + embed_norm + context_block_total + token_block_total

    return {
        'total': total,
        'context_block': context_block_total,
        'token_block': token_block_total,
        'trainable_phase1': context_block_total,
        'trainable_phase2': token_block_total,
    }


def run_experiment(config_name, num_layers, context_dim, embed_dim, num_input_tokens,
                   train_token_ids, val_token_ids, device, output_dir):
    """単一の実験を実行"""
    print_flush(f"\n{'='*70}")
    print_flush(f"実験: {config_name}")
    print_flush(f"  num_layers={num_layers}, context_dim={context_dim}, "
                f"embed_dim={embed_dim}, num_input_tokens={num_input_tokens}")
    print_flush(f"{'='*70}")

    set_seed(42)

    # パラメータ数計算
    params = calculate_params(
        num_layers=num_layers,
        context_dim=context_dim,
        embed_dim=embed_dim,
        num_input_tokens=num_input_tokens,
        vocab_size=50257
    )
    print_flush(f"  Phase1 params: {params['trainable_phase1']/1e6:.2f}M")
    print_flush(f"  Phase2 params: {params['trainable_phase2']/1e6:.2f}M")

    # モデル作成
    model = LLM(
        vocab_size=50257,
        embed_dim=embed_dim,
        context_dim=context_dim,
        num_layers=num_layers,
        num_input_tokens=num_input_tokens,
        use_pretrained_embeddings=True,
        use_weight_tying=True
    ).to(device)

    # Phase 1用の設定を作成
    class Phase1Config:
        def __init__(self, base_config, ctx_dim, n_layers, n_input_tokens):
            self.context_dim = ctx_dim
            self.embed_dim = base_config.embed_dim
            self.num_layers = n_layers
            self.num_input_tokens = n_input_tokens
            self.phase1_learning_rate = base_config.phase1_learning_rate
            self.phase1_max_iterations = base_config.phase1_max_iterations
            self.phase1_min_iterations = base_config.phase1_min_iterations
            self.phase1_convergence_threshold = base_config.phase1_convergence_threshold
            self.phase1_min_converged_ratio = base_config.phase1_min_converged_ratio
            self.phase1_context_noise = base_config.phase1_context_noise
            self.phase1_batch_size = base_config.phase1_batch_size
            self.phase1_gradient_clip = base_config.phase1_gradient_clip
            self.dist_reg_weight = base_config.dist_reg_weight
            self.device = device

    base_config = ResidualConfig()
    phase1_config = Phase1Config(base_config, context_dim, num_layers, num_input_tokens)

    # Phase 1 実行
    print_flush("\n--- Phase 1: CVFP Learning ---")
    start_time = time.time()

    phase1_trainer = MemoryPhase1Trainer(model, phase1_config)
    train_contexts, train_context_cache, train_token_embeds = phase1_trainer.train(
        train_token_ids, return_all_layers=True
    )
    val_contexts, val_context_cache, val_token_embeds = phase1_trainer.evaluate(
        val_token_ids, return_all_layers=True
    )

    phase1_time = time.time() - start_time
    print_flush(f"  Phase 1 time: {phase1_time:.1f}s")

    # Effective Rank計算
    train_metrics = analyze_fixed_points(train_contexts, label="Train", verbose=False)
    val_metrics = analyze_fixed_points(val_contexts, label="Val", verbose=False)

    train_er = train_metrics['effective_rank']
    val_er = val_metrics['effective_rank']
    train_er_pct = train_er / context_dim * 100
    val_er_pct = val_er / context_dim * 100

    print_flush(f"  Train ER: {train_er:.1f}/{context_dim} ({train_er_pct:.1f}%)")
    print_flush(f"  Val ER: {val_er:.1f}/{context_dim} ({val_er_pct:.1f}%)")

    # Phase 2用の設定を作成
    class Phase2Config:
        def __init__(self, base_config, ctx_dim, n_layers, n_input_tokens):
            self.context_dim = ctx_dim
            self.embed_dim = base_config.embed_dim
            self.num_layers = n_layers
            self.num_input_tokens = n_input_tokens
            self.phase2_learning_rate = base_config.phase2_learning_rate
            self.phase2_epochs = base_config.phase2_epochs
            self.phase2_patience = base_config.phase2_patience
            self.phase2_batch_size = base_config.phase2_batch_size
            self.phase2_gradient_clip = base_config.phase2_gradient_clip
            self.phase2_freeze_embedding = base_config.phase2_freeze_embedding
            self.phase2_memory_safety_factor = base_config.phase2_memory_safety_factor
            self.phase2_min_batch_size = base_config.phase2_min_batch_size
            self.phase2_max_batch_size = base_config.phase2_max_batch_size
            self.device = device

    phase2_config = Phase2Config(base_config, context_dim, num_layers, num_input_tokens)

    # Phase 2 実行
    print_flush("\n--- Phase 2: Token Prediction ---")
    start_time = time.time()

    phase2_trainer = Phase2Trainer(model, phase2_config)
    history = phase2_trainer.train_full(
        train_token_ids=train_token_ids,
        val_token_ids=val_token_ids,
        train_context_cache=train_context_cache,
        train_token_embeds=train_token_embeds,
        val_context_cache=val_context_cache,
        val_token_embeds=val_token_embeds
    )

    phase2_time = time.time() - start_time
    print_flush(f"  Phase 2 time: {phase2_time:.1f}s")

    # 結果取得
    best_epoch = history['best_epoch']
    best_ppl = history['val_ppl'][best_epoch - 1]
    best_acc = history['val_acc'][best_epoch - 1] * 100

    print_flush(f"\n  Best Results (epoch {best_epoch}):")
    print_flush(f"    Val PPL: {best_ppl:.1f}")
    print_flush(f"    Val Acc: {best_acc:.1f}%")

    # 結果を保存
    result = {
        'config_name': config_name,
        'num_layers': num_layers,
        'context_dim': context_dim,
        'embed_dim': embed_dim,
        'num_input_tokens': num_input_tokens,
        'params': params,
        'phase1_time': phase1_time,
        'phase2_time': phase2_time,
        'total_time': phase1_time + phase2_time,
        'train_effective_rank': train_er,
        'train_effective_rank_pct': train_er_pct,
        'val_effective_rank': val_er,
        'val_effective_rank_pct': val_er_pct,
        'best_epoch': best_epoch,
        'best_val_ppl': best_ppl,
        'best_val_acc': best_acc,
        'history': {
            'train_loss': history['train_loss'],
            'val_ppl': history['val_ppl'],
            'val_acc': [a * 100 for a in history['val_acc']],
        }
    }

    # 個別結果を保存
    result_file = os.path.join(output_dir, f'{config_name}.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    print_flush(f"  Saved: {result_file}")

    # メモリ解放
    del model, phase1_trainer, phase2_trainer
    del train_contexts, val_contexts
    del train_context_cache, val_context_cache
    del train_token_embeds, val_token_embeds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def main():
    print_flush("="*70)
    print_flush("アーキテクチャ比較実験")
    print_flush("="*70)

    # 出力ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/architecture_comparison_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print_flush(f"Output: {output_dir}")

    # デバイス設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_flush(f"Device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print_flush(f"  GPU: {gpu_name} ({gpu_mem:.1f}GB)")

    # データロード
    print_flush("\nLoading data...")
    config = ResidualConfig()
    train_token_ids, val_token_ids = load_data(config, device)
    print_flush(f"  Train tokens: {len(train_token_ids):,}")
    print_flush(f"  Val tokens: {len(val_token_ids):,}")

    # 実験設定
    experiments = [
        {
            'name': 'baseline',
            'num_layers': 6,
            'context_dim': 768,
            'embed_dim': 768,
            'num_input_tokens': 1,
        },
        {
            'name': 'input_tokens_2',
            'num_layers': 6,
            'context_dim': 768,
            'embed_dim': 768,
            'num_input_tokens': 2,
        },
        {
            'name': 'context_dim_1152',
            'num_layers': 6,
            'context_dim': 1152,  # 768 * 1.5
            'embed_dim': 768,
            'num_input_tokens': 1,
        },
        {
            'name': 'layers_9',
            'num_layers': 9,
            'context_dim': 768,
            'embed_dim': 768,
            'num_input_tokens': 1,
        },
    ]

    # 実験実行
    results = []
    total_start = time.time()

    for exp in experiments:
        result = run_experiment(
            config_name=exp['name'],
            num_layers=exp['num_layers'],
            context_dim=exp['context_dim'],
            embed_dim=exp['embed_dim'],
            num_input_tokens=exp['num_input_tokens'],
            train_token_ids=train_token_ids,
            val_token_ids=val_token_ids,
            device=device,
            output_dir=output_dir
        )
        results.append(result)

    total_time = time.time() - total_start

    # サマリー出力
    print_flush("\n" + "="*70)
    print_flush("実験結果サマリー")
    print_flush("="*70)

    print_flush(f"\n{'Config':<20} {'Phase1':>10} {'P1 Time':>8} {'Val ER':>8} {'Val PPL':>10} {'Val Acc':>10}")
    print_flush("-"*70)

    for r in results:
        print_flush(
            f"{r['config_name']:<20} "
            f"{r['params']['trainable_phase1']/1e6:>8.2f}M "
            f"{r['phase1_time']:>7.0f}s "
            f"{r['val_effective_rank_pct']:>7.1f}% "
            f"{r['best_val_ppl']:>10.1f} "
            f"{r['best_val_acc']:>9.1f}%"
        )

    print_flush("-"*70)
    print_flush(f"Total time: {total_time/60:.1f} min")

    # 全結果を保存
    summary = {
        'timestamp': timestamp,
        'total_time_sec': total_time,
        'device': device,
        'train_tokens': len(train_token_ids),
        'val_tokens': len(val_token_ids),
        'results': results
    }

    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print_flush(f"\nSaved: {summary_file}")


if __name__ == '__main__':
    main()
