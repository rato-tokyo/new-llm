#!/usr/bin/env python3
"""
アーキテクチャ比較実験スクリプト（スケーリング則計算付き）

4つの設定を複数サンプル数で比較し、α値を算出:
1. Baseline: 6層、768次元、1トークン入力
2. Exp1: 6層、768次元、2トークン入力 (num_input_tokens=2)
3. Exp2: 6層、1152次元、1トークン入力 (context_dim*1.5)
4. Exp3: 9層、768次元、1トークン入力 (num_layers=9)

スケーリング則: PPL = A × tokens^α
- α値が小さい（より負）ほど、データ効率が良い

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
from scipy import stats

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import LLM
from src.trainers.phase1.memory import MemoryPhase1Trainer
from src.trainers.phase2 import Phase2Trainer
from src.evaluation.metrics import analyze_fixed_points
from src.providers.data import MemoryDataProvider

# 実験設定
SAMPLE_SIZES = [50, 100, 200, 500]  # スケーリング則計算用
RANDOM_SEED = 42


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


def calculate_scaling_law(results: list):
    """スケーリング則を計算: PPL = A × tokens^α"""
    if len(results) < 2:
        return {'alpha': None, 'A': None, 'r_squared': None, 'p_value': None, 'std_err': None}

    tokens = np.array([r['train_tokens'] for r in results])
    ppl = np.array([r['val_ppl'] for r in results])

    # 対数変換
    log_tokens = np.log(tokens)
    log_ppl = np.log(ppl)

    # 線形回帰
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_tokens, log_ppl)

    alpha = slope  # 負の値が期待される
    A = np.exp(intercept)
    r_squared = r_value ** 2

    return {
        'alpha': alpha,
        'A': A,
        'r_squared': r_squared,
        'p_value': p_value,
        'std_err': std_err,
    }


def create_val_data_from_train(train_token_ids, tokenizer, val_file_path, val_ratio=0.1):
    """訓練データから検証データを生成"""
    val_size = int(len(train_token_ids) * val_ratio)
    val_token_ids = train_token_ids[-val_size:]
    val_text = tokenizer.decode(val_token_ids.tolist())
    os.makedirs(os.path.dirname(val_file_path), exist_ok=True)
    with open(val_file_path, 'w', encoding='utf-8') as f:
        f.write(val_text)
    return val_size


def run_single_experiment(config_name, num_layers, context_dim, embed_dim, num_input_tokens,
                          num_samples, device, base_config):
    """単一設定・単一サンプル数での実験"""
    set_seed(RANDOM_SEED)

    # データ読み込み
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    provider = MemoryDataProvider(
        dataset_name="HuggingFaceH4/ultrachat_200k",
        dataset_split="train_sft",
        num_samples=num_samples,
        tokenizer_name="gpt2",
        cache_dir="./cache",
        val_ratio=0.1
    )
    train_token_ids, val_token_ids = provider.get_data(device)
    train_tokens = len(train_token_ids)
    val_tokens = len(val_token_ids)

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

    # Phase 1用の設定
    class Phase1Config:
        def __init__(self, base, ctx_dim, n_layers, n_input_tokens):
            self.context_dim = ctx_dim
            self.embed_dim = base.embed_dim
            self.num_layers = n_layers
            self.num_input_tokens = n_input_tokens
            self.phase1_learning_rate = base.phase1_learning_rate
            self.phase1_max_iterations = base.phase1_max_iterations
            self.phase1_min_iterations = base.phase1_min_iterations
            self.phase1_convergence_threshold = base.phase1_convergence_threshold
            self.phase1_min_converged_ratio = base.phase1_min_converged_ratio
            self.phase1_context_noise = base.phase1_context_noise
            self.phase1_batch_size = base.phase1_batch_size
            self.phase1_gradient_clip = base.phase1_gradient_clip
            self.dist_reg_weight = base.dist_reg_weight
            self.device = device

    phase1_config = Phase1Config(base_config, context_dim, num_layers, num_input_tokens)

    # Phase 1 実行
    phase1_trainer = MemoryPhase1Trainer(model, phase1_config)
    train_contexts, train_context_cache, train_token_embeds = phase1_trainer.train(
        train_token_ids, return_all_layers=True
    )
    val_contexts, val_context_cache, val_token_embeds = phase1_trainer.evaluate(
        val_token_ids, return_all_layers=True
    )

    # Effective Rank計算
    train_metrics = analyze_fixed_points(train_contexts, label="Train", verbose=False)
    val_metrics = analyze_fixed_points(val_contexts, label="Val", verbose=False)
    train_er = train_metrics['effective_rank'] / context_dim
    val_er = val_metrics['effective_rank'] / context_dim

    # Phase 2用の設定
    class Phase2Config:
        def __init__(self, base, ctx_dim, n_layers, n_input_tokens):
            self.context_dim = ctx_dim
            self.embed_dim = base.embed_dim
            self.num_layers = n_layers
            self.num_input_tokens = n_input_tokens
            self.phase2_learning_rate = base.phase2_learning_rate
            self.phase2_epochs = base.phase2_epochs
            self.phase2_patience = base.phase2_patience
            self.phase2_batch_size = base.phase2_batch_size
            self.phase2_gradient_clip = base.phase2_gradient_clip
            self.phase2_freeze_embedding = base.phase2_freeze_embedding
            self.phase2_memory_safety_factor = base.phase2_memory_safety_factor
            self.phase2_min_batch_size = base.phase2_min_batch_size
            self.phase2_max_batch_size = base.phase2_max_batch_size
            self.device = device

    phase2_config = Phase2Config(base_config, context_dim, num_layers, num_input_tokens)

    # Phase 2 実行
    phase2_trainer = Phase2Trainer(model, phase2_config)
    history = phase2_trainer.train_full(
        train_token_ids=train_token_ids,
        val_token_ids=val_token_ids,
        train_context_cache=train_context_cache,
        train_token_embeds=train_token_embeds,
        val_context_cache=val_context_cache,
        val_token_embeds=val_token_embeds
    )

    best_epoch = history['best_epoch']
    best_ppl = history['val_ppl'][best_epoch - 1]
    best_acc = history['val_acc'][best_epoch - 1]

    # メモリ解放
    del model, phase1_trainer, phase2_trainer
    del train_contexts, val_contexts
    del train_context_cache, val_context_cache
    del train_token_embeds, val_token_embeds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'num_samples': num_samples,
        'train_tokens': train_tokens,
        'val_tokens': val_tokens,
        'val_ppl': best_ppl,
        'val_acc': best_acc,
        'train_effective_rank': train_er,
        'val_effective_rank': val_er,
        'best_epoch': best_epoch,
    }


def run_architecture_experiment(config_name, num_layers, context_dim, embed_dim, num_input_tokens,
                                 device, base_config, output_dir):
    """単一アーキテクチャで複数サンプル数の実験を実行"""
    print_flush(f"\n{'='*70}")
    print_flush(f"Architecture: {config_name}")
    print_flush(f"  num_layers={num_layers}, context_dim={context_dim}, num_input_tokens={num_input_tokens}")
    print_flush(f"{'='*70}")

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

    results = []
    start_time = time.time()

    for num_samples in SAMPLE_SIZES:
        print_flush(f"\n  --- {num_samples} samples ---")
        result = run_single_experiment(
            config_name=config_name,
            num_layers=num_layers,
            context_dim=context_dim,
            embed_dim=embed_dim,
            num_input_tokens=num_input_tokens,
            num_samples=num_samples,
            device=device,
            base_config=base_config
        )
        results.append(result)
        print_flush(f"    Tokens: {result['train_tokens']:,}, PPL: {result['val_ppl']:.1f}, "
                   f"Acc: {result['val_acc']*100:.1f}%, ER: {result['val_effective_rank']*100:.1f}%")

    total_time = time.time() - start_time

    # スケーリング則計算
    scaling = calculate_scaling_law(results)

    print_flush(f"\n  Scaling Law: α = {scaling['alpha']:.4f} (R² = {scaling['r_squared']:.4f})")

    # 結果をまとめる
    arch_result = {
        'config_name': config_name,
        'num_layers': num_layers,
        'context_dim': context_dim,
        'embed_dim': embed_dim,
        'num_input_tokens': num_input_tokens,
        'params': params,
        'total_time': total_time,
        'sample_results': results,
        'scaling_law': scaling,
    }

    # 個別結果を保存
    result_file = os.path.join(output_dir, f'{config_name}.json')
    with open(result_file, 'w') as f:
        json.dump(arch_result, f, indent=2)

    return arch_result


def main():
    print_flush("="*70)
    print_flush("Architecture Comparison Experiment (Scaling Law)")
    print_flush("="*70)
    print_flush(f"Sample sizes: {SAMPLE_SIZES}")

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

    # ベース設定を読み込み
    from config import ResidualConfig
    base_config = ResidualConfig()

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
    all_results = []
    total_start = time.time()

    for exp in experiments:
        result = run_architecture_experiment(
            config_name=exp['name'],
            num_layers=exp['num_layers'],
            context_dim=exp['context_dim'],
            embed_dim=exp['embed_dim'],
            num_input_tokens=exp['num_input_tokens'],
            device=device,
            base_config=base_config,
            output_dir=output_dir
        )
        all_results.append(result)

    total_time = time.time() - total_start

    # サマリー出力
    print_flush("\n" + "="*70)
    print_flush("SUMMARY: Scaling Law Comparison")
    print_flush("="*70)

    print_flush(f"\n{'Config':<20} {'Phase1 Params':>14} {'α (slope)':>12} {'R²':>8} {'Best PPL':>10}")
    print_flush("-"*70)

    for r in all_results:
        # 500サンプルでの最良PPL
        best_ppl = r['sample_results'][-1]['val_ppl'] if r['sample_results'] else 0
        alpha = r['scaling_law']['alpha']
        r2 = r['scaling_law']['r_squared']
        print_flush(
            f"{r['config_name']:<20} "
            f"{r['params']['trainable_phase1']/1e6:>12.2f}M "
            f"{alpha:>12.4f} "
            f"{r2:>8.4f} "
            f"{best_ppl:>10.1f}"
        )

    print_flush("-"*70)
    print_flush(f"\nTotal time: {total_time/60:.1f} min")

    # α値の解釈
    print_flush("\n" + "="*70)
    print_flush("INTERPRETATION")
    print_flush("="*70)
    print_flush("Scaling Law: PPL = A × tokens^α")
    print_flush("  - より負のα → データ効率が良い（少ないデータでPPL低下）")
    print_flush("  - R² > 0.9 → スケーリング則への適合度が高い")

    # ランキング
    sorted_results = sorted(all_results, key=lambda x: x['scaling_law']['alpha'] or 0)
    print_flush("\nRanking by α (better = more negative):")
    for i, r in enumerate(sorted_results, 1):
        alpha = r['scaling_law']['alpha']
        if alpha is not None:
            print_flush(f"  {i}. {r['config_name']}: α = {alpha:.4f}")

    # 全結果を保存
    summary = {
        'timestamp': timestamp,
        'sample_sizes': SAMPLE_SIZES,
        'total_time_sec': total_time,
        'device': device,
        'results': all_results
    }

    summary_file = os.path.join(output_dir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print_flush(f"\nSaved: {summary_file}")


if __name__ == '__main__':
    main()
