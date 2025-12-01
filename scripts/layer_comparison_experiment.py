#!/usr/bin/env python3
"""
Layer構成比較実験スクリプト

3つの設定を比較:
1. layer=1, fnn_num_layers=1 (ベースライン)
2. layer=1, fnn_num_layers=2 (FFN深化)
3. layer=2, fnn_num_layers=1 (レイヤー追加)

使用方法:
  # Colab（GPU）: 2000サンプルで比較
  python3 scripts/layer_comparison_experiment.py

  # カスタムサンプル数
  python3 scripts/layer_comparison_experiment.py --samples 1000

  # context_dimを指定
  python3 scripts/layer_comparison_experiment.py --context-dim 768
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List
from dataclasses import dataclass

import torch

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from config.experiment import DataConfig, Phase1TrainerConfig, Phase2TrainerConfig
from src.models import LLM
from src.trainers.phase1 import MemoryPhase1Trainer
from src.trainers.phase2 import Phase2Trainer
from src.evaluation.metrics import analyze_fixed_points
from src.providers.data import MemoryDataProvider
from src.utils.io import print_flush
from src.utils.device import clear_gpu_cache
from src.utils.seed import set_seed


@dataclass
class ExperimentConfig:
    """実験設定"""
    name: str
    num_layers: int
    fnn_num_layers: int
    description: str


# 比較する3つの設定
EXPERIMENT_CONFIGS = [
    ExperimentConfig(
        name="L1_F1",
        num_layers=1,
        fnn_num_layers=1,
        description="layer=1, fnn=1 (baseline)"
    ),
    ExperimentConfig(
        name="L1_F2",
        num_layers=1,
        fnn_num_layers=2,
        description="layer=1, fnn=2 (FFN deepened)"
    ),
    ExperimentConfig(
        name="L2_F1",
        num_layers=2,
        fnn_num_layers=1,
        description="layer=2, fnn=1 (layer added)"
    ),
]


def run_single_experiment(
    exp_config: ExperimentConfig,
    num_samples: int,
    base_config: Config,
    device: torch.device,
    seed: int = 42,
    context_dim: int = 500,
) -> Dict[str, Any]:
    """単一の実験を実行（Phase 1 + Phase 2）"""

    set_seed(seed)

    # base_configを一時的に変更
    original_num_layers = base_config.num_layers
    original_fnn_num_layers = base_config.fnn_num_layers

    base_config.num_layers = exp_config.num_layers
    base_config.fnn_num_layers = exp_config.fnn_num_layers

    try:
        # データ読み込み用設定
        data_config = DataConfig.from_base(base_config, num_samples=num_samples)

        # データプロバイダー
        data_provider = MemoryDataProvider(data_config)
        train_token_ids, val_token_ids = data_provider.load_data()
        train_token_ids = train_token_ids.to(device)
        val_token_ids = val_token_ids.to(device)

        num_train_tokens = len(train_token_ids)
        num_val_tokens = len(val_token_ids)

        print_flush(f"    Data: {num_train_tokens:,} train, {num_val_tokens:,} val tokens")

        # モデル作成
        set_seed(seed)
        model = LLM(
            vocab_size=base_config.vocab_size,
            embed_dim=base_config.embed_dim,
            context_dim=context_dim,
            num_layers=exp_config.num_layers,
            num_input_tokens=base_config.num_input_tokens,
            use_pretrained_embeddings=base_config.use_pretrained_embeddings,
            use_weight_tying=base_config.use_weight_tying,
            config=base_config
        )
        model.to(device)

        # パラメータ数を取得
        total_params = sum(p.numel() for p in model.parameters())

        print_flush(f"    Model: {total_params:,} params (layers={exp_config.num_layers}, fnn={exp_config.fnn_num_layers})")

        # Phase 1用設定
        phase1_config = Phase1TrainerConfig.from_base(
            base_config, device,
            context_dim=context_dim,
            num_layers=exp_config.num_layers,
        )

        # Phase 1 トレーナー作成
        phase1_trainer = MemoryPhase1Trainer(model, phase1_config, device)

        # Phase 1 実行
        phase1_start = time.time()
        train_result = phase1_trainer.train(
            train_token_ids,
            label="OACD",
            return_all_layers=True,
            val_token_ids=val_token_ids
        )
        phase1_time = time.time() - phase1_start

        # Phase1Result dataclass から値を取得
        train_contexts = train_result.contexts
        train_context_cache = train_result.cache
        train_token_embeds = train_result.token_embeds

        # Phase 1 統計を取得
        phase1_stats = phase1_trainer._training_stats
        phase1_iterations = phase1_stats.get('iterations', 0)
        best_val_er = phase1_stats.get('best_val_er', 0.0)
        convergence_rate = phase1_stats.get('convergence_rate', 0.0)

        # 検証データのキャッシュ収集
        val_result = phase1_trainer.evaluate(val_token_ids, return_all_layers=True)
        val_contexts = val_result.contexts
        val_context_cache = val_result.cache
        val_token_embeds = val_result.token_embeds

        # Effective Rank計算
        train_metrics = analyze_fixed_points(train_contexts, label="Train", verbose=False)
        val_metrics = analyze_fixed_points(val_contexts, label="Val", verbose=False)
        train_er = train_metrics['effective_rank']
        val_er = val_metrics['effective_rank']
        train_er_pct = train_er / context_dim * 100
        val_er_pct = val_er / context_dim * 100
        best_val_er_pct = best_val_er / context_dim * 100

        print_flush(f"    Phase 1: {phase1_time:.1f}s, {phase1_iterations} iter, "
                    f"ER={train_er_pct:.1f}%/{best_val_er_pct:.1f}%")

        # Phase 2用設定
        phase2_config = Phase2TrainerConfig.from_base(
            base_config, device,
            context_dim=context_dim,
            num_layers=exp_config.num_layers,
        )

        # Phase 2 トレーナー作成
        phase2_trainer = Phase2Trainer(model, phase2_config)

        # Phase 2 実行
        phase2_start = time.time()
        history = phase2_trainer.train_full(
            train_token_ids=train_token_ids,
            val_token_ids=val_token_ids,
            device=device,
            train_context_cache=train_context_cache,
            train_token_embeds=train_token_embeds,
            val_context_cache=val_context_cache,
            val_token_embeds=val_token_embeds
        )
        phase2_time = time.time() - phase2_start

        best_epoch = history['best_epoch']
        best_ppl = history['val_ppl'][best_epoch - 1]
        best_acc = history['val_acc'][best_epoch - 1]
        best_train_ppl = history['train_ppl'][best_epoch - 1]

        print_flush(f"    Phase 2: {phase2_time:.1f}s, PPL={best_ppl:.1f}, Acc={best_acc*100:.1f}%")

        total_time = phase1_time + phase2_time

        # メモリ解放
        del model, phase1_trainer, phase2_trainer
        del train_contexts, val_contexts
        del train_context_cache, val_context_cache
        del train_token_embeds, val_token_embeds
        data_provider.close()
        clear_gpu_cache(device)

        return {
            'name': exp_config.name,
            'description': exp_config.description,
            'num_layers': exp_config.num_layers,
            'fnn_num_layers': exp_config.fnn_num_layers,
            'context_dim': context_dim,
            'num_samples': num_samples,
            'train_tokens': num_train_tokens,
            'val_tokens': num_val_tokens,
            'total_params': total_params,
            'phase1_iterations': phase1_iterations,
            'phase1_time': phase1_time,
            'train_er': train_er,
            'train_er_pct': train_er_pct,
            'val_er': val_er,
            'val_er_pct': val_er_pct,
            'best_val_er': best_val_er,
            'best_val_er_pct': best_val_er_pct,
            'convergence_rate': convergence_rate,
            'phase2_time': phase2_time,
            'best_epoch': best_epoch,
            'train_ppl': best_train_ppl,
            'val_ppl': best_ppl,
            'val_acc': best_acc,
            'total_time': total_time,
        }

    finally:
        # 設定を元に戻す
        base_config.num_layers = original_num_layers
        base_config.fnn_num_layers = original_fnn_num_layers


def main():
    parser = argparse.ArgumentParser(
        description='Layer configuration comparison experiment'
    )
    parser.add_argument(
        '--samples', '-s',
        type=int,
        default=2000,
        help='Number of samples (default: 2000)'
    )
    parser.add_argument(
        '--context-dim', '-c',
        type=int,
        default=500,
        help='Context dimension (default: 500)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default=None,
        help='Output directory (default: auto-generated)'
    )

    args = parser.parse_args()

    # 設定
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # 出力ディレクトリ
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f"importants/logs/{timestamp}_layer_comparison"

    os.makedirs(output_dir, exist_ok=True)

    # 情報表示
    print_flush("=" * 80)
    print_flush("LAYER CONFIGURATION COMPARISON EXPERIMENT")
    print_flush("=" * 80)
    print_flush(f"Device: {device}")
    if device.type == "cuda":
        print_flush(f"GPU: {torch.cuda.get_device_name(0)}")
        print_flush(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    print_flush(f"\nSamples: {args.samples}")
    print_flush(f"Context dim: {args.context_dim}")
    print_flush(f"Output: {output_dir}")
    print_flush("\nConfigurations to compare:")
    for exp_cfg in EXPERIMENT_CONFIGS:
        print_flush(f"  - {exp_cfg.name}: {exp_cfg.description}")
    print_flush("=" * 80)

    # 実験実行
    results: List[Dict[str, Any]] = []
    total_start = time.time()

    for exp_cfg in EXPERIMENT_CONFIGS:
        print_flush(f"\n[{exp_cfg.name}] {exp_cfg.description}")

        try:
            result = run_single_experiment(
                exp_config=exp_cfg,
                num_samples=args.samples,
                base_config=config,
                device=device,
                seed=42,
                context_dim=args.context_dim,
            )
            results.append(result)
        except Exception as e:
            print_flush(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_time = time.time() - total_start

    # サマリー表示
    print_flush("\n" + "=" * 80)
    print_flush("SUMMARY")
    print_flush("=" * 80)

    print_flush(f"\n{'Config':<10} {'Layers':<8} {'FFN':<6} {'Params':<12} "
                f"{'Val PPL':<10} {'Acc':<8} {'ER%':<8} {'Time':<10}")
    print_flush("-" * 80)

    for r in results:
        print_flush(f"{r['name']:<10} {r['num_layers']:<8} {r['fnn_num_layers']:<6} "
                    f"{r['total_params']:>10,}  "
                    f"{r['val_ppl']:<10.1f} {r['val_acc']*100:<8.1f}% "
                    f"{r['val_er_pct']:<8.1f} {r['total_time']:<10.1f}s")

    # 最良の設定を表示
    if results:
        best_ppl = min(results, key=lambda x: x['val_ppl'])
        best_acc = max(results, key=lambda x: x['val_acc'])

        print_flush("\n" + "-" * 80)
        print_flush(f"\nBest PPL:  {best_ppl['name']} (PPL={best_ppl['val_ppl']:.1f})")
        print_flush(f"Best Acc:  {best_acc['name']} (Acc={best_acc['val_acc']*100:.1f}%)")

    print_flush(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # 結果をファイルに保存
    result_file = os.path.join(output_dir, "results.txt")
    with open(result_file, 'w') as f:
        f.write("Layer Configuration Comparison Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Samples: {args.samples}\n")
        f.write(f"Context dim: {args.context_dim}\n")
        f.write(f"Device: {device}\n\n")

        for r in results:
            f.write(f"\n{r['name']}: {r['description']}\n")
            f.write(f"  Params: {r['total_params']:,}\n")
            f.write(f"  Train tokens: {r['train_tokens']:,}\n")
            f.write(f"  Val PPL: {r['val_ppl']:.2f}\n")
            f.write(f"  Val Acc: {r['val_acc']*100:.2f}%\n")
            f.write(f"  Val ER: {r['val_er_pct']:.1f}%\n")
            f.write(f"  Phase 1: {r['phase1_iterations']} iter, {r['phase1_time']:.1f}s\n")
            f.write(f"  Phase 2: epoch {r['best_epoch']}, {r['phase2_time']:.1f}s\n")
            f.write(f"  Total time: {r['total_time']:.1f}s\n")

    print_flush(f"\nResults saved to: {result_file}")

    print_flush("\n" + "=" * 80)
    print_flush("DONE")
    print_flush("=" * 80)


if __name__ == '__main__':
    main()
