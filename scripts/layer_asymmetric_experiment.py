#!/usr/bin/env python3
"""
非対称レイヤー実験スクリプト

ContextBlockとTokenBlockのレイヤー数を別々に設定して比較実験。

比較パターン:
- C2T1: ContextBlock 2層, TokenBlock 1層
- C1T2: ContextBlock 1層, TokenBlock 2層
- C2T2: ContextBlock 2層, TokenBlock 2層（ベースライン）

使用方法:
  # C2T1実験（Context 2層, Token 1層）
  python3 scripts/layer_asymmetric_experiment.py --mode c2t1

  # C1T2実験（Context 1層, Token 2層）
  python3 scripts/layer_asymmetric_experiment.py --mode c1t2

  # カスタムサンプル数
  python3 scripts/layer_asymmetric_experiment.py --mode c2t1 --samples 1000

  # context_dim指定
  python3 scripts/layer_asymmetric_experiment.py --mode c1t2 --context-dim 500
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, Any

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


def run_asymmetric_experiment(
    num_samples: int,
    context_layers: int,
    token_layers: int,
    base_config: Config,
    device: torch.device,
    seed: int = 42,
    context_dim: int = 500,
) -> Dict[str, Any]:
    """非対称レイヤー実験を実行"""

    set_seed(seed)

    config_name = f"C{context_layers}T{token_layers}"
    print_flush(f"\n{'='*60}")
    print_flush(f"Running {config_name}: Context {context_layers}L, Token {token_layers}L")
    print_flush(f"{'='*60}")

    # データ読み込み用設定
    data_config = DataConfig.from_base(base_config, num_samples=num_samples)

    # データプロバイダー
    data_provider = MemoryDataProvider(data_config)
    train_token_ids, val_token_ids = data_provider.load_data()
    train_token_ids = train_token_ids.to(device)
    val_token_ids = val_token_ids.to(device)

    num_train_tokens = len(train_token_ids)
    num_val_tokens = len(val_token_ids)

    print_flush(f"Data: {num_train_tokens:,} train, {num_val_tokens:,} val tokens")

    # モデル作成（非対称レイヤー）
    set_seed(seed)
    model = LLM(
        vocab_size=base_config.vocab_size,
        embed_dim=base_config.embed_dim,
        context_dim=context_dim,
        context_layers=context_layers,
        token_layers=token_layers,
        num_input_tokens=base_config.num_input_tokens,
        use_pretrained_embeddings=base_config.use_pretrained_embeddings,
        use_weight_tying=base_config.use_weight_tying,
    )
    model.to(device)

    # パラメータ数表示
    params = model.num_params()
    print_flush(f"Parameters: {params['total']:,} total")
    print_flush(f"  ContextBlock: {params['context_block']:,}")
    print_flush(f"  TokenBlock: {params['token_block']:,}")

    # Phase 1用設定（context_layersを使用）
    phase1_config = Phase1TrainerConfig.from_base(
        base_config, device,
        context_dim=context_dim,
        num_layers=context_layers,  # Phase1はContextBlockのレイヤー数
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
    assert train_result.cache is not None
    assert train_result.token_embeds is not None
    train_contexts = train_result.contexts
    # G案: 最終レイヤーのキャッシュのみ使用
    train_context_cache = train_result.cache[-1]
    train_token_embeds = train_result.token_embeds

    # Phase 1 統計を取得
    phase1_stats = phase1_trainer._training_stats
    phase1_iterations = phase1_stats.get('iterations', 0)
    convergence_rate = phase1_stats.get('convergence_rate', 0.0)

    # 検証データのキャッシュ収集
    val_result = phase1_trainer.evaluate(val_token_ids, return_all_layers=True)
    assert val_result.cache is not None
    assert val_result.token_embeds is not None
    val_contexts = val_result.contexts
    val_context_cache = val_result.cache[-1]
    val_token_embeds = val_result.token_embeds

    # Effective Rank計算
    train_metrics = analyze_fixed_points(train_contexts, label="Train", verbose=False)
    val_metrics = analyze_fixed_points(val_contexts, label="Val", verbose=False)
    train_er = train_metrics['effective_rank']
    val_er = val_metrics['effective_rank']
    train_er_pct = train_er / context_dim * 100
    val_er_pct = val_er / context_dim * 100

    print_flush(f"Phase 1: {phase1_time:.1f}s, {phase1_iterations} iter, "
                f"conv={convergence_rate*100:.0f}%, ER={train_er_pct:.1f}%/{val_er_pct:.1f}%")

    # Phase 2用設定（token_layersを使用）
    phase2_config = Phase2TrainerConfig.from_base(
        base_config, device,
        context_dim=context_dim,
        num_layers=token_layers,  # Phase2はTokenBlockのレイヤー数
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

    total_time = phase1_time + phase2_time

    print_flush(f"Phase 2: {phase2_time:.1f}s, Best epoch {best_epoch}")
    print_flush(f"Result: PPL={best_ppl:.1f}, Acc={best_acc*100:.1f}%")
    print_flush(f"Total time: {total_time:.1f}s")

    # メモリ解放
    del model, phase1_trainer, phase2_trainer
    del train_contexts, val_contexts
    del train_context_cache, val_context_cache
    del train_token_embeds, val_token_embeds
    data_provider.close()
    clear_gpu_cache(device)

    return {
        'config_name': config_name,
        'context_layers': context_layers,
        'token_layers': token_layers,
        'context_dim': context_dim,
        'num_samples': num_samples,
        'train_tokens': num_train_tokens,
        'val_tokens': num_val_tokens,
        'total_params': params['total'],
        'context_block_params': params['context_block'],
        'token_block_params': params['token_block'],
        'phase1_iterations': phase1_iterations,
        'phase1_time': phase1_time,
        'train_er': train_er,
        'train_er_pct': train_er_pct,
        'val_er': val_er,
        'val_er_pct': val_er_pct,
        'convergence_rate': convergence_rate,
        'phase2_time': phase2_time,
        'best_epoch': best_epoch,
        'train_ppl': best_train_ppl,
        'val_ppl': best_ppl,
        'val_acc': best_acc,
        'total_time': total_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Run asymmetric layer experiment (Context vs Token layers)'
    )
    parser.add_argument(
        '--mode', '-m',
        choices=['c1t1', 'c2t1', 'c1t2', 'c2t2', 'all'],
        default='c2t1',
        help='Experiment mode: c1t1 (1L+1L), c2t1 (2L+1L), c1t2 (1L+2L), c2t2 (2L+2L), all (run all)'
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
        output_dir = f"importants/logs/{timestamp}_asymmetric_{args.mode}"

    os.makedirs(output_dir, exist_ok=True)

    # 情報表示
    print_flush("=" * 70)
    print_flush("ASYMMETRIC LAYER EXPERIMENT")
    print_flush("=" * 70)
    print_flush(f"Device: {device}")
    if device.type == "cuda":
        print_flush(f"GPU: {torch.cuda.get_device_name(0)}")
        print_flush(f"Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    print_flush(f"\nMode: {args.mode}")
    print_flush(f"Samples: {args.samples}")
    print_flush(f"Context dim: {args.context_dim}")
    print_flush(f"Output: {output_dir}")
    print_flush("=" * 70)

    # 実験設定
    experiments = []
    if args.mode == 'c1t1' or args.mode == 'all':
        experiments.append({'context_layers': 1, 'token_layers': 1, 'name': 'C1T1'})
    if args.mode == 'c2t1' or args.mode == 'all':
        experiments.append({'context_layers': 2, 'token_layers': 1, 'name': 'C2T1'})
    if args.mode == 'c1t2' or args.mode == 'all':
        experiments.append({'context_layers': 1, 'token_layers': 2, 'name': 'C1T2'})
    if args.mode == 'c2t2' or args.mode == 'all':
        experiments.append({'context_layers': 2, 'token_layers': 2, 'name': 'C2T2'})

    # 実験実行
    results = []

    for exp in experiments:
        try:
            result = run_asymmetric_experiment(
                num_samples=args.samples,
                context_layers=exp['context_layers'],
                token_layers=exp['token_layers'],
                base_config=config,
                device=device,
                seed=42,
                context_dim=args.context_dim,
            )
            results.append(result)
        except Exception as e:
            print_flush(f"ERROR in {exp['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # サマリー表示
    if results:
        print_flush("\n" + "=" * 70)
        print_flush("SUMMARY")
        print_flush("=" * 70)

        print_flush(f"\n{'Config':<10} {'Context':<8} {'Token':<8} {'Params':<12} "
                    f"{'Val PPL':<10} {'Acc':<8} {'ER%':<8} {'Time':<8}")
        print_flush("-" * 80)

        for r in results:
            print_flush(f"{r['config_name']:<10} {r['context_layers']:<8} {r['token_layers']:<8} "
                       f"{r['total_params']:,}  {r['val_ppl']:<10.1f} "
                       f"{r['val_acc']*100:<8.1f} {r['val_er_pct']:<8.1f} {r['total_time']:<8.1f}")

        # 結果をファイルに保存
        output_file = os.path.join(output_dir, f"results_{args.mode}.txt")
        with open(output_file, 'w') as f:
            f.write(f"Asymmetric Layer Experiment Results\n")
            f.write(f"Mode: {args.mode}\n")
            f.write(f"Samples: {args.samples}\n")
            f.write(f"Context dim: {args.context_dim}\n")
            f.write(f"Device: {device}\n\n")

            for r in results:
                f.write(f"\n{r['config_name']}: Context {r['context_layers']}L, Token {r['token_layers']}L\n")
                f.write(f"  Parameters: {r['total_params']:,}\n")
                f.write(f"    ContextBlock: {r['context_block_params']:,}\n")
                f.write(f"    TokenBlock: {r['token_block_params']:,}\n")
                f.write(f"  Train tokens: {r['train_tokens']:,}\n")
                f.write(f"  Val PPL: {r['val_ppl']:.2f}\n")
                f.write(f"  Val Acc: {r['val_acc']*100:.2f}%\n")
                f.write(f"  Val ER: {r['val_er_pct']:.1f}%\n")
                f.write(f"  Phase 1: {r['phase1_iterations']} iter, {r['phase1_time']:.1f}s\n")
                f.write(f"  Phase 2: epoch {r['best_epoch']}, {r['phase2_time']:.1f}s\n")
                f.write(f"  Total time: {r['total_time']:.1f}s\n")

        print_flush(f"\nResults saved to: {output_file}")

    print_flush("\n" + "=" * 70)
    print_flush("DONE")
    print_flush("=" * 70)


if __name__ == '__main__':
    main()
