#!/usr/bin/env python3
"""
UltraChat全データ訓練スクリプト

Phase 1 CVFP学習をディスクオフロード方式で実行。
200kサンプル（約25.6Mトークン）を32GB RAMで処理可能。

使用方法:
    # 1. データ準備（初回のみ）
    python3 scripts/prepare_disk_offload.py --output_dir /mnt/nvme/cvfp

    # 2. 訓練開始
    python3 train_full_ultrachat.py --disk_dir /mnt/nvme/cvfp

    # 3. 再開
    python3 train_full_ultrachat.py --disk_dir /mnt/nvme/cvfp --resume

    # 4. テストモード
    python3 train_full_ultrachat.py --disk_dir ./test_offload --test
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from config import ResidualConfig
from src.models.llm import LLM
from src.trainers.phase1_disk_offload import Phase1DiskOffloadTrainer


def set_seed(seed: int = 42):
    """乱数シードを固定。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_flush(msg: str):
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='UltraChat全データ訓練（ディスクオフロード）'
    )
    parser.add_argument(
        '--disk_dir',
        type=str,
        required=True,
        help='ディスクオフロードディレクトリ（prepare_disk_offload.pyで作成）'
    )
    parser.add_argument(
        '--max_iterations',
        type=int,
        default=10,
        help='最大イテレーション数（デフォルト: 10）'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='学習率（デフォルト: config.pyの値）'
    )
    parser.add_argument(
        '--dist_reg_weight',
        type=float,
        default=None,
        help='多様性損失の重み（デフォルト: config.pyの値）'
    )
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=None,
        help='チャンクサイズ（デフォルト: config.pyの値）'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='最新のチェックポイントから再開'
    )
    parser.add_argument(
        '--resume_from',
        type=str,
        default=None,
        help='指定したチェックポイントから再開'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='テストモード（小さいチャンクサイズ）'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=6,
        help='レイヤー数（デフォルト: 6）'
    )
    args = parser.parse_args()

    # 設定をロード
    config = ResidualConfig()

    # メタデータを読み込み
    metadata_path = os.path.join(args.disk_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print_flush(f"エラー: メタデータが見つかりません: {metadata_path}")
        print_flush("まず scripts/prepare_disk_offload.py を実行してください。")
        sys.exit(1)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    num_tokens = metadata['num_tokens']
    context_dim = metadata['context_dim']
    use_bf16 = metadata['use_bf16']

    # パラメータ設定
    learning_rate = args.learning_rate or config.phase1_learning_rate
    dist_reg_weight = args.dist_reg_weight or config.dist_reg_weight
    chunk_size = args.chunk_size or config.disk_offload_chunk_size
    num_layers = args.num_layers

    if args.test:
        chunk_size = min(chunk_size, 100_000)
        print_flush("テストモード: チャンクサイズを100,000に制限")

    # シード固定
    set_seed(config.random_seed)

    # デバイス
    device = torch.device(config.device)
    print_flush(f"\n{'='*70}")
    print_flush("UltraChat全データ訓練")
    print_flush(f"{'='*70}")
    print_flush(f"  ディスクディレクトリ: {args.disk_dir}")
    print_flush(f"  トークン数: {num_tokens:,}")
    print_flush(f"  レイヤー数: {num_layers}")
    print_flush(f"  精度: {'bf16' if use_bf16 else 'float32'}")
    print_flush(f"  チャンクサイズ: {chunk_size:,}")
    print_flush(f"  デバイス: {device}")
    print_flush("")

    # モデル作成
    print_flush("モデルを作成中...")
    model = LLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=context_dim,
        num_layers=num_layers,
        use_pretrained_embeddings=config.use_pretrained_embeddings
    )

    total_params = sum(p.numel() for p in model.parameters())
    context_params = sum(p.numel() for p in model.context_block.parameters())
    print_flush(f"  総パラメータ: {total_params:,}")
    print_flush(f"  ContextBlockパラメータ: {context_params:,}")

    # トレーナー作成
    trainer = Phase1DiskOffloadTrainer(
        model=model,
        storage_dir=args.disk_dir,
        num_tokens=num_tokens,
        context_dim=context_dim,
        use_bf16=use_bf16,
        chunk_size=chunk_size,
        device=device
    )

    # 再開チェックポイントを探す
    resume_path = None
    if args.resume_from:
        resume_path = args.resume_from
    elif args.resume:
        checkpoint_dir = os.path.join(args.disk_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = sorted([
                f for f in os.listdir(checkpoint_dir)
                if f.startswith("iteration_") and f.endswith(".pt")
            ])
            if checkpoints:
                resume_path = os.path.join(checkpoint_dir, checkpoints[-1])
                print_flush(f"最新チェックポイントから再開: {resume_path}")

    # 訓練実行
    results = trainer.train(
        max_iterations=args.max_iterations,
        learning_rate=learning_rate,
        dist_reg_weight=dist_reg_weight,
        context_noise=config.phase1_context_noise,
        convergence_threshold=config.phase1_convergence_threshold,
        resume_from=resume_path
    )

    # 結果表示
    print_flush(f"\n{'='*70}")
    print_flush("訓練完了")
    print_flush(f"{'='*70}")
    print_flush(f"  イテレーション数: {results['iterations']}")
    print_flush(f"  最終モデル: {results['final_model_path']}")

    if results['stats']:
        last_stats = results['stats'][-1]
        if last_stats['type'] == 'parallel':
            print_flush(f"  最終CVFP損失: {last_stats['avg_cvfp_loss']:.6f}")
            print_flush(f"  最終多様性損失: {last_stats['avg_diversity_loss']:.6f}")
            print_flush(f"  最終収束率: {last_stats['convergence_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
