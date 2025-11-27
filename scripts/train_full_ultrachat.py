#!/usr/bin/env python3
"""
UltraChat全データ訓練スクリプト

Phase 1 CVFP学習をディスクオフロード方式で実行。
200kサンプル（約25.6Mトークン）を32GB RAMで処理可能。

使用方法:
    # 1. データ準備（初回のみ）
    python3 scripts/prepare_disk_offload.py --output_dir /mnt/nvme/cvfp

    # 2. 訓練開始
    python3 scripts/train_full_ultrachat.py --disk_dir /mnt/nvme/cvfp

    # 3. テストモード
    python3 scripts/train_full_ultrachat.py --disk_dir ./test_offload --test
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import ResidualConfig
from src.models.llm import LLM
from src.providers import create_phase1_trainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_flush(msg: str):
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser(description='UltraChat全データ訓練（ディスクオフロード）')
    parser.add_argument('--disk_dir', type=str, required=True, help='ディスクオフロードディレクトリ')
    parser.add_argument('--max_iterations', type=int, default=10, help='最大イテレーション数')
    parser.add_argument('--test', action='store_true', help='テストモード')
    parser.add_argument('--num_layers', type=int, default=6, help='レイヤー数')
    args = parser.parse_args()

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

    # config設定を上書き
    config.disk_offload_dir = args.disk_dir
    config.phase1_max_iterations = args.max_iterations
    config.num_layers = args.num_layers

    if args.test:
        config.disk_offload_chunk_size = min(config.disk_offload_chunk_size, 100_000)
        print_flush("テストモード: チャンクサイズを100,000に制限")

    set_seed(config.random_seed)
    device = torch.device(config.device)

    print_flush(f"\n{'='*70}")
    print_flush("UltraChat全データ訓練")
    print_flush(f"{'='*70}")
    print_flush(f"  ディスクディレクトリ: {args.disk_dir}")
    print_flush(f"  トークン数: {num_tokens:,}")
    print_flush(f"  レイヤー数: {args.num_layers}")
    print_flush(f"  デバイス: {device}")

    # モデル作成
    print_flush("\nモデルを作成中...")
    model = LLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        context_dim=context_dim,
        num_layers=args.num_layers,
        use_pretrained_embeddings=config.use_pretrained_embeddings
    )

    total_params = sum(p.numel() for p in model.parameters())
    print_flush(f"  総パラメータ: {total_params:,}")

    # トレーナー作成（依存性注入）
    trainer = create_phase1_trainer("storage", model, config, device)

    # トークンIDをロード
    from src.utils.disk_offload import TokenIDCache
    token_cache = TokenIDCache(args.disk_dir, num_tokens)
    token_cache.open('r')
    token_ids = token_cache.get_chunk(0, num_tokens)
    token_cache.close()

    # 訓練実行
    contexts = trainer.train(token_ids.to(device), label="Train")

    # 結果表示
    stats = trainer.get_training_stats()
    print_flush(f"\n{'='*70}")
    print_flush("訓練完了")
    print_flush(f"{'='*70}")
    print_flush(f"  イテレーション数: {stats.get('iterations', 'N/A')}")
    print_flush(f"  収束率: {stats.get('convergence_rate', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
