#!/usr/bin/env python3
"""
ディスクオフロード用データ準備スクリプト

UltraChat 200kをストリーミングでダウンロードし、
トークン化・埋め込み計算してNVMeに保存。

使用方法:
    python3 scripts/prepare_disk_offload.py --output_dir /mnt/nvme/cvfp

    # サンプル数を指定
    python3 scripts/prepare_disk_offload.py --output_dir /mnt/nvme/cvfp --num_samples 100000

    # テスト（少数サンプル）
    python3 scripts/prepare_disk_offload.py --output_dir ./test_offload --num_samples 1000 --test
"""

import os
import sys
import argparse

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from src.utils.streaming_loader import StreamingDataLoader


def main():
    parser = argparse.ArgumentParser(
        description='ディスクオフロード用データ準備'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='出力ディレクトリ（NVMeマウントポイント）'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=200_000,
        help='処理するサンプル数（デフォルト: 200000）'
    )
    parser.add_argument(
        '--no_bf16',
        action='store_true',
        help='bf16を使用しない（float32を使用）'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='テストモード（少数サンプル、CPU）'
    )
    args = parser.parse_args()

    # テストモードの場合
    if args.test:
        args.num_samples = min(args.num_samples, 1000)
        print(f"テストモード: {args.num_samples}サンプルで実行")

    # デバイス
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"デバイス: {device}")

    # ストリーミングローダー作成
    loader = StreamingDataLoader(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        use_bf16=not args.no_bf16
    )

    # 既に準備済みかチェック
    if loader.is_prepared():
        metadata = loader.load_metadata()
        print(f"\nデータは既に準備済みです:")
        print(f"  トークン数: {metadata['num_tokens']:,}")
        print(f"  サンプル数: {metadata['num_samples']:,}")
        print(f"  メタデータ: {loader.metadata_path}")

        response = input("\n上書きしますか? [y/N]: ")
        if response.lower() != 'y':
            print("キャンセルしました。")
            return

    # データ準備
    metadata = loader.prepare(device=device)

    print(f"\n準備完了:")
    print(f"  トークン数: {metadata['num_tokens']:,}")
    print(f"  サンプル数: {metadata['num_samples']:,}")

    # ストレージ見積もり
    bytes_per_element = 2 if metadata['use_bf16'] else 4
    embed_size = metadata['num_tokens'] * metadata['embed_dim'] * bytes_per_element
    context_size = metadata['num_tokens'] * metadata['context_dim'] * bytes_per_element
    token_size = metadata['num_tokens'] * 8  # int64

    print(f"\nストレージ使用量:")
    print(f"  トークンID: {token_size / 1e9:.2f} GB")
    print(f"  埋め込み: {embed_size / 1e9:.2f} GB")
    print(f"  コンテキスト（訓練用×2）: {context_size * 2 / 1e9:.2f} GB")
    print(f"  合計: {(token_size + embed_size + context_size * 2) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
