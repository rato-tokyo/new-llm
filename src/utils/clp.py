"""
Continuous Learning Policy (CLP) Utilities - 削除禁止

モデルの継続的成長を支援するユーティリティ。
- 重みの保存・読み込み（メモリ状態は除外）
- タイムスタンプベースのデータオフセット計算

⚠️ このモジュールはCLPポリシーの実装です。削除・変更禁止。
"""

import os
import time
from typing import Callable

import torch
import torch.nn as nn


# ============================================================
# 定数 - 削除禁止
# ============================================================

CHECKPOINT_DIR = "checkpoints"
SENRI_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "senri_model.pt")
PYTHIA_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "pythia_model.pt")

# タイムスタンプベースオフセット計算の基準点
# 2025-12-09 00:00:00 JST (UTC+9) = 2025-12-08 15:00:00 UTC
CLP_BASE_TIMESTAMP = 1733662800

# 1時間ごとに進むサンプル数（データセットのサイズに応じて調整）
CLP_SAMPLES_PER_HOUR = 10000


# ============================================================
# モデル保存・読み込み - 削除禁止
# ============================================================

def load_or_create_model(
    model_fn: Callable[[], nn.Module],
    checkpoint_path: str,
) -> nn.Module:
    """
    既存の重みがあれば読み込み、なければ新規作成。
    メモリ状態は常にリセット。

    Args:
        model_fn: モデルを作成する関数（引数なし）
        checkpoint_path: チェックポイントファイルのパス

    Returns:
        モデル（重みロード済み or 新規作成）
    """
    model = model_fn()

    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"✓ CLP: Loaded weights from {checkpoint_path}")
    else:
        print(f"✓ CLP: Created new model (no checkpoint at {checkpoint_path})")

    # メモリは常にリセット（重要）
    if hasattr(model, 'reset_memory'):
        model.reset_memory()

    return model


def save_model(model: nn.Module, checkpoint_path: str) -> None:
    """
    重みのみ保存（メモリ状態は保存しない）。

    Args:
        model: 保存するモデル
        checkpoint_path: 保存先パス
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✓ CLP: Saved weights to {checkpoint_path}")


# ============================================================
# タイムスタンプベースデータオフセット - 削除禁止
# ============================================================

def get_data_offset(
    base_timestamp: int = CLP_BASE_TIMESTAMP,
    samples_per_hour: int = CLP_SAMPLES_PER_HOUR,
) -> int:
    """
    現在時刻からデータオフセットを計算。
    時間が経過するほど、データセットの後ろの方を使用する。

    Args:
        base_timestamp: 基準タイムスタンプ（デフォルト: 2025-12-09 00:00:00 JST）
        samples_per_hour: 1時間あたりのサンプル数

    Returns:
        データオフセット（サンプル数）
    """
    elapsed_seconds = time.time() - base_timestamp
    elapsed_hours = max(0, elapsed_seconds / 3600)
    offset = int(elapsed_hours * samples_per_hour)
    return offset


def get_data_range(
    total_samples: int,
    num_samples: int,
    base_timestamp: int = CLP_BASE_TIMESTAMP,
    samples_per_hour: int = CLP_SAMPLES_PER_HOUR,
) -> tuple[int, int]:
    """
    訓練に使用するデータ範囲を計算。

    Args:
        total_samples: データセット全体のサンプル数
        num_samples: 使用するサンプル数
        base_timestamp: 基準タイムスタンプ
        samples_per_hour: 1時間あたりのサンプル数

    Returns:
        (start_index, end_index) のタプル
    """
    offset = get_data_offset(base_timestamp, samples_per_hour)

    # オフセットがデータセットを超えた場合は先頭に戻る（循環）
    start = offset % total_samples
    end = start + num_samples

    # 終端がデータセットを超える場合の調整
    if end > total_samples:
        # 簡易実装: 先頭から使用
        start = 0
        end = num_samples

    return start, end


def print_clp_status() -> None:
    """CLPの現在の状態を表示"""
    offset = get_data_offset()
    elapsed_hours = (time.time() - CLP_BASE_TIMESTAMP) / 3600

    print(f"✓ CLP Status:")
    print(f"    Elapsed: {elapsed_hours:.1f} hours since base timestamp")
    print(f"    Data offset: {offset:,} samples")
    print(f"    Senri checkpoint: {'exists' if os.path.exists(SENRI_CHECKPOINT) else 'not found'}")
    print(f"    Pythia checkpoint: {'exists' if os.path.exists(PYTHIA_CHECKPOINT) else 'not found'}")
