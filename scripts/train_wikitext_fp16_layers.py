#!/usr/bin/env python3
"""
WikiText-2 FP16 Training with Variable Number of Layers

Usage:
    python scripts/train_wikitext_fp16_layers.py --num_layers 5
    python scripts/train_wikitext_fp16_layers.py --num_layers 7
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader
from src.utils.config import NewLLML4Config
from src.models.context_vector_llm import ContextVectorLLM
from src.training.wikitext_dataset import load_wikitext_data
from src.training.fp16_trainer import FP16Trainer
from src.utils.train_utils import print_git_info, print_gpu_info
import time
import argparse


class LayerExperimentConfig(NewLLML4Config):
    """Layer experiment configuration

    Fixed parameters:
    - batch_size = 2048 (L4 GPU optimized)
    - learning_rate = 0.0008 (Square Root Scaling)
    - context_vector_dim = 256

    Variable parameter:
    - num_layers (set via __init__)
    """
    # データ関連（WikiText-2用）
    max_seq_length = 64
    vocab_size = 1000

    # モデルアーキテクチャ
    embed_dim = 256
    hidden_dim = 512
    num_layers = 6  # Default, will be overridden in __init__
    context_vector_dim = 256  # Fixed
    dropout = 0.1

    # 訓練ハイパーパラメータ（NewLLML4Configから継承）
    # batch_size = 2048     ← 固定（L4用）
    # learning_rate = 0.0008 ← 固定（Square Root Scaling）
    # device = "cuda"       ← 固定
    num_epochs = 150
    weight_decay = 0.0
    gradient_clip = 1.0

    # Early Stopping
    patience = 30

    # FP16設定
    use_amp = True

    def __init__(self, num_layers=6):
        """Initialize config with specified num_layers"""
        super().__init__()
        self.num_layers = num_layers


def main():
    """FP16混合精度訓練のメイン処理（レイヤー数実験）"""

    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='WikiText-2 FP16 Training with Variable Layers')
    parser.add_argument('--num_layers', type=int, required=True,
                        help='Number of layers (e.g., 5, 6, 7)')
    args = parser.parse_args()

    num_layers = args.num_layers

    print("\n" + "="*80)
    print(f"WikiText-2 FP16 Training - {num_layers} Layers Experiment")
    print("="*80)

    # Git version information
    print_git_info()
    print("="*80)

    # 設定作成
    config = LayerExperimentConfig(num_layers=num_layers)

    # GPU必須チェックと情報表示
    print_gpu_info()
    print(f"   FP16 Mixed Precision: ENABLED ✓")

    print(f"\n🔬 実験設定:")
    print(f"  Model: New-LLM Baseline")
    print(f"  Num Layers: {num_layers} ← Variable")
    print(f"  Context Vector Dim: {config.context_vector_dim} (Fixed)")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch Size: {config.batch_size} (Fixed)")
    print(f"  Learning Rate: {config.learning_rate} (Fixed)")
    print(f"  Precision: FP16 (Mixed)")
    print(f"\n{'='*80}\n")

    # データロード
    print("Loading WikiText-2 dataset...")
    train_dataset, val_dataset, tokenizer = load_wikitext_data(config)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # モデル作成
    print(f"\nCreating New-LLM model with {num_layers} layers...")
    model = ContextVectorLLM(config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # FP16 Trainer作成
    model_name = f"new_llm_wikitext_fp16_layers{num_layers}"
    trainer = FP16Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        model_name=model_name,
        use_amp=config.use_amp
    )

    # 訓練実行
    print(f"\nStarting FP16 training with {num_layers} layers...")
    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print(f"{num_layers} Layers Training Completed!")
    print("="*80)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Checkpoint saved: checkpoints/best_{model_name}.pt")

    # 性能サマリー
    if trainer.val_losses:
        best_val_loss = min(trainer.val_losses)
        best_val_ppl = min(trainer.val_ppls)
        best_epoch = trainer.val_ppls.index(best_val_ppl) + 1

        print(f"\n📊 最終結果:")
        print(f"  Num Layers: {num_layers}")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Best Val Perplexity: {best_val_ppl:.2f} (Epoch {best_epoch})")

        # Baseline (6 layers) との比較
        baseline_ppl = 20.60  # 6 layers, 150 epochs
        diff = best_val_ppl - baseline_ppl
        diff_pct = (diff / baseline_ppl) * 100

        if diff < 0:
            print(f"  vs Baseline (6 layers): {diff:.2f} ({diff_pct:+.1f}%) ✓ BETTER")
        elif diff > 0:
            print(f"  vs Baseline (6 layers): +{diff:.2f} ({diff_pct:+.1f}%)")
        else:
            print(f"  vs Baseline (6 layers): Same")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
