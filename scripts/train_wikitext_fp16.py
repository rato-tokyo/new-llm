#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WikiText-2 Training with FP16 Mixed Precision
FP16混合精度訓練実験用スクリプト

実験内容:
1. PyTorch AMPを使ったFP16混合精度訓練
2. FP32訓練との速度・精度比較
3. メモリ使用量の測定

FP16の利点:
- 訓練速度: 約2倍高速化（GPU）
- メモリ使用量: 約半分
- 精度低下: ほぼなし（<0.1% PPL差）
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.utils.train_utils import print_git_info, print_gpu_info
from src.utils.config import NewLLML4Config
from src.models.context_vector_llm import ContextVectorLLM
from src.training.wikitext_dataset import load_wikitext_data
from src.training.fp16_trainer import FP16Trainer
from torch.utils.data import DataLoader
import time


class FP16Config(NewLLML4Config):
    """FP16混合精度訓練用の設定（L4 GPU最適化）

    L4 GPU最適化設定（batch_size=2048, device="cuda"）を継承
    WikiText-2用にmax_seq_lengthなどを調整
    """
    # データ関連（WikiText-2用）
    max_seq_length = 64
    vocab_size = 1000

    # モデルアーキテクチャ（Baseline）
    embed_dim = 256
    hidden_dim = 512
    num_layers = 6
    context_vector_dim = 256
    dropout = 0.1

    # 訓練ハイパーパラメータ（NewLLML4Configから継承）
    # batch_size = 2048     ← NewLLML4Configから自動継承（L4用）
    # learning_rate = 0.0008 ← NewLLML4Configから自動継承（Square Root Scaling適用済み）
    # device = "cuda"       ← NewLLML4Configから自動継承
    num_epochs = 50
    weight_decay = 0.0
    gradient_clip = 1.0

    # Early Stopping
    patience = 15

    # FP16設定
    use_amp = True  # Automatic Mixed Precision (GPU必須)


def main():
    """FP16混合精度訓練のメイン処理"""
    print("\n" + "="*80)
    print("WikiText-2 Training with FP16 Mixed Precision")
    print("="*80)

    # Git version information
    try:
        import subprocess
        git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(__file__) + '/..').decode().strip()
        git_commit_short = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=os.path.dirname(__file__) + '/..').decode().strip()
        git_date = subprocess.check_output(['git', 'log', '-1', '--format=%cd', '--date=short'], cwd=os.path.dirname(__file__) + '/..').decode().strip()
        print(f"\n📌 Git Version: {git_commit_short} ({git_date})")
        print(f"   Full commit: {git_commit}")
    except Exception:
        print(f"\n📌 Git Version: Unknown (not a git repository)")

    print("="*80)

    config = FP16Config()

    # GPU必須チェック
    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU not available! FP16 training requires CUDA GPU.")

    # デバイス情報表示
    print(f"\n🖥️  Device Information:")
    print(f"  Device: CUDA (GPU)")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {gpu_memory:.1f} GB")
    print(f"  FP16 Mixed Precision: ENABLED ✓")
    print(f"  ⚡ Expected speedup: 2x faster than FP32")
    print(f"  💾 Expected memory saving: ~50%")

    print(f"\n実験設定:")
    print(f"  Model: New-LLM Baseline")
    print(f"  Context Vector Dim: {config.context_vector_dim}")
    print(f"  Num Layers: {config.num_layers}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Precision: FP16 (Mixed)")
    print(f"\n{'='*80}\n")

    # データロード
    print("Loading WikiText-2 dataset...")
    train_dataset, val_dataset, tokenizer = load_wikitext_data(config)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # モデル作成
    print("\nCreating New-LLM model...")
    model = ContextVectorLLM(config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # FP16 Trainer作成
    trainer = FP16Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        model_name="new_llm_wikitext_fp16",
        use_amp=config.use_amp
    )

    # 訓練実行
    print("\nStarting FP16 training...")
    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("FP16 Mixed Precision Training Completed!")
    print("="*80)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Checkpoint saved: checkpoints/best_new_llm_wikitext_fp16.pt")

    # 性能サマリー
    if trainer.val_losses:
        best_val_loss = min(trainer.val_losses)
        best_val_ppl = min(trainer.val_ppls)
        print(f"\n最終結果:")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Best Val Perplexity: {best_val_ppl:.2f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
