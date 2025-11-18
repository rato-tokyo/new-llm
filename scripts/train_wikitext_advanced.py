#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WikiText-2 Advanced Training - 拡張実験用

実験内容:
1. コンテキストベクトル次元の拡張（256 → 512, 1024など）
2. レイヤー数の柔軟な変更（6 → 12, 24など）
3. int8量子化サポート（オプション）

使い方:
1. AdvancedConfigクラスで設定を変更
2. python scripts/train_wikitext_advanced.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.quantization
from src.utils.config import NewLLMAdvancedGPUConfig
from src.models.context_vector_llm import ContextVectorLLM
from src.training.wikitext_dataset import load_wikitext_data
from src.training.trainer import Trainer


class AdvancedConfig(NewLLMAdvancedGPUConfig):
    """拡張実験用の柔軟な設定クラス

    GPU最適化設定（batch_size=512, device="cuda", context_vector_dim=512, num_layers=12）を継承

    簡単に変更できるパラメータ:
    - context_vector_dim: コンテキストベクトルの次元数（512, 1024, 2048など）
    - num_layers: レイヤー数（12, 24, 48など）
    - quantization_mode: 量子化モード ('none', 'int8')
    """

    # ========================================
    # 実験パラメータ（ここを変更するだけ！）
    # ========================================

    # コンテキストベクトル次元 - NewLLMAdvancedGPUConfigから継承
    # context_vector_dim = 512  ← 自動継承（変更したい場合のみ上書き）

    # レイヤー数 - NewLLMAdvancedGPUConfigから継承
    # num_layers = 12  ← 自動継承（変更したい場合のみ上書き）

    # 量子化モード: 'none', 'int8'
    quantization_mode = 'none'  # 'int8'で有効化

    # ========================================
    # 基本設定（通常は変更不要）
    # ========================================

    # データ関連（WikiText-2用）
    max_seq_length = 64
    vocab_size = 1000

    # モデルアーキテクチャ
    embed_dim = 256
    hidden_dim = 512
    dropout = 0.1

    # 訓練ハイパーパラメータ（NewLLMAdvancedGPUConfigから継承）
    # batch_size = 2048  ← NewLLMAdvancedL4Configから自動継承
    # device = "cuda"    ← NewLLMAdvancedL4Configから自動継承
    num_epochs = 50
    learning_rate = 0.0004  # Linear Scaling Rule: batch_size 4x → LR 4x
    weight_decay = 0.0
    gradient_clip = 1.0

    # Early Stopping
    patience = 15

    def get_experiment_name(self):
        """実験名を自動生成"""
        name = f"new_llm_wikitext"
        name += f"_ctx{self.context_vector_dim}"
        name += f"_layers{self.num_layers}"
        if self.quantization_mode != 'none':
            name += f"_{self.quantization_mode}"
        return name


def apply_quantization(model, mode='int8'):
    """量子化を適用

    Args:
        model: 量子化するモデル
        mode: 'int8' or 'none'

    Returns:
        量子化されたモデル
    """
    if mode == 'none':
        return model

    elif mode == 'int8':
        print(f"\n{'='*60}")
        print(f"Applying int8 quantization...")
        print(f"{'='*60}")

        # int8量子化（動的量子化 - 推論時のメモリ削減）
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},  # 線形層のみ量子化
            dtype=torch.qint8   # int8
        )

        # パラメータ数比較
        original_size = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)  # MB
        quantized_size = sum(p.numel() for p in quantized_model.parameters()) * 1 / (1024**2)  # MB (int8 = 1 byte)

        print(f"Original model size: {original_size:.2f} MB (fp32)")
        print(f"Quantized model size: {quantized_size:.2f} MB (int8)")
        print(f"Compression ratio: {original_size/quantized_size:.2f}x")

        return quantized_model

    else:
        raise ValueError(f"Unknown quantization mode: {mode}")


def train_new_llm_advanced():
    """拡張実験でNew-LLMを訓練"""

    config = AdvancedConfig()
    experiment_name = config.get_experiment_name()

    print("="*80)
    print("Advanced WikiText-2 Training Experiment")
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

    # GPU/CPU情報を明示的に表示
    print(f"\n🖥️  Device Information:")
    print(f"  Device: {config.device.upper()}")
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {gpu_memory:.1f} GB")
        print(f"  Batch Size: {config.batch_size} (optimized for GPU RAM)")

        # 予想GPU RAM使用量
        model_params = 4.84  # 4.84M params
        estimated_usage = model_params * 0.004 * config.batch_size / 32  # rough estimate
        print(f"  Estimated GPU RAM usage: {estimated_usage:.1f} GB ({estimated_usage/gpu_memory*100:.0f}%)")
        print(f"  ⚡ GPU acceleration ENABLED - Maximum performance mode")

        if estimated_usage < gpu_memory * 0.5:
            print(f"  💡 TIP: GPU RAM underutilized. Can increase batch_size to {config.batch_size * 2}")
    else:
        print(f"  ⚠️  WARNING: Running on CPU (will be VERY SLOW)")
        print(f"  💡 Solution: Runtime → Change runtime type → GPU (T4)")

    print(f"\n実験設定:")
    print(f"  Context Vector Dim: {config.context_vector_dim}")
    print(f"  Number of Layers: {config.num_layers}")
    print(f"  Quantization: {config.quantization_mode}")
    print(f"  Experiment Name: {experiment_name}")
    print(f"\n{'='*80}\n")

    # データロード
    print("Loading WikiText-2 dataset...")
    train_dataset, val_dataset, tokenizer = load_wikitext_data(config)

    # モデル作成
    print("\nCreating New-LLM model...")
    model = ContextVectorLLM(config)

    # パラメータ数表示
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # 量子化適用（オプション）
    if config.quantization_mode != 'none':
        model = apply_quantization(model, config.quantization_mode)

    # DataLoader作成
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Trainer作成
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        model_name=experiment_name
    )

    # 訓練実行
    print("\nStarting training...")
    trainer.train()

    print("\n" + "="*80)
    print("Advanced Training Completed!")
    print("="*80)
    print(f"Checkpoint saved: checkpoints/best_{experiment_name}.pt")

    return trainer


def main():
    """メイン実行"""
    print("\n" + "="*80)
    print("New-LLM Advanced Training Experiment")
    print("="*80)

    # 設定確認プロンプト
    config = AdvancedConfig()
    print(f"\n現在の設定:")
    print(f"  Context Vector Dim: {config.context_vector_dim}")
    print(f"  Number of Layers: {config.num_layers}")
    print(f"  Quantization: {config.quantization_mode}")

    # 実行
    trainer = train_new_llm_advanced()

    # 結果サマリー
    if trainer.val_losses:
        best_val_loss = min(trainer.val_losses)
        best_val_ppl = min(trainer.val_ppls)
        print(f"\n最終結果:")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Best Val Perplexity: {best_val_ppl:.2f}")


if __name__ == "__main__":
    main()
