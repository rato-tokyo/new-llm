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
from src.utils.config import NewLLMConfig
from src.models.context_vector_llm import ContextVectorLLM
from src.training.wikitext_dataset import load_wikitext_data
from src.training.trainer import Trainer


class AdvancedConfig(NewLLMConfig):
    """拡張実験用の柔軟な設定クラス

    簡単に変更できるパラメータ:
    - context_vector_dim: コンテキストベクトルの次元数
    - num_layers: レイヤー数
    - quantization_mode: 量子化モード ('none', 'int8')
    """

    # ========================================
    # 実験パラメータ（ここを変更するだけ！）
    # ========================================

    # コンテキストベクトル次元（256, 512, 1024, 2048など）
    context_vector_dim = 512  # デフォルト256の2倍

    # レイヤー数（6, 12, 24, 48など）
    num_layers = 12  # デフォルト6の2倍

    # 量子化モード: 'none', 'int8'
    quantization_mode = 'none'  # 'int8'で有効化

    # ========================================
    # 基本設定（通常は変更不要）
    # ========================================

    # データ関連
    max_seq_length = 64
    vocab_size = 1000

    # モデルアーキテクチャ
    embed_dim = 256
    hidden_dim = 512
    dropout = 0.1

    # 訓練ハイパーパラメータ
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.0001
    weight_decay = 0.0
    gradient_clip = 1.0

    # Early Stopping
    patience = 15

    # デバイス（GPU自動検出）
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
