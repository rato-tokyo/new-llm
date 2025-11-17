#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dialogue Fine-tuning for New-LLM

WikiTextで事前学習したモデルをDailyDialogで
対話タスクにファインチューニング。

ステップ:
1. WikiTextで訓練したcheckpointをロード
2. DailyDialogデータでファインチューニング
3. TinyGPT2と比較
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.utils.config import NewLLMConfig
from src.models.context_vector_llm import ContextVectorLLM
from src.models.gpt2_baseline import create_gpt2_baseline
from src.training.dialog_dataset import load_dailydialog_data
from src.training.trainer import Trainer


class DialogConfig(NewLLMConfig):
    """対話ファインチューニング用の設定"""
    # データ関連
    max_seq_length = 64      # 対話は比較的長い
    vocab_size = 1000

    # ファインチューニング用ハイパーパラメータ
    num_epochs = 50          # WikiTextより短く
    batch_size = 16
    learning_rate = 0.00005  # より小さい学習率（既に事前学習済み）
    gradient_clip = 1.0

    # Early Stopping
    patience = 10

    # デバイス
    device = "cpu"


def finetune_new_llm_on_dialog(pretrained_path: str = None):
    """New-LLMを対話データでファインチューニング

    Args:
        pretrained_path: WikiTextで事前学習したcheckpointのパス
                        Noneの場合はスクラッチから訓練
    """
    print("="*80)
    print("Fine-tuning New-LLM on DailyDialog")
    print("="*80)

    config = DialogConfig()

    # データロード
    print("\nLoading DailyDialog dataset...")
    train_dataset, val_dataset, tokenizer = load_dailydialog_data(config)

    # モデル作成
    print("\nCreating New-LLM model...")
    model = ContextVectorLLM(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        context_vector_dim=config.context_vector_dim,
        dropout=config.dropout
    )

    # 事前学習済みモデルをロード（存在すれば）
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"\nLoading pretrained weights from {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Pretrained weights loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load pretrained weights: {e}")
            print("  Starting from scratch...")
    else:
        if pretrained_path:
            print(f"⚠ Warning: Pretrained checkpoint not found at {pretrained_path}")
        print("  Starting from scratch...")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Trainer作成
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        model_name="new_llm_dialog"
    )

    # 訓練実行
    print("\nStarting fine-tuning...")
    history = trainer.train()

    print("\n" + "="*80)
    print("New-LLM Dialog Fine-tuning Completed!")
    print("="*80)
    print(f"Best Val Loss: {min(history['val_loss']):.4f}")
    print(f"Best Val Perplexity: {min(history['val_perplexity']):.2f}")
    print(f"Checkpoint saved: checkpoints/best_new_llm_dialog.pt")

    return history


def finetune_tinygpt2_on_dialog(pretrained_path: str = None):
    """TinyGPT2を対話データでファインチューニング（比較用）"""
    print("="*80)
    print("Fine-tuning TinyGPT2 on DailyDialog (Baseline)")
    print("="*80)

    config = DialogConfig()

    # データロード
    print("\nLoading DailyDialog dataset...")
    train_dataset, val_dataset, tokenizer = load_dailydialog_data(config)

    # TinyGPT2モデル作成
    print("\nCreating TinyGPT2 baseline model...")
    model = create_gpt2_baseline(config, tiny=True)

    # 事前学習済みモデルをロード（存在すれば）
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"\nLoading pretrained weights from {pretrained_path}")
        try:
            checkpoint = torch.load(pretrained_path, map_location=config.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Pretrained weights loaded successfully")
        except Exception as e:
            print(f"⚠ Warning: Could not load pretrained weights: {e}")
            print("  Starting from scratch...")
    else:
        if pretrained_path:
            print(f"⚠ Warning: Pretrained checkpoint not found at {pretrained_path}")
        print("  Starting from scratch...")

    num_params = model.get_num_parameters()
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Trainer作成
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        model_name="tinygpt2_dialog"
    )

    # 訓練実行
    print("\nStarting fine-tuning...")
    history = trainer.train()

    print("\n" + "="*80)
    print("TinyGPT2 Dialog Fine-tuning Completed!")
    print("="*80)
    print(f"Best Val Loss: {min(history['val_loss']):.4f}")
    print(f"Best Val Perplexity: {min(history['val_perplexity']):.2f}")
    print(f"Checkpoint saved: checkpoints/best_tinygpt2_dialog.pt")

    return history


def main():
    """両モデルを対話データでファインチューニングして比較"""
    print("\n" + "="*80)
    print("Dialogue Fine-tuning Experiment")
    print("Comparing New-LLM vs TinyGPT2 on DailyDialog")
    print("="*80)

    # 事前学習済みcheckpointのパス
    new_llm_pretrained = "checkpoints/best_new_llm_wikitext.pt"
    tinygpt2_pretrained = "checkpoints/best_tinygpt2_wikitext.pt"

    # 1. New-LLMファインチューニング
    print("\n[1/2] Fine-tuning New-LLM on DailyDialog...")
    new_llm_history = finetune_new_llm_on_dialog(new_llm_pretrained)

    # 2. TinyGPT2ファインチューニング
    print("\n[2/2] Fine-tuning TinyGPT2 on DailyDialog...")
    tinygpt2_history = finetune_tinygpt2_on_dialog(tinygpt2_pretrained)

    # 結果比較
    print("\n" + "="*80)
    print("FINAL COMPARISON (Dialogue Task)")
    print("="*80)

    new_llm_best_ppl = min(new_llm_history['val_perplexity'])
    tinygpt2_best_ppl = min(tinygpt2_history['val_perplexity'])

    print(f"\nNew-LLM:")
    print(f"  Best Val Perplexity: {new_llm_best_ppl:.2f}")
    print(f"  Best Val Loss: {min(new_llm_history['val_loss']):.4f}")

    print(f"\nTinyGPT2:")
    print(f"  Best Val Perplexity: {tinygpt2_best_ppl:.2f}")
    print(f"  Best Val Loss: {min(tinygpt2_history['val_loss']):.4f}")

    # 差分計算
    ppl_diff = ((new_llm_best_ppl - tinygpt2_best_ppl) / tinygpt2_best_ppl) * 100
    print(f"\nPerplexity Difference: {ppl_diff:+.1f}%")

    if new_llm_best_ppl < tinygpt2_best_ppl:
        print("✓ New-LLM wins on dialogue task!")
    else:
        print("✓ TinyGPT2 wins on dialogue task")

    print("\nNext step: Evaluate generation quality")
    print("Run: python scripts/evaluate_comparison.py")


if __name__ == "__main__":
    main()
