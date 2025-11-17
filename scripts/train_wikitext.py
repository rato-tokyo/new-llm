#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WikiText-2 Pre-training for New-LLM

WikiTextは実際のWikipedia記事から抽出されたデータで、
ランダム生成データより自然な文章の流れを学習できる。

実験:
1. New-LLM + WikiText-2
2. TinyGPT2 + WikiText-2
3. 両者の比較（Perplexity, 生成品質）
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.utils.config import NewLLMConfig
from src.models.context_vector_llm import ContextVectorLLM
from src.models.gpt2_baseline import create_gpt2_baseline
from src.training.wikitext_dataset import load_wikitext_data
from src.training.trainer import Trainer


class WikiTextConfig(NewLLMConfig):
    """WikiText事前学習用の設定

    ランダム生成データ (500文) → WikiText (数千文) なので
    より長い訓練が必要。
    """
    # データ関連
    max_seq_length = 64      # WikiTextは長いので拡張
    vocab_size = 1000        # 維持（メモリ制約）

    # 訓練ハイパーパラメータ
    num_epochs = 100         # WikiTextは大きいので長く訓練
    batch_size = 16
    learning_rate = 0.0001
    gradient_clip = 1.0

    # Early Stopping（過学習防止）
    patience = 15            # バリデーションロスが改善しなければ停止

    # デバイス
    device = "cpu"


def train_new_llm_on_wikitext():
    """New-LLMをWikiTextで事前学習"""
    print("="*80)
    print("Training New-LLM on WikiText-2")
    print("="*80)

    config = WikiTextConfig()

    # データロード
    print("\nLoading WikiText-2 dataset...")
    train_dataset, val_dataset, tokenizer = load_wikitext_data(config)

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

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Trainer作成
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        model_name="new_llm_wikitext"
    )

    # 訓練実行
    print("\nStarting training...")
    history = trainer.train()

    print("\n" + "="*80)
    print("New-LLM WikiText Training Completed!")
    print("="*80)
    print(f"Best Val Loss: {min(history['val_loss']):.4f}")
    print(f"Best Val Perplexity: {min(history['val_perplexity']):.2f}")
    print(f"Checkpoint saved: checkpoints/best_new_llm_wikitext.pt")

    return history


def train_tinygpt2_on_wikitext():
    """TinyGPT2をWikiTextで事前学習（比較用）"""
    print("="*80)
    print("Training TinyGPT2 on WikiText-2 (Baseline)")
    print("="*80)

    config = WikiTextConfig()

    # データロード
    print("\nLoading WikiText-2 dataset...")
    train_dataset, val_dataset, tokenizer = load_wikitext_data(config)

    # TinyGPT2モデル作成
    print("\nCreating TinyGPT2 baseline model...")
    model = create_gpt2_baseline(config, tiny=True)

    num_params = model.get_num_parameters()
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Trainer作成
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        model_name="tinygpt2_wikitext"
    )

    # 訓練実行
    print("\nStarting training...")
    history = trainer.train()

    print("\n" + "="*80)
    print("TinyGPT2 WikiText Training Completed!")
    print("="*80)
    print(f"Best Val Loss: {min(history['val_loss']):.4f}")
    print(f"Best Val Perplexity: {min(history['val_perplexity']):.2f}")
    print(f"Checkpoint saved: checkpoints/best_tinygpt2_wikitext.pt")

    return history


def main():
    """両モデルを訓練して比較"""
    print("\n" + "="*80)
    print("WikiText-2 Pre-training Experiment")
    print("Comparing New-LLM vs TinyGPT2")
    print("="*80)

    # 1. New-LLM訓練
    print("\n[1/2] Training New-LLM...")
    new_llm_history = train_new_llm_on_wikitext()

    # 2. TinyGPT2訓練
    print("\n[2/2] Training TinyGPT2 (baseline)...")
    tinygpt2_history = train_tinygpt2_on_wikitext()

    # 結果比較
    print("\n" + "="*80)
    print("FINAL COMPARISON")
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
        print("✓ New-LLM wins!")
    else:
        print("✓ TinyGPT2 wins")

    print("\nNext step: Fine-tune on dialogue data (DailyDialog)")
    print("Run: python scripts/finetune_dialog.py")


if __name__ == "__main__":
    main()
