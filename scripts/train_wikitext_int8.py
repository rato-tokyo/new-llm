#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WikiText-2 Training with INT8 Quantization
INT8量子化実験用訓練スクリプト

実験内容:
1. 通常のfp32で訓練（50 epochs）
2. 訓練後にINT8 dynamic quantizationを適用
3. 量子化前後の性能を比較

目的:
- メモリ削減効果の測定（31MB → 8MB）
- 精度低下の測定（期待: 1-3% PPL上昇）
- 推論速度向上の確認
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
from torch.utils.data import DataLoader
import time


class INT8Config(NewLLMConfig):
    """INT8量子化実験用の設定

    Baselineと同じ設定で訓練し、最後にINT8量子化
    """
    # データ関連
    max_seq_length = 64
    vocab_size = 1000

    # モデルアーキテクチャ（Baselineと同じ）
    embed_dim = 256
    hidden_dim = 512
    num_layers = 6
    context_vector_dim = 256
    dropout = 0.1

    # 訓練ハイパーパラメータ
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.0001
    weight_decay = 0.0
    gradient_clip = 1.0

    # Early Stopping
    patience = 15

    # デバイス
    device = "cpu"  # 量子化はCPUで実施


def measure_model_size(model, model_name="Model"):
    """モデルのメモリサイズを測定"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024**2
    print(f"\n{model_name} size: {size_mb:.2f} MB")
    return size_mb


def apply_int8_quantization(model):
    """訓練済みモデルにINT8 dynamic quantizationを適用

    Dynamic Quantization:
    - 線形層（nn.Linear）のweightsをINT8に変換
    - Activationsは動的に量子化
    - メモリ削減と推論高速化を実現
    """
    print("\n" + "="*60)
    print("Applying INT8 Dynamic Quantization...")
    print("="*60)

    # 量子化前のサイズ測定
    original_size = measure_model_size(model, "Original (fp32)")

    # INT8 dynamic quantization適用
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # 線形層のみ量子化
        dtype=torch.qint8   # INT8
    )

    # 量子化後のサイズ測定
    quantized_size = measure_model_size(quantized_model, "Quantized (int8)")

    # 圧縮率
    compression_ratio = original_size / quantized_size
    print(f"\nCompression ratio: {compression_ratio:.2f}x")
    print(f"Memory reduction: {original_size - quantized_size:.2f} MB")
    print("="*60 + "\n")

    return quantized_model


def evaluate_model(model, val_dataloader, config, model_name="Model"):
    """モデルの性能を評価"""
    from src.evaluation.metrics import compute_loss, compute_perplexity, compute_accuracy

    model.eval()
    model.to(config.device)

    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0

    print(f"\nEvaluating {model_name}...")
    start_time = time.time()

    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs = inputs.to(config.device)
            targets = targets.to(config.device)

            # Forward pass
            outputs = model(inputs)

            # Handle different return types
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Compute metrics
            loss = compute_loss(logits, targets, pad_idx=0)
            accuracy = compute_accuracy(logits, targets, pad_idx=0)

            total_loss += loss.item()
            total_accuracy += accuracy
            total_batches += 1

    avg_loss = total_loss / total_batches
    avg_ppl = compute_perplexity(avg_loss)
    avg_accuracy = total_accuracy / total_batches
    elapsed = time.time() - start_time

    print(f"  Loss: {avg_loss:.4f}")
    print(f"  Perplexity: {avg_ppl:.2f}")
    print(f"  Accuracy: {avg_accuracy*100:.2f}%")
    print(f"  Evaluation time: {elapsed:.2f}s")

    return avg_loss, avg_ppl, avg_accuracy


def main():
    """INT8量子化実験のメイン処理"""
    print("\n" + "="*80)
    print("WikiText-2 Training with INT8 Quantization Experiment")
    print("="*80)

    config = INT8Config()

    print(f"\n実験設定:")
    print(f"  Model: New-LLM Baseline")
    print(f"  Context Vector Dim: {config.context_vector_dim}")
    print(f"  Num Layers: {config.num_layers}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Quantization: INT8 (post-training)")
    print(f"\n{'='*80}\n")

    # ========================================
    # Phase 1: データロード
    # ========================================
    print("Phase 1: Loading WikiText-2 dataset...")
    train_dataset, val_dataset, tokenizer = load_wikitext_data(config)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # ========================================
    # Phase 2: FP32モデルで訓練
    # ========================================
    print("\nPhase 2: Training model in FP32...")
    model = ContextVectorLLM(config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Trainer作成
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        model_name="new_llm_wikitext_int8_fp32"
    )

    # 訓練実行
    print("\nStarting FP32 training...")
    trainer.train()

    # ========================================
    # Phase 3: FP32モデルの評価
    # ========================================
    print("\nPhase 3: Evaluating FP32 model...")
    fp32_loss, fp32_ppl, fp32_acc = evaluate_model(
        model, val_dataloader, config, "FP32 Model"
    )

    # ========================================
    # Phase 4: INT8量子化適用
    # ========================================
    print("\nPhase 4: Applying INT8 quantization...")
    quantized_model = apply_int8_quantization(model)

    # ========================================
    # Phase 5: INT8モデルの評価
    # ========================================
    print("\nPhase 5: Evaluating INT8 quantized model...")
    int8_loss, int8_ppl, int8_acc = evaluate_model(
        quantized_model, val_dataloader, config, "INT8 Model"
    )

    # ========================================
    # Phase 6: 結果比較
    # ========================================
    print("\n" + "="*80)
    print("FINAL COMPARISON: FP32 vs INT8")
    print("="*80)

    print(f"\nFP32 Model:")
    print(f"  Loss: {fp32_loss:.4f}")
    print(f"  Perplexity: {fp32_ppl:.2f}")
    print(f"  Accuracy: {fp32_acc*100:.2f}%")

    print(f"\nINT8 Model:")
    print(f"  Loss: {int8_loss:.4f}")
    print(f"  Perplexity: {int8_ppl:.2f}")
    print(f"  Accuracy: {int8_acc*100:.2f}%")

    # 精度変化
    ppl_change = ((int8_ppl - fp32_ppl) / fp32_ppl) * 100
    loss_change = ((int8_loss - fp32_loss) / fp32_loss) * 100

    print(f"\nAccuracy Impact:")
    print(f"  PPL change: {ppl_change:+.2f}%")
    print(f"  Loss change: {loss_change:+.2f}%")

    # 判定
    if abs(ppl_change) < 3.0:
        print(f"  ✓ INT8 quantization successful! (< 3% degradation)")
    elif abs(ppl_change) < 5.0:
        print(f"  ⚠ Moderate degradation (3-5%)")
    else:
        print(f"  ✗ Significant degradation (> 5%)")

    # INT8モデルの保存
    print(f"\nSaving INT8 quantized model...")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(quantized_model.state_dict(), "checkpoints/new_llm_wikitext_int8.pt")
    print(f"  ✓ Saved to: checkpoints/new_llm_wikitext_int8.pt")

    print("\n" + "="*80)
    print("INT8 Quantization Experiment Completed!")
    print("="*80)


if __name__ == "__main__":
    main()
