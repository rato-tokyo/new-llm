#!/usr/bin/env python3
"""
WikiText-2 Advanced Training - Large Model Experiments

Experiments with larger models:
- Larger context vector dimensions (512, 1024, 2048)
- More layers (12, 24, 48)
- FP16 mixed precision training

Usage:
    python scripts/train_wikitext_advanced.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.utils.config import NewLLMAdvancedL4Config
from src.models.context_vector_llm import ContextVectorLLM
from src.training.wikitext_dataset import load_wikitext_data
from src.training.fp16_trainer import FP16Trainer
from torch.utils.data import DataLoader
import time


class AdvancedConfig(NewLLMAdvancedL4Config):
    """Advanced experiment configuration

    Inherits from NewLLMAdvancedL4Config:
    - batch_size = 2048 (L4 GPU optimized)
    - learning_rate = 0.0004 (Model Size Scaling applied)
    - context_vector_dim = 512
    - num_layers = 12
    - device = "cuda"
    """
    # データ関連（WikiText-2用）
    max_seq_length = 64
    vocab_size = 1000

    # モデルアーキテクチャ（Advancedから継承）
    embed_dim = 256
    hidden_dim = 512
    dropout = 0.1

    # 訓練ハイパーパラメータ
    num_epochs = 150  # Longer training for larger model
    weight_decay = 0.0
    gradient_clip = 1.0
    patience = 30

    # FP16設定
    use_amp = True


def main():
    """Advanced model training with FP16"""
    print("\n" + "="*80)
    print("WikiText-2 Advanced Training (Large Model)")
    print("="*80)

    # Git version information
    try:
        import subprocess
        git_commit_short = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(__file__) + '/..'
        ).decode().strip()
        git_date = subprocess.check_output(
            ['git', 'log', '-1', '--format=%cd', '--date=short'],
            cwd=os.path.dirname(__file__) + '/..'
        ).decode().strip()
        print(f"\n📌 Git Version: {git_commit_short} ({git_date})")
    except Exception:
        print(f"\n📌 Git Version: Unknown")

    print("="*80)

    config = AdvancedConfig()

    # GPU check
    if not torch.cuda.is_available():
        raise RuntimeError("❌ GPU not available! Advanced training requires GPU.")

    # Device info
    print(f"\n🖥️  Device Information:")
    print(f"  Device: CUDA (GPU)")
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {gpu_memory:.1f} GB")
    print(f"  FP16 Mixed Precision: ENABLED ✓")

    print(f"\n🔬 Model Configuration:")
    print(f"  Context Vector Dim: {config.context_vector_dim}")
    print(f"  Num Layers: {config.num_layers}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")

    # Estimate parameters
    # Rough estimate: embed + context_proj + layers + output
    embed_params = config.vocab_size * config.embed_dim
    context_proj_params = (config.embed_dim + config.context_vector_dim) * config.context_vector_dim * 2
    layer_params = config.num_layers * (
        (config.embed_dim + config.context_vector_dim) * config.hidden_dim * 2 +
        config.hidden_dim * config.context_vector_dim
    )
    output_params = config.hidden_dim * config.vocab_size
    total_params = embed_params + context_proj_params + layer_params + output_params
    print(f"  Estimated Parameters: {total_params/1e6:.2f}M")

    print(f"\n{'='*80}\n")

    # Load data
    print("Loading WikiText-2 dataset...")
    train_dataset, val_dataset, tokenizer = load_wikitext_data(config)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    # Create model
    print(f"\nCreating Advanced New-LLM model...")
    print(f"  Context Vector Dim: {config.context_vector_dim}")
    print(f"  Num Layers: {config.num_layers}")
    model = ContextVectorLLM(config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Actual parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Create FP16 trainer
    model_name = f"new_llm_wikitext_ctx{config.context_vector_dim}_layers{config.num_layers}"
    trainer = FP16Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        model_name=model_name,
        use_amp=config.use_amp
    )

    # Train
    print(f"\nStarting Advanced model training...")
    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time

    print("\n" + "="*80)
    print("Advanced Training Completed!")
    print("="*80)
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Checkpoint saved: checkpoints/best_{model_name}.pt")

    # Performance summary
    if trainer.val_ppls:
        best_val_ppl = min(trainer.val_ppls)
        best_epoch = trainer.val_ppls.index(best_val_ppl) + 1

        print(f"\n📊 Final Results:")
        print(f"  Model: ctx{config.context_vector_dim}_layers{config.num_layers}")
        print(f"  Parameters: {num_params/1e6:.2f}M")
        print(f"  Best Val PPL: {best_val_ppl:.2f} (Epoch {best_epoch})")

        # Compare with baseline (6 layers, 256 ctx, 2.74M params, PPL 20.60)
        baseline_ppl = 20.60
        diff = best_val_ppl - baseline_ppl
        diff_pct = (diff / baseline_ppl) * 100

        if diff < 0:
            print(f"  vs Baseline: {diff:.2f} ({diff_pct:+.1f}%) ✓ BETTER")
        elif diff > 0:
            print(f"  vs Baseline: +{diff:.2f} ({diff_pct:+.1f}%)")
        else:
            print(f"  vs Baseline: Same")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
